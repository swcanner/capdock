import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import random
from torch.utils import data
from torch_geometric.loader import DataLoader
from omegaconf import DictConfig
from src.models.components.egnn_net import EGNN_Net
from src.utils.so3_diffuser import SO3Diffuser 
from src.utils.r3_diffuser import R3Diffuser 
from src.utils.geometry import axis_angle_to_matrix, axis_angle_to_rotation_6d
from scipy.spatial.transform import Rotation 
from src.data.docking_dataset import DockingDataset


class EGNN_Model(pl.LightningModule):
    def __init__(
        self,
        model,
        diffuser,
        experiment,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = experiment.lr
        self.weight_decay = experiment.weight_decay

        # if calculate gradient of energy
        self.grad_energy = experiment.grad_energy
        self.separate_ec_loss = experiment.separate_ec_loss
        
        # if penalize clash
        self.penalize_clash = experiment.penalize_clash

        # translation
        self.separate_tr_loss = experiment.separate_tr_loss
        self.perturb_tr = experiment.perturb_tr

        # rotation
        self.separate_rot_loss = experiment.separate_rot_loss
        self.perturb_rot = experiment.perturb_rot

        # diffuser
        if self.perturb_tr:
            self.r3_diffuser = R3Diffuser(diffuser.r3)
        if self.perturb_rot:
            self.so3_diffuser = SO3Diffuser(diffuser.so3)

        # net
        self.net = EGNN_Net(model)
    
    def forward(self, rec_x, lig_x, rec_pos, lig_pos, t):
        # move the lig center to origin
        center = lig_pos[..., 1, :].mean(dim=0)
        rec_pos -= center
        lig_pos -= center

        # contact matrix
        n1 = rec_pos.size(0)
        n2 = lig_pos.size(0)
        contact_matrix = torch.zeros((n1 + n2, n1 + n2), device=self.device)

        # predict
        energy, fa_rep, f, tr_score, rot_score = self.net(rec_x, lig_x, rec_pos, lig_pos, t, contact_matrix, predict=True)

        return energy, fa_rep, f, tr_score, rot_score

    def loss_fn(self, rec_x, lig_x, rec_pos, lig_pos, eps=1e-5):
        with torch.no_grad():
            # uniformly sample a timestep
            t = torch.rand(1, device=self.device) * (1. - eps) + eps

            # random rotation augmentation
            rec_pos, lig_pos = self.random_rotation(rec_pos, lig_pos)

            # sample perturbation for translation and rotation
            if self.perturb_tr:
                tr_score_scale = self.r3_diffuser.score_scaling(t.item())
                tr_update, tr_score_gt = self.r3_diffuser.forward_marginal(t.item())
                tr_update = torch.from_numpy(tr_update).float().to(self.device)
                tr_score_gt = torch.from_numpy(tr_score_gt).float().to(self.device)
            else:
                tr_update = np.zeros(3)
                tr_update = torch.from_numpy(tr_update).float().to(self.device)

            if self.perturb_rot:
                rot_score_scale = self.so3_diffuser.score_scaling(t.item())
                rot_update, rot_score_gt = self.so3_diffuser.forward_marginal(t.item())
                rot_update = torch.from_numpy(rot_update).float().to(self.device)
                rot_score_gt = torch.from_numpy(rot_score_gt).float().to(self.device)
            else:
                rot_update = np.zeros(3)
                rot_update = torch.from_numpy(rot_update).float().to(self.device)

            # get sampled contact matrix
            contact_matrix = self.get_sampled_contact_matrix(
                rec_pos[..., 1, :], lig_pos[..., 1, :], num_samples=random.randint(0, 3))

            # update poses
            lig_pos = self.modify_coords(lig_pos, rot_update, tr_update)
            
            # move the lig center to origin
            center = lig_pos[..., 1, :].mean(dim=0)
            rec_pos = rec_pos - center
            lig_pos = lig_pos - center

        # predict score based on the current state
        if self.grad_energy:
            energy, dedx, f, tr_score, rot_score = self.net(rec_x, lig_x, rec_pos, lig_pos, t, contact_matrix)
            # energy conservation loss
            if self.separate_ec_loss:
                f_angle = torch.norm(f, dim=-1, keepdim=True)
                f_axis = f / (f_angle + 1e-6)

                dedx_angle = torch.norm(dedx, dim=-1, keepdim=True)
                dedx_axis = dedx / (dedx_angle + 1e-6)

                ec_axis_loss = torch.mean((f_axis - dedx_axis)**2)
                #ec_axis_loss = 1.0 - torch.mean(torch.sum(f_axis * dedx_axis, dim=-1))
                ec_angle_loss = torch.mean((f_angle - dedx_angle)**2)
                ec_loss = 0.5 * (ec_axis_loss + ec_angle_loss)
                
            else:
                ec_loss = torch.mean((dedx - f)**2)
        else:
            energy, fa_rep, f, tr_score, rot_score = self.net(rec_x, lig_x, rec_pos, lig_pos, t, contact_matrix, predict=True)
            # energy conservation loss
            ec_loss = torch.tensor(0.0, device=self.device)
        
        # calculate losses
        if self.perturb_tr:
            if self.separate_tr_loss:
                gt_tr_angle = torch.norm(tr_score_gt, dim=-1, keepdim=True)
                gt_tr_axis = tr_score_gt / (gt_tr_angle + 1e-6)

                pred_tr_angle = torch.norm(tr_score, dim=-1, keepdim=True)
                pred_tr_axis = tr_score / (pred_tr_angle + 1e-6)

                tr_axis_loss = torch.mean((pred_tr_axis - gt_tr_axis)**2)
                #tr_axis_loss = 1.0 - torch.mean(torch.sum(pred_tr_axis * gt_tr_axis, dim=-1))
                tr_angle_loss = torch.mean((pred_tr_angle - gt_tr_angle)**2 / tr_score_scale**2)
                tr_loss = 0.5 * (tr_axis_loss + tr_angle_loss)

            else:
                tr_loss = torch.mean((tr_score - tr_score_gt)**2 / tr_score_scale**2)
        else:
            tr_loss = torch.tensor(0.0, device=self.device)

        if self.perturb_rot:
            if self.separate_rot_loss:
                gt_rot_angle = torch.norm(rot_score_gt, dim=-1, keepdim=True)
                gt_rot_axis = rot_score_gt / (gt_rot_angle + 1e-6)

                pred_rot_angle = torch.norm(rot_score, dim=-1, keepdim=True)
                pred_rot_axis = rot_score / (pred_rot_angle + 1e-6)

                rot_axis_loss = torch.mean((pred_rot_axis - gt_rot_axis)**2)
                #rot_axis_loss = 1.0 - torch.mean(torch.sum(pred_rot_axis * gt_rot_axis, dim=-1))
                rot_angle_loss = torch.mean((pred_rot_angle - gt_rot_angle)**2 / rot_score_scale**2)
                rot_loss = 0.5 * (rot_axis_loss + rot_angle_loss)

            else:
                rot_loss = torch.mean((rot_score - rot_score_gt)**2 / rot_score_scale**2)
        else:
            rot_loss = torch.tensor(0.0, device=self.device)

        if self.penalize_clash:
            # get updated pose
            tr_pred = tr_score / tr_score_scale**2 
            rot_pred = rot_score / rot_score_scale**2
            lig_pos_pred = self.modify_coords(lig_pos, rot_pred, tr_pred)
            D = torch.norm((rec_pos[:, None, 1, :] - lig_pos_pred[None, :, 1, :]), dim=-1)
            mask = (D < 3.0).float()
            clash_loss = ((3.0 - D) * mask).sum() / (mask.sum() + 1e-6) 
            clash_loss *= (1.0 - t.item())
        else:
            clash_loss = torch.tensor(0.0, device=self.device)

        # total losses
        loss = tr_loss + rot_loss + ec_loss + clash_loss
        losses = {"tr_loss": tr_loss, "rot_loss": rot_loss, "ec_loss": ec_loss, "clash_loss": clash_loss, "loss": loss}

        return losses

    def modify_coords(self, lig_pos, rot_update, tr_update):
        lig_cen = lig_pos[..., 1, :].mean(dim=0)
        rot = axis_angle_to_matrix(rot_update.squeeze())
        tr = tr_update.squeeze()
        lig_pos = (lig_pos - lig_cen) @ rot.T + lig_cen + tr
        return lig_pos

    def random_rotation(self, rec_pos, lig_pos):
        rot = torch.from_numpy(Rotation.random().as_matrix()).float().to(self.device)
        pos = torch.cat([rec_pos, lig_pos], dim=0)
        cen = pos[..., 1, :].mean(dim=0)
        pos = (pos - cen) @ rot.T
        rec_pos_out = pos[:rec_pos.size(0)]
        lig_pos_out = pos[rec_pos.size(0):]
        return rec_pos_out, lig_pos_out
    
    def get_sampled_contact_matrix(self, set1, set2, threshold=8.0, num_samples=None):
        """
        Constructs a contact matrix for two sets of residues with 1 indicating sampled contact pairs.
        
        :param set1: PyTorch tensor of shape [n1, 3] for residues in set 1
        :param set2: PyTorch tensor of shape [n2, 3] for residues in set 2
        :param threshold: Distance threshold to define contact residues
        :param num_samples: Number of contact pairs to sample. If None, use all valid contacts.
        :return: PyTorch tensor of shape [(n1+n2), (n1+n2)] representing the contact matrix with sampled contact pairs
        """
        n1 = set1.size(0)
        n2 = set2.size(0)
        
        # Compute the pairwise distances between set1 and set2
        dists = torch.cdist(set1, set2)
        
        # Find pairs where distances are less than or equal to the threshold
        contact_pairs = (dists <= threshold)
        
        # Get indices of valid contact pairs
        contact_indices = contact_pairs.nonzero(as_tuple=False)
        
        # Initialize the contact matrix with zeros
        contact_matrix = torch.zeros((n1 + n2, n1 + n2), device=self.device)

        # Determine the number of samples
        if num_samples is None or num_samples > contact_indices.size(0):
            num_samples = contact_indices.size(0)
        
        if num_samples > 0:
            # Sample contact indices uniformly
            sampled_indices = contact_indices[torch.randint(0, contact_indices.size(0), (num_samples,))]
            
            # Fill in the contact matrix for the sampled contacts
            contact_matrix[sampled_indices[:, 0], sampled_indices[:, 1] + n1] = 1.0
            contact_matrix[sampled_indices[:, 1] + n1, sampled_indices[:, 0]] = 1.0
        
        return contact_matrix

    
    def step(self, batch, batch_idx):
        rec_x = batch['rec_x'].squeeze(0)
        lig_x = batch['lig_x'].squeeze(0)
        rec_pos = batch['rec_pos'].squeeze(0)
        lig_pos = batch['lig_pos'].squeeze(0)

        # get losses
        losses = self.loss_fn(rec_x, lig_x, rec_pos, lig_pos)
        return losses

    def training_step(self, batch, batch_idx):
        losses = self.step(batch, batch_idx)
        for loss_name, indiv_loss in losses.items():
            self.log(
                f"train/{loss_name}", 
                indiv_loss, 
                batch_size=1,
            )
        return losses["loss"]

    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)
    
    def on_validation_model_train(self, *args, **kwargs):
        super().on_validation_model_train(*args, **kwargs)
        torch.set_grad_enabled(True)

    def validation_step(self, batch, batch_idx):
        losses = self.step(batch, batch_idx)
        for loss_name, indiv_loss in losses.items():
            self.log(
                f"val/{loss_name}", 
                indiv_loss, 
                batch_size=1,
            )
        return losses["loss"]

    def test_step(self, batch, batch_idx):
        losses = self.step(batch, batch_idx)
        for loss_name, indiv_loss in losses.items():
            self.log(
                f"test/{loss_name}", 
                indiv_loss, 
                batch_size=1,
            )
        return losses["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        return optimizer


@hydra.main(version_base=None, config_path="/home/lchu11/scr4_jgray21/lchu11/graylab_repos/Dock_Diffusion/configs/model", config_name="egnn_model.yaml")
def main(conf: DictConfig):
    dataset = DockingDataset(dataset='dips_train_hetero')

    subset_indices = [0]
    subset = data.Subset(dataset, subset_indices)

    #load dataset
    dataloader = DataLoader(subset)
    
    model = EGNN_Model(
        model=conf.model, 
        diffuser=conf.diffuser,
        experiment=conf.experiment
    )
    trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=10, inference_mode=False)
    trainer.validate(model, dataloader)

if __name__ == '__main__':
    main()


    

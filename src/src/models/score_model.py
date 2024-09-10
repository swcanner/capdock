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
from models.score_net import Tor_Net
from utils.so3_diffuser import SO3Diffuser 
from utils.tor_diffuser import TorDiffuser 
from utils.r3_diffuser import R3Diffuser 
from utils.geometry import axis_angle_to_matrix
from carb_utils import *
#from data.docking_dataset import DockingDataset

#----------------------------------------------------------------------------
# Main wrapper for training the model

class Score_Model(pl.LightningModule):
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

        # energy
        self.grad_energy = experiment.grad_energy
        self.separate_energy_loss = experiment.separate_energy_loss
        
        # translation
        self.perturb_tr = experiment.perturb_tr
        self.separate_tr_loss = experiment.separate_tr_loss

        # rotation
        self.perturb_rot = experiment.perturb_rot
        self.separate_rot_loss = experiment.separate_rot_loss

        # diffuser
        if self.perturb_tr:
            self.r3_diffuser = R3Diffuser(diffuser.r3)
        if self.perturb_rot:
            self.so3_diffuser = SO3Diffuser(diffuser.so3)

        # net
        self.net = Score_Net(model)
    
    def forward(self, rec_x, lig_x, rec_pos, lig_pos, contact_matrix, t):
        # move the lig center to origin
        center = lig_pos[..., 1, :].mean(dim=0)
        rec_pos -= center
        lig_pos -= center

        # predict
        energy, fa_rep, f, tr_score, rot_score = self.net(rec_x, lig_x, rec_pos, lig_pos, t, contact_matrix, predict=True)

        return energy, fa_rep, f, tr_score, rot_score

    def loss_fn(self, rec_x, lig_x, rec_pos, lig_pos, contact_matrix, position_matrix, eps=1e-5):
        with torch.no_grad():
            # uniformly sample a timestep
            t = torch.rand(1, device=self.device) * (1. - eps) + eps

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

            # update poses
            lig_pos = self.modify_coords(lig_pos, rot_update, tr_update)
            
            # move the lig center to origin
            center = lig_pos[..., 1, :].mean(dim=0)
            rec_pos = rec_pos - center
            lig_pos = lig_pos - center

        # predict score based on the current state
        if self.grad_energy:
            energy, dedx, f, tr_score, rot_score = self.net(rec_x, lig_x, rec_pos, lig_pos, t, contact_matrix, position_matrix)
            # energy conservation loss
            if self.separate_energy_loss:
                f_angle = torch.norm(f, dim=-1, keepdim=True)
                f_axis = f / (f_angle + 1e-6)

                dedx_angle = torch.norm(dedx, dim=-1, keepdim=True)
                dedx_axis = dedx / (dedx_angle + 1e-6)

                ec_axis_loss = torch.mean((f_axis - dedx_axis)**2)
                ec_angle_loss = torch.mean((f_angle - dedx_angle)**2)
                ec_loss = 0.5 * (ec_axis_loss + ec_angle_loss)
                
            else:
                ec_loss = torch.mean((dedx - f)**2)
        else:
            energy, fa_rep, f, tr_score, rot_score = self.net(rec_x, lig_x, rec_pos, lig_pos, t, contact_matrix, position_matrix, predict=True)
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
                rot_angle_loss = torch.mean((pred_rot_angle - gt_rot_angle)**2 / rot_score_scale**2)
                rot_loss = 0.5 * (rot_axis_loss + rot_angle_loss)

            else:
                rot_loss = torch.mean((rot_score - rot_score_gt)**2 / rot_score_scale**2)
        else:
            rot_loss = torch.tensor(0.0, device=self.device)

        # total losses
        loss = tr_loss + rot_loss + ec_loss
        losses = {"tr_loss": tr_loss, "rot_loss": rot_loss, "ec_loss": ec_loss, "loss": loss}

        return losses

    def modify_coords(self, lig_pos, rot_update, tr_update):
        lig_cen = lig_pos[..., 1, :].mean(dim=0)
        rot = axis_angle_to_matrix(rot_update.squeeze())
        tr = tr_update.squeeze()
        lig_pos = (lig_pos - lig_cen) @ rot.T + lig_cen + tr
        return lig_pos

    def step(self, batch, batch_idx):
        rec_x = batch['rec_x'].squeeze(0)
        lig_x = batch['lig_x'].squeeze(0)
        rec_pos = batch['rec_pos'].squeeze(0)
        lig_pos = batch['lig_pos'].squeeze(0)
        contact_matrix = batch['contact_matrix'].squeeze(0)
        position_matrix = batch['position_matrix'].squeeze(0)

        # get losses
        losses = self.loss_fn(rec_x, lig_x, rec_pos, lig_pos, contact_matrix, position_matrix)
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


class Tor_Model(pl.LightningModule):
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

        # energy
        self.grad_energy = experiment.grad_energy
        self.separate_energy_loss = experiment.separate_energy_loss
        
        # translation
        self.perturb_tr = experiment.perturb_tr
        self.separate_tr_loss = experiment.separate_tr_loss

        # rotation
        self.perturb_rot = experiment.perturb_rot
        self.separate_rot_loss = experiment.separate_rot_loss

        # torsion
        self.perturb_tor = experiment.perturb_tor
        self.separate_tor_loss = experiment.separate_tor_loss

        # diffuser
        if self.perturb_tr:
            self.r3_diffuser = R3Diffuser(diffuser.r3)
        if self.perturb_rot:
            self.so3_diffuser = SO3Diffuser(diffuser.so3)
        if self.perturb_tor:
            self.tor_diffuser = TorDiffuser(diffuser.tor)

        # net
        self.net = Tor_Net(model)
    
    def forward(self, polymer, contact_matrix, t):
        # move the lig center to origin
        #center = lig_pos[..., 1, :].mean(dim=0)
        #rec_pos -= center
        #lig_pos -= center

        # predict
        tor_pred = self.net(polymer, t, contact_matrix, predict=True)

        return tor_pred

    def loss_fn(self, polymer, eps=1e-5):
        with torch.no_grad():
            # uniformly sample a timestep
            t = torch.rand(1, device=self.device) * (1. - eps) + eps

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

            if self.perturb_tor:
                tor_score_scale = self.tor_diffuser.score_scaling(t.item())
                print('tor_scale:',t.item())
                tor_update, tor_score_gt = self.tor_diffuser.forward_marginal(t.item())
                tor_update = torch.from_numpy(tor_update).float().to(self.device)
                tor_score_gt = torch.from_numpy(tor_score_gt).float().to(self.device)
            else:
                tor_update = np.zeros(3)
                tor_update = torch.from_numpy(tor_update).float().to(self.device)

            # update poses
            polymer = self.modify_coords(polymer, tr_update, rot_update, tor_update)
            
            
            # move the lig center to origin
            #center = lig_pos[..., 1, :].mean(dim=0)
            #rec_pos = rec_pos - center
            #lig_pos = lig_pos - center

        # predict score based on the current state
        tor_score = self.net(polymer, t)

        
        # calculate losses
        if self.perturb_tr:
            if self.separate_tr_loss:
                gt_tr_angle = torch.norm(tr_score_gt, dim=-1, keepdim=True)
                gt_tr_axis = tr_score_gt / (gt_tr_angle + 1e-6)

                pred_tr_angle = torch.norm(tr_score, dim=-1, keepdim=True)
                pred_tr_axis = tr_score / (pred_tr_angle + 1e-6)

                tr_axis_loss = torch.mean((pred_tr_axis - gt_tr_axis)**2)
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
                rot_angle_loss = torch.mean((pred_rot_angle - gt_rot_angle)**2 / rot_score_scale**2)
                rot_loss = 0.5 * (rot_axis_loss + rot_angle_loss)

            else:
                rot_loss = torch.mean((rot_score - rot_score_gt)**2 / rot_score_scale**2)
        else:
            rot_loss = torch.tensor(0.0, device=self.device)

        if self.perturb_tor:
            if self.separate_rot_loss:
                gt_tor_angle = torch.norm(tor_score_gt, dim=-1, keepdim=True)
                #gt_tor_axis = tor_score_gt / (gt_tor_angle + 1e-6)

                pred_tor_angle = torch.norm(tor_score, dim=-1, keepdim=True)
                #pred_rot_axis = rot_score / (pred_rot_angle + 1e-6)

                #rot_axis_loss = torch.mean((pred_rot_axis - gt_rot_axis)**2)
                tor_angle_loss = torch.mean((pred_tor_angle - gt_tor_angle)**2 / tor_score_scale**2)
                tor_loss = tor_angle_loss

            else:
                tor_loss = torch.mean((tor_score - gt_tor_angle)**2 / tor_score_scale**2)
        else:
            tor_loss = torch.tensor(0.0, device=self.device)

        # total losses
        loss = tr_loss + rot_loss + tor_loss
        losses = {"tr_loss": tr_loss, "rot_loss": rot_loss, "tor_loss": tor_loss, "loss": loss}

        return losses

    def modify_coords(self, polymer, d_x, d_rot, d_tor):
        print(d_x, d_rot, d_tor)
        polymer.translation(d_x)
        polymer.euler_rotation(d_rot)
        polymer.torsion_structure(d_tor)
        return polymer

    def chain_to_poly(self, my_chain, coor, res):
        """
        params:
            chain (str): chain identifier
            coor (arr n x 3): coordinates
            res (arr n x 4): residue information of each atom (aname, resnum, chain, resname)
        return:
            polymer
        """

        #print(my_chain)
        #print(coor)
        #print(res)

        polymer = []
        c_resnum = -1;
        c_resname = ''
        n = []
        c = []
        print(len(res))
        
        for i in range(len(res)):
            ii = res[i]
            aname = ii[0]
            resnum = ii[1]
            chain = ii[2]
            resname = ii[3]
            
            

            if c_resnum != resnum:
                if c_resnum != -1 and len(n) > 1:
                    #print(str(c_resnum),c_resname,np.array(c),n)
                    m = mono(str(c_resnum),c_resname,np.array(c),n)
                    polymer.append(m)
                #reset
                c_resnum = resnum;
                c_resname = resname
                n = []
                c = []

            #if chain == my_chain:
            if True:

                n.append(aname)
                c.append(coor[i].cpu().detach().numpy())

        
        if len(n) > 1:
            m = mono(str(c_resnum),c_resname,np.array(c),n)
            print(c_resnum,c_resname,np.array(c),n)
            polymer.append(m)



        return poly(polymer)


    def step(self, batch, batch_idx):
        #rec_x = batch['rec_x'].squeeze(0)
        #lig_x = batch['lig_x'].squeeze(0)
        #rec_pos = batch['rec_pos'].squeeze(0)
        #lig_pos = batch['lig_pos'].squeeze(0)
        #contact_matrix = batch['contact_matrix'].squeeze(0)
        #position_matrix = batch['position_matrix'].squeeze(0)
        #print(batch)

        #print(batch[0], batch[1], batch[2])

        print(np.shape(batch[0]), np.shape(batch[1]), np.shape(batch[2]))
        #print(batch[0][0])
        #print(batch[1][0,...])
        #print(batch[2])
        polymer = self.chain_to_poly(batch[0][0], batch[1][0,...], batch[2])

        # get losses
        losses = self.loss_fn(polymer)
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


#----------------------------------------------------------------------------
# Testing run

@hydra.main(version_base=None, config_path="/scratch4/jgray21/lchu11/graylab_repos/DL_Gen_Docking/configs/model", config_name="score_model.yaml")
def main(conf: DictConfig):
    dataset = DockingDataset(
        dataset='dips_train_hetero',
        use_sasa=True,
        use_interface=True,
        use_contact=True,
    )

    subset_indices = [0]
    subset = data.Subset(dataset, subset_indices)

    #load dataset
    dataloader = DataLoader(subset)
    
    model = Score_Model(
        model=conf.model, 
        diffuser=conf.diffuser,
        experiment=conf.experiment
    )
    trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=10, inference_mode=False)
    trainer.validate(model, dataloader)

if __name__ == '__main__':
    main()

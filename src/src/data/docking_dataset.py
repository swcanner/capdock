import os
import csv
import random
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import warnings
from tqdm import tqdm
from typing import Optional
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import HeteroData
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB import Structure, Model, Chain, Residue, Atom
from scipy.spatial.transform import Rotation 
from src.utils import residue_constants

#----------------------------------------------------------------------------
# Helper functions

def get_interface_residues(coords, asym_id, interface_threshold=10.0):
    coord_diff = coords[..., None, :, :] - coords[..., None, :, :, :]
    pairwise_dists = torch.sqrt(torch.sum(coord_diff ** 2, dim=-1))
    diff_chain_mask = (asym_id[..., None, :] != asym_id[..., :, None]).float()
    mask = diff_chain_mask[..., None].bool()
    min_dist_per_res, _ = torch.where(mask, pairwise_dists, torch.inf).min(dim=-1)
    valid_interfaces = torch.sum((min_dist_per_res < interface_threshold).float(), dim=-1)
    interface_residues_idxs = torch.nonzero(valid_interfaces, as_tuple=True)[0]

    return interface_residues_idxs

def get_spatial_crop_idx(coords, asym_id, crop_size=256, interface_threshold=10.0):
    interface_residues = get_interface_residues(coords, asym_id, interface_threshold=interface_threshold)

    if not torch.any(interface_residues):
        return get_contiguous_crop_idx(asym_id, crop_size)

    target_res_idx = randint(lower=0, upper=interface_residues.shape[-1] - 1)
    target_res = interface_residues[target_res_idx]

    ca_positions = coords[..., 1, :]
    coord_diff = ca_positions[..., None, :] - ca_positions[..., None, :, :]
    ca_pairwise_dists = torch.sqrt(torch.sum(coord_diff ** 2, dim=-1))
    to_target_distances = ca_pairwise_dists[target_res]

    break_tie = (
            torch.arange(
                0, to_target_distances.shape[-1] 
            ).float()
            * 1e-3
    )
    to_target_distances += break_tie
    ret = torch.argsort(to_target_distances)[:crop_size]
    return ret.sort().values

def get_contiguous_crop_idx(asym_id, crop_size):
    unique_asym_ids, chain_idxs, chain_lens = asym_id.unique(dim=-1,
                                                             return_inverse=True,
                                                             return_counts=True)
    
    shuffle_idx = torch.randperm(chain_lens.shape[-1])
    

    _, idx_sorted = torch.sort(chain_idxs, stable=True)
    cum_sum = chain_lens.cumsum(dim=0)
    cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]), dim=0)
    asym_offsets = idx_sorted[cum_sum]

    num_budget = crop_size
    num_remaining = len(chain_idxs)

    crop_idxs = []
    for i, idx in enumerate(shuffle_idx):
        chain_len = int(chain_lens[idx])
        num_remaining -= chain_len

        if i == 0:
            crop_size_max = min(num_budget - 50, chain_len)
            crop_size_min = min(chain_len, 50)
        else:
            crop_size_max = min(num_budget, chain_len)
            crop_size_min = min(chain_len, max(50, num_budget - num_remaining))

        chain_crop_size = randint(lower=crop_size_min,
                                  upper=crop_size_max)

        num_budget -= chain_crop_size

        chain_start = randint(lower=0,
                              upper=chain_len - chain_crop_size)

        asym_offset = asym_offsets[idx]
        crop_idxs.append(
            torch.arange(asym_offset + chain_start, asym_offset + chain_start + chain_crop_size)
        )

    return torch.concat(crop_idxs).sort().values

def randint(lower, upper):
    return int(torch.randint(
        lower,
        upper + 1,
        (1,),
    )[0])

def numpy_array_to_structure(numpy_array):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore warnings from BioPython
        structure = Structure.Structure('example_structure')
        model = Model.Model(0)
        chain = Chain.Chain('A')
        residue_counter = 1
        
        for residue_coords in numpy_array:
            residue_id = (' ', residue_counter, ' ')  # Unique residue ID
            residue_counter += 1
            
            residue = Residue.Residue(residue_id, 'ALA', 1)
            
            for atom_name, coords in zip(['N', 'CA', 'C'], residue_coords):
                atom = Atom.Atom(atom_name, coords, 0, 0, ' ', atom_name, 0)
                residue.add(atom)
            
            chain.add(residue)
        
        model.add(chain)
        structure.add(model)
    
    return structure

def extract_sasa_to_torch(structure, chain_id='A'):
    sasa_values = []

    for residue in structure[0][chain_id]:
        if hasattr(residue, 'sasa'):
            sasa_values.append(residue.sasa)
        else:
            sasa_values.append(0.0)  # or np.nan, if you want to handle missing values differently

    return torch.tensor(sasa_values).float().unsqueeze(-1)

def get_interface_residue_tensors(set1, set2, threshold=8.0):
    n1_len = set1.shape[0]
    n2_len = set2.shape[0]
    
    # Calculate the Euclidean distance between each pair of points from the two sets
    dists = torch.cdist(set1, set2)

    # Find the indices where the distance is less than the threshold
    close_points = dists < threshold

    # Create indicator tensors initialized to 0
    indicator_set1 = torch.zeros((n1_len, 1), dtype=torch.float32)
    indicator_set2 = torch.zeros((n2_len, 1), dtype=torch.float32)

    # Set the corresponding indices to 1 where the points are close
    indicator_set1[torch.any(close_points, dim=1)] = 1.0
    indicator_set2[torch.any(close_points, dim=0)] = 1.0

    return indicator_set1, indicator_set2

def get_sampled_contact_matrix(set1, set2, threshold=8.0, num_samples=None):
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
    contact_matrix = torch.zeros((n1 + n2, n1 + n2))

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

def get_position_matrix(rec_len, lig_len):
    """
    edge positional embedding (one-hot) for different chains of the complex.
    [1, 0] for intra-chain edges;
    [0, 1] for inter-chain edges.
    """
    # chain embedding
    total_len = rec_len + lig_len
    chains = torch.zeros(total_len, total_len)
    chains[:rec_len, rec_len:] = 1
    chains[rec_len:, :rec_len] = 1
    chains = F.one_hot(chains.long(), num_classes=2).float()

    # residue embedding
    rmax = 32
    rec = torch.arange(0, rec_len)
    lig = torch.arange(0, lig_len) 
    total = torch.cat([rec, lig], dim=0)
    pairs = total[None, :] - total[:, None]
    pairs = torch.clamp(pairs, min=-rmax, max=rmax)
    pairs = pairs + rmax 
    pairs[:rec_len, rec_len:] = 2*rmax + 1
    pairs[rec_len:, :rec_len] = 2*rmax + 1 
    relpos = F.one_hot(pairs, num_classes=2*rmax+2).float()

    return torch.cat([relpos, chains], dim=-1)

def one_hot(x, v_bins):
    reshaped_bins = v_bins.view(((1,) * len(x.shape)) + (len(v_bins),))
    diffs = x[..., None] - reshaped_bins
    am = torch.argmin(torch.abs(diffs), dim=-1)
    return F.one_hot(am, num_classes=len(v_bins)).float()

def relpos(res_id, asym_id, use_chain_relative=True):
    max_relative_idx = 32
    pos = res_id
    asym_id_same = (asym_id[..., None] == asym_id[..., None, :])
    offset = pos[..., None] - pos[..., None, :]

    clipped_offset = torch.clamp(
        offset + max_relative_idx, 0, 2 * max_relative_idx
    )

    rel_feats = []
    if use_chain_relative:
        final_offset = torch.where(
            asym_id_same, 
            clipped_offset,
            (2 * max_relative_idx + 1) * 
            torch.ones_like(clipped_offset)
        )

        boundaries = torch.arange(
            start=0, end=2 * max_relative_idx + 2
        )
        rel_pos = one_hot(
            final_offset,
            boundaries,
        )

        rel_feats.append(rel_pos)

        """
        entity_id = batch["entity_id"]
        entity_id_same = (entity_id[..., None] == entity_id[..., None, :])
        rel_feats.append(entity_id_same[..., None].to(dtype=rel_pos.dtype))

        sym_id = batch["sym_id"]
        rel_sym_id = sym_id[..., None] - sym_id[..., None, :]

        max_rel_chain = self.max_relative_chain
        clipped_rel_chain = torch.clamp(
            rel_sym_id + max_rel_chain,
            0,
            2 * max_rel_chain,
        )

        final_rel_chain = torch.where(
            entity_id_same,
            clipped_rel_chain,
            (2 * max_rel_chain + 1) *
            torch.ones_like(clipped_rel_chain)
        )

        boundaries = torch.arange(
            start=0, end=2 * max_rel_chain + 2, device=final_rel_chain.device
        )
        rel_chain = one_hot(
            final_rel_chain,
            boundaries,
        )

        rel_feats.append(rel_chain)
        """
    else:
        boundaries = torch.arange(
            start=0, end=2 * max_relative_idx + 1
        )
        rel_pos = one_hot(
            clipped_offset, boundaries,
        )
        rel_feats.append(rel_pos)

    rel_feat = torch.cat(rel_feats, dim=-1).float()

    return rel_feat

def random_rotation(rec_pos, lig_pos):
    rot = torch.from_numpy(Rotation.random().as_matrix()).float()
    pos = torch.cat([rec_pos, lig_pos], dim=0)
    cen = pos[..., 1, :].mean(dim=0)
    pos = (pos - cen) @ rot.T
    rec_pos_out = pos[:rec_pos.size(0)]
    lig_pos_out = pos[rec_pos.size(0):]
    return rec_pos_out, lig_pos_out

#----------------------------------------------------------------------------
# Dataset class

class DockingDataset(Dataset):
    def __init__(
        self, 
        dataset: str,
        training: bool = True,
        use_sasa: bool = False,
        use_interface: bool = False,
        use_contact: bool = False,
        crop_size: int = 1000,
    ):
        self.dataset = dataset 
        self.training = training
        self.use_sasa = use_sasa
        self.use_interface = use_interface
        self.use_contact = use_contact
        self.crop_size = crop_size

        if self.use_sasa:
            self.sr = ShrakeRupley()

        if dataset == 'dips_train_0.3_rep':
            self.data_dir = "/scratch4/jgray21/lchu11/data/pt/dips_bb"
            self.data_list = "/scratch4/jgray21/lchu11/data/dips/data_list/geodock/train_0.3_rep.txt" 
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 

        elif dataset == 'dips_val_0.3_rep':
            self.data_dir = "/scratch4/jgray21/lchu11/data/pt/dips_bb"
            self.data_list = "/scratch4/jgray21/lchu11/data/dips/data_list/geodock/val_0.3_rep.txt" 
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 

        elif dataset == 'dips_train':
            self.data_dir = "/scratch4/jgray21/lchu11/data/pt/dips_bb"
            self.data_list = "/scratch4/jgray21/lchu11/data/dips/data_list/geodock/train.txt" 
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 

        elif dataset == 'dips_val':
            self.data_dir = "/scratch4/jgray21/lchu11/data/pt/dips_bb"
            self.data_list = "/scratch4/jgray21/lchu11/data/dips/data_list/geodock/val.txt" 
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 

        elif dataset == 'dips_train_hetero':
            self.data_dir = "/scratch4/jgray21/lchu11/data/pt/dips_bb"
            self.data_list = "/scratch4/jgray21/lchu11/graylab_repos/Dock_Diffusion/src/data/txt_files/dips_train_hetero.txt" 
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 

        elif dataset == 'dips_val_hetero':
            self.data_dir = "/scratch4/jgray21/lchu11/data/pt/dips_bb"
            self.data_list = "/scratch4/jgray21/lchu11/graylab_repos/Dock_Diffusion/src/data/txt_files/dips_val_hetero.txt" 
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 

        elif dataset == 'pinder_train':
            self.data_dir = "/scratch4/jgray21/lchu11/data/pt/pinder_train"
            self.data_list = "/scratch4/jgray21/lchu11/graylab_repos/Dock_Diffusion/src/data/txt_files/pinder_train.txt" 
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 

        elif dataset == 'pinder_val':
            self.data_dir = "/scratch4/jgray21/lchu11/data/pt/pinder_val"
            self.data_list = "/scratch4/jgray21/lchu11/graylab_repos/Dock_Diffusion/src/data/txt_files/pinder_val.txt" 
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 
            
        elif dataset == 'db5_test':
            self.data_dir = "/scratch4/jgray21/lchu11/data/pt/db5_bound"
            self.data_list = "/scratch4/jgray21/lchu11/data/db5/test_bound.txt"
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 
            
        elif dataset == 'dips_test':
            self.data_dir = "/scratch4/jgray21/lchu11/data/pt/dips_test"
            self.data_list = "/scratch4/jgray21/lchu11/data/dips/data_list/geodock/test.txt" 
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 

        elif dataset == 'db5_bound':
            self.data_dir = "/scratch4/jgray21/lchu11/data/pt/db5_bound"
            self.data_list = "/scratch4/jgray21/lchu11/data/db5/bound.txt"
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 

        elif dataset == 'ppi3d_train':
            self.data_dir = "/scratch4/jgray21/lchu11/data/pt/ppi3d"
            self.data_list = "/scratch4/jgray21/lchu11/data/ppi3d/train.txt"
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 

        elif dataset == 'ppi3d_val':
            self.data_dir = "/scratch4/jgray21/lchu11/data/pt/ppi3d"
            self.data_list = "/scratch4/jgray21/lchu11/data/ppi3d/val.txt"
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 

        elif dataset == 'ppi3d_test':
            self.data_dir = "/scratch4/jgray21/lchu11/data/pt/ppi3d"
            self.data_list = "/scratch4/jgray21/lchu11/data/ppi3d/test_300.txt"
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 


    def __getitem__(self, idx: int):
        if self.dataset[:4] == 'dips':
            # Get info from file_list 
            _id = self.file_list[idx]
            split_string = _id.split('/')
            _id = split_string[0] + '_' + split_string[1].rsplit('.', 1)[0]
            data = torch.load(os.path.join(self.data_dir, _id+'.pt'))
        else:
            _id = self.file_list[idx]
            data = torch.load(os.path.join(self.data_dir, _id+'.pt'))

        rec_x = data['receptor'].x
        rec_seq = data['receptor'].seq
        rec_pos = data['receptor'].pos
        lig_x = data['ligand'].x
        lig_seq = data['ligand'].seq
        lig_pos = data['ligand'].pos

        # One-Hot embeddings
        rec_onehot = torch.from_numpy(residue_constants.sequence_to_onehot(
            sequence=rec_seq,
            mapping=residue_constants.restype_order_with_x,
            map_unknown_to_x=True,
        )).float()

        lig_onehot = torch.from_numpy(residue_constants.sequence_to_onehot(
            sequence=lig_seq,
            mapping=residue_constants.restype_order_with_x,
            map_unknown_to_x=True,
        )).float()

        rec_x = torch.cat([rec_x, rec_onehot], dim=-1)
        lig_x = torch.cat([lig_x, lig_onehot], dim=-1)

        # SASA embeddings
        if self.use_sasa:
            rec_struct = numpy_array_to_structure(rec_pos)
            lig_struct = numpy_array_to_structure(lig_pos)
            self.sr.compute(rec_struct, level="R")
            self.sr.compute(lig_struct, level="R")

            rec_sasa = extract_sasa_to_torch(rec_struct)
            rec_rsasa = rec_sasa / rec_sasa.max()
            lig_sasa = extract_sasa_to_torch(lig_struct)
            lig_rsasa = lig_sasa / lig_sasa.max()
        else:
            rec_rsasa = torch.zeros(rec_pos.size(0), 1).float()
            lig_rsasa = torch.zeros(lig_pos.size(0), 1).float()

        rec_x = torch.cat([rec_x, rec_rsasa], dim=-1)
        lig_x = torch.cat([lig_x, lig_rsasa], dim=-1)
        
        # Interface embeddings
        if self.use_interface:
            rec_ires, lig_ires = get_interface_residue_tensors(rec_pos[..., 1, :], lig_pos[..., 1, :])

            if self.training:
                # Generate a random number between 0 and 1
                rand_num = random.random()

                # Apply masks based on the random number
                if rand_num < 0.25:
                    # Both tensors are masked with zeros (25% probability)
                    rec_ires = torch.zeros_like(rec_ires)
                    lig_ires = torch.zeros_like(lig_ires)
                elif rand_num < 0.75:
                    # One of the tensors is masked with zeros (50% probability)
                    if random.random() < 0.5:
                        rec_ires = torch.zeros_like(rec_ires)
                    else:
                        lig_ires = torch.zeros_like(lig_ires)
        else:
            rec_ires = torch.zeros(rec_pos.size(0), 1).float()
            lig_ires = torch.zeros(lig_pos.size(0), 1).float()

        rec_x = torch.cat([rec_x, rec_ires], dim=-1)
        lig_x = torch.cat([lig_x, lig_ires], dim=-1)

        if self.training:
            # shuffle the order of rec and lig
            vars_list = [(rec_x, rec_seq, rec_pos), (lig_x, lig_seq, lig_pos)]
            random.shuffle(vars_list)
            rec_x, rec_seq, rec_pos = vars_list[0]
            lig_x, lig_seq, lig_pos = vars_list[1]

            # crop to size
            rec_x, lig_x, rec_pos, lig_pos, res_id, asym_id = self.crop_to_size(rec_x, lig_x, rec_pos, lig_pos)
            
        else:
            # make the smaller subunit to be lig
            vars_list = [(rec_x, rec_seq, rec_pos), (lig_x, lig_seq, lig_pos)]
            if rec_x.size(0) < lig_x.size(0):
                rec_x, rec_seq, rec_pos = vars_list[1]
                lig_x, lig_seq, lig_pos = vars_list[0]

        # Contact embeddings
        if self.use_contact:
            if self.training:
                contact_matrix = get_sampled_contact_matrix(
                    rec_pos[..., 1, :], lig_pos[..., 1, :], num_samples=random.randint(0, 3))
            else:
                contact_matrix = get_sampled_contact_matrix(
                    rec_pos[..., 1, :], lig_pos[..., 1, :], num_samples=1)
        else:
            n = rec_pos.size(0) + lig_pos.size(0)
            contact_matrix = torch.zeros(n, n)

        # Positional embeddings
        position_matrix = relpos(res_id, asym_id)

        # random rotation augmentation
        rec_pos, lig_pos = random_rotation(rec_pos, lig_pos)

        # Output
        output = {
            'id': _id,
            'rec_seq': rec_seq,
            'lig_seq': lig_seq,
            'rec_x': rec_x,
            'lig_x': lig_x,
            'rec_pos': rec_pos,
            'lig_pos': lig_pos,
            'contact_matrix': contact_matrix,
            'position_matrix': position_matrix,
        }
        
        return {key: value for key, value in output.items()}

    def __len__(self):
        return len(self.file_list)

    def crop_to_size(self, rec_x, lig_x, rec_pos, lig_pos):
        n = rec_x.size(0) + lig_x.size(0)
        res_id = torch.arange(n).long()
        asym_id = torch.zeros(n).long()
        asym_id[rec_x.size(0):] = 1
        x = torch.cat([rec_x, lig_x], dim=0)
        pos = torch.cat([rec_pos, lig_pos], dim=0)

        use_spatial_crop = random.random() < 0.5
        num_res = asym_id.size(0)

        if num_res <= self.crop_size:
            crop_idxs = torch.arange(num_res)
        elif use_spatial_crop:
            crop_idxs = get_spatial_crop_idx(pos, asym_id, crop_size=self.crop_size)
        else:
            crop_idxs = get_contiguous_crop_idx(asym_id, crop_size=self.crop_size)

        asym_id = torch.index_select(asym_id, 0, crop_idxs)
        res_id = torch.index_select(res_id, 0, crop_idxs)
        x = torch.index_select(x, 0, crop_idxs)
        pos = torch.index_select(pos, 0, crop_idxs)

        sep = asym_id.tolist().index(1)
        rec_x = x[:sep]
        lig_x = x[sep:]
        rec_pos = pos[:sep]
        lig_pos = pos[sep:]

        return rec_x, lig_x, rec_pos, lig_pos, res_id, asym_id

#----------------------------------------------------------------------------
# DataModule class

class DockingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: str = "data/",
        val_dataset: str = "data/",
        batch_size: int = 1,
        use_sasa: bool = False,
        use_interface: bool = False,
        use_contact: bool = False,
        **kwargs
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.use_sasa = use_sasa
        self.use_interface = use_interface
        self.use_contact = use_contact
        self.num_workers = kwargs['num_workers']
        self.pin_memory = kwargs['pin_memory']

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
    
    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.data_train = DockingDataset(
            dataset=self.train_dataset, 
            use_sasa=self.use_sasa,
            use_interface=self.use_interface,
            use_contact=self.use_contact,
        )
        self.data_val = DockingDataset(
            dataset=self.val_dataset, 
            use_sasa=self.use_sasa,
            use_interface=self.use_interface,
            use_contact=self.use_contact,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

#----------------------------------------------------------------------------
# Testing

if __name__ == '__main__':
    dataset = DockingDataset(
        dataset="ppi3d_train",
        training=True,
        use_sasa=True,
        use_interface=True,
        use_contact=True,
    )
    dataset[0]

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from einops import repeat
from src.models.egnn import E_GCL
from src.utils.coords6d import get_coords6d

#----------------------------------------------------------------------------
# Data class for model config

@dataclass
class ModelConfig:
    lm_embed_dim: int
    positional_embed_dim: int
    spatial_embed_dim: int
    contact_embed_dim: int
    node_dim: int
    edge_dim: int
    inner_dim: int
    depth: int
    dropout: float = 0.0
    cut_off: float = 30.0
    normalize: bool = False
    n_tor_bins: int

#----------------------------------------------------------------------------
# Helper functions

def get_knn_and_sample(points, knn=20, sample_size=40, epsilon=1e-10):
    device = points.device
    n_points = points.size(0)

    if n_points < knn + sample_size:
        sample_size = n_points - knn
    
    # Step 1: Compute pairwise distances
    dist_matrix = torch.cdist(points, points)
    
    # Step 2: Find the 20 nearest neighbors (including the point itself)
    _, knn_indices = torch.topk(dist_matrix, k=knn, largest=False)
    
    # Step 3: Create a mask for the non-knn points
    mask = torch.ones(n_points, n_points, dtype=torch.bool, device=device)
    mask.scatter_(1, knn_indices, False)
    
    # Select the non-knn distances and compute inverse cubic distances
    non_knn_distances = dist_matrix[mask].view(n_points, -1)
    
    # Replace zero distances with a small value to avoid division by zero
    non_knn_distances = torch.where(non_knn_distances < epsilon, torch.tensor(epsilon, device=device), non_knn_distances)
    
    inv_cubic_distances = 1 / torch.pow(non_knn_distances, 3)
    
    # Normalize the inverse cubic distances to get probabilities
    probabilities = inv_cubic_distances / inv_cubic_distances.sum(dim=1, keepdim=True)
    
    # Ensure there are no NaNs or negative values
    probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
    probabilities = torch.clamp(probabilities, min=0)
    
    # Normalize again to ensure it's a proper probability distribution
    probabilities /= probabilities.sum(dim=1, keepdim=True)
    
    # Generate a tensor of indices excluding knn_indices
    all_indices = torch.arange(n_points, device=device).expand(n_points, n_points)
    non_knn_indices = all_indices[mask].view(n_points, -1)
    
    # Step 4: Sample 40 indices based on the probability distribution
    sample_indices = torch.multinomial(probabilities, sample_size, replacement=False)
    sampled_points_indices = non_knn_indices.gather(1, sample_indices)
    
    return knn_indices, sampled_points_indices

#def ang_to_rbf(p,n_bins):
#    #encodings RBF = gaus(phi) = exp( (eps * (x - u) )^2 )
#    self.rbf_dist_means = np.linspace(0,20,16)
#    self.rbf_eps = (self.rbf_dist_means[-1] - self.rbf_dist_means[0]) / len(self.rbf_dist_means);

#----------------------------------------------------------------------------
# Modules

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""  
    def __init__(self, embed_dim, scale=1.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class EGNNLayer(nn.Module):
    def __init__(
        self, 
        node_dim, 
        edge_dim=0, 
        act_fn=nn.SiLU(), 
        residual=True, 
        attention=False, 
        normalize=False, 
        tanh=False, 
        update_coords=False,
    ):
        super(EGNNLayer, self).__init__()
        self.egcl = E_GCL(
            input_nf=node_dim, 
            output_nf=node_dim, 
            hidden_nf=node_dim, 
            edges_in_d=edge_dim,
            act_fn=act_fn, 
            residual=residual, 
            attention=attention,
            normalize=normalize, 
            tanh=tanh, 
            update_coords=update_coords,
        )

    def forward(self, h, x, edges, edge_attr=None):
        h, x, _ = self.egcl(h, edges, x, edge_attr=edge_attr)
        return h, x


class EGNN(nn.Module):
    def __init__(
        self, 
        node_dim, 
        edge_dim=0, 
        act_fn=nn.SiLU(), 
        depth=4, 
        residual=True, 
        attention=False, 
        normalize=False, 
        tanh=False, 
    ):
        super(EGNN, self).__init__()
        self.depth = depth
        for i in range(depth):
            is_last = i == depth - 1
            self.add_module("EGNN_%d" % i, EGNNLayer(
                node_dim=node_dim, 
                edge_dim=edge_dim,
                act_fn=act_fn, 
                residual=residual, 
                attention=attention,
                normalize=normalize, 
                tanh=tanh, 
                update_coords=is_last
            )
        )

    def forward(self, h, x, edges, edge_attr=None):
        for i in range(self.depth):
            h, x = self._modules["EGNN_%d" % i](h, x, edges, edge_attr=edge_attr)
        return h, x



#----------------------------------------------------------------------------
# Main score network

class Score_Net(nn.Module):
    """EGNN backbone for translation and rotation scores"""
    def __init__(
        self, 
        conf,
    ):
        super().__init__()
        lm_embed_dim = conf.lm_embed_dim
        positional_embed_dim = conf.positional_embed_dim
        contact_embed_dim = conf.contact_embed_dim
        node_dim = conf.node_dim
        edge_dim = conf.edge_dim
        inner_dim = conf.inner_dim
        depth = conf.depth
        dropout = conf.dropout
        normalize = conf.normalize
        n_tor_bins = conf.n_tor_bins
        
        self.cut_off = conf.cut_off
        
        # single init embedding
        node_in_dim = lm_embed_dim
        self.single_embed = nn.Linear(node_in_dim, node_dim, bias=False)

        # pair init embedding
        self.pair_i_embed = nn.Linear(node_in_dim, edge_dim, bias=False)
        self.pair_j_embed = nn.Linear(node_in_dim, edge_dim, bias=False)
        self.positional_embed = nn.Linear(positional_embed_dim, edge_dim, bias=False)
        self.contact_embed = nn.Linear(contact_embed_dim, edge_dim, bias=False)

        #Atom-> atom network
        self.atom_atom_network = EGNN(
            node_dim=node_dim, 
            edge_dim=edge_dim, 
            act_fn=nn.SiLU(), 
            depth=depth, 
            residual=True, 
            attention=False, 
            normalize=normalize, 
            tanh=False,
        )

        #Atom -> ring network
        self.atom_ring_network = EGNN(
            node_dim=node_dim, 
            edge_dim=edge_dim, 
            act_fn=nn.SiLU(), 
            depth=depth, 
            residual=True, 
            attention=False, 
            normalize=False, 
            tanh=False,
        )

        #ring -> ring network
        self.ring_ring_network = EGNN(
            node_dim=node_dim, 
            edge_dim=edge_dim, 
            act_fn=nn.SiLU(), 
            depth=depth, 
            residual=True, 
            attention=False, 
            normalize=normalize, 
            tanh=False,
        )


        # denoising score network
        self.network = EGNN(
            node_dim=node_dim, 
            edge_dim=edge_dim, 
            act_fn=nn.SiLU(), 
            depth=depth, 
            residual=True, 
            attention=False, 
            normalize=normalize, 
            tanh=False,
        )

        # energy head
        self.to_energy = nn.Sequential(
            nn.Linear(2*node_dim, node_dim, bias=False),
            nn.LayerNorm(node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 1, bias=False),
        )

        # timestep embedding
        self.t_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=inner_dim),
            nn.Linear(inner_dim, inner_dim, bias=False),
            nn.Sigmoid(),
        )

        # tr_scale mlp
        self.tr_scale = nn.Sequential(
            nn.Linear(inner_dim + 1, inner_dim, bias=False),
            nn.LayerNorm(inner_dim),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(inner_dim, 1, bias=False),
            nn.Softplus()
        )

        # rot_scale mlp
        self.rot_scale = nn.Sequential(
            nn.Linear(inner_dim + 1, inner_dim, bias=False),
            nn.LayerNorm(inner_dim),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(inner_dim, 1, bias=False),
            nn.Softplus()
        )

        # rot_scale mlp
        self.tor_scale = nn.Sequential(
            nn.Linear(node_dim * 2 + 2, node_dim, bias=False),
            nn.LayerNorm(node_dim),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(node_dim, 1, bias=False),
            nn.Softplus()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def forward(self, polymer, t, contact_matrix, position_matrix, predict=False):
        # set device
        self.device = t.device

        # get the current complex pose
        atom_pos = polymer.get_atom_coor()
        atom_x = polymer.atom_onehot()

        ring_pos = polymer.get_ring_coms()
        ring_x = torch.zeros((len(ring_pos),self.node_dim))

        ring_atom_edge = polymer.get_ring_atoms_edge();
        atom_ring_edge = polymer.get_ring_atoms_edge(reverse=True)

        pos = torch.cat([atom_pos,ring_pos],0)
        x = torch.cat([atom_x,ring_x],0)
        is_ring = torch.cat([torch.zeros(len(atom_pos)), torch.ones(len(ring_pos)) ],0)

        #very simple dumb encoding. Should be changed for my boy but doesn't matter
        edge_all = self.pair_i_embed(x)[None, :, :] + self.pair_j_embed(x)[:, None, :]
        edge_atom = self.pair_i_embed(atom_x)[None, :, :] + self.pair_j_embed(atom_x)[:, None, :]
        edge_ring = self.pair_i_embed(ring_x)[None, :, :] + self.pair_j_embed(ring_x)[:, None, :]

        edge_atom_index, edge_atom_attr = self.get_knn_and_sample_graph(atom_pos[...,1,:], edge_atom)
        edge_ring_index, edge_ring_attr = self.get_knn_and_sample_graph(ring_pos[...,1,:], edge_ring)
        edge_ring_atom_index, edge_ring_atom_attr = self.get_ring_graph(atom_pos[...,1,:], edge_all)

        bonded_rings = polymer.edges
        _, ang_diff = polymer.calc_normal_angles()
        



        #lig_pos.requires_grad_()
        #pos = lig_pos

        # get ca distance matrix 
        #D = torch.norm((rec_pos[:, None, 1, :] - lig_pos[None, :, 1, :]), dim=-1)

        # node feature embedding
        node = self.single_embed(x) # [n, c]

        # edge feature embedding
        
        #edge += self.positional_embed(position_matrix)
        #edge += self.contact_embed(contact_matrix.unsqueeze(-1)) # [n, n, c]

        # TODO pair former

        # sample edge_index and get edge_attr
        #edge_index, edge_attr = self.get_knn_and_sample_graph(pos[..., 1, :], edge)

        # atom -> atomRing -> Ring
        node_atom, pos = self.atom_atom_network(node, pos[..., 1, :], edge_atom_index, edge_atom_attr) # [R+L, H]
        node_ring, pos = self.atom_ring_network(
                node_atom, pos[..., 1, :], edge_ring_atom_index, edge_ring_atom_attr)
        node_ring, pos = self.ring_ring_network(
                node_ring, pos[..., 1, :], edge_ring_index, edge_ring_attr)

        #get outputs
        # torsion
        #tor = torch.cross(r, f, dim=-1).mean(dim=0, keepdim=True)
        tor_pred = torch.zeros(len(bonded_rings))
        for ii in range(len(bonded_rings)):
            tor_pred[ii] = self.torsion_network(torch.cat([
                node_ring[ bonded_rings[ii,0] ],node_ring[ bonded_rings[ii,1]], [torch.sin(ang_diff[ii]), torch.cos(ang_diff[ii])] ] ) )


        # energy
        #h_rec = repeat(node_out[:rec_pos.size(0)], 'n h -> n m h', m=lig_pos.size(0))
        #h_lig = repeat(node_out[rec_pos.size(0):], 'm h -> n m h', n=rec_pos.size(0))
        #energy = self.to_energy(torch.cat([h_rec, h_lig], dim=-1)).squeeze(-1) # [R, L]
        #mask_2D = (D < self.cut_off).float() # [R, L]
        #energy = (energy * mask_2D).sum() / (mask_2D.sum() + 1e-6) # [] E / kT

        # get translation and rotation vectors
        #lig_pos_curr = pos_out[rec_pos.size(0):] 
        #r = lig_pos[..., 1, :].detach()
        #f = lig_pos_curr - r # f / kT

        # translation
        #tr_pred = f.mean(dim=0, keepdim=True)

        # rotation
        #rot_pred = torch.cross(r, f, dim=-1).mean(dim=0, keepdim=True)

        # scale
        t = self.t_embed(t)
        tor_pred *= self.tor_scale(torch.cat([tor_pred,t], dim=-1))

        #tr_norm = torch.linalg.vector_norm(tr_pred, keepdim=True)
        #tr_pred = tr_pred / (tr_norm + 1e-6) * self.tr_scale(torch.cat([tr_norm, t], dim=-1))
        #rot_norm = torch.linalg.vector_norm(rot_pred, keepdim=True)
        #rot_pred = rot_pred / (rot_norm + 1e-6) * self.rot_scale(torch.cat([rot_norm, t], dim=-1))

        if predict:
            fa_rep = self.fa_rep(D).sum()
            return energy, fa_rep, f, tr_pred, rot_pred

        # dedx
        """
        dedx = torch.autograd.grad(
            outputs=energy, 
            inputs=lig_pos, 
            grad_outputs=torch.ones_like(energy),
            create_graph=self.training, 
            retain_graph=self.training,
            only_inputs=True, 
            allow_unused=True,
        )[0]
        
        dedx = -dedx[..., 1, :] # F / kT
        #"""
        
        #return energy, dedx, f, tr_pred, rot_pred
        return tor_pred

    def get_cross_graph(self, x, e, sep, num_self, num_cross):
        """cross graph from the complex pose"""

        # distance matrix
        d = torch.norm((x[:, None, :] - x[None, :, :]), dim=-1)

        # make sure the knn not exceed the size
        rec_len = sep
        lig_len = x.size(0) - sep

        # self and cross
        num_self_lig = num_self
        num_cross_lig = num_cross
        num_self_rec = num_self
        num_cross_rec = num_cross

        if num_self_lig > lig_len:
            num_self_lig = lig_len
        if num_cross_lig > rec_len:
            num_cross_lig = rec_len
        if num_self_rec > rec_len:
            num_self_rec = rec_len
        if num_cross_rec > lig_len:
            num_cross_rec = lig_len

        # intra and inter topk
        nbhd_ranking_ii, nbhd_indices_ii = d[..., :sep, :sep].topk(num_self_rec, dim=-1, largest=False)
        nbhd_ranking_jj, nbhd_indices_jj = d[..., sep:, sep:].topk(num_self_lig, dim=-1, largest=False)
        nbhd_ranking_ij, nbhd_indices_ij = d[..., :sep, sep:].topk(num_cross_rec, dim=-1, largest=False)
        nbhd_ranking_ji, nbhd_indices_ji = d[..., sep:, :sep].topk(num_cross_lig, dim=-1, largest=False)

        # edge src and dst
        edge_src_rec = torch.arange(start=0, end=rec_len, device=self.device)[..., None].repeat(1, num_self_rec+num_cross_rec)
        edge_src_lig = torch.arange(start=rec_len, end=rec_len+lig_len, device=self.device)[..., None].repeat(1, num_self_lig+num_cross_lig)
        edge_dst_rec = torch.cat([nbhd_indices_ii, nbhd_indices_ij + rec_len], dim=1)
        edge_dst_lig = torch.cat([nbhd_indices_ji, nbhd_indices_jj + rec_len], dim=1)
        edge_src = torch.cat([edge_src_rec.reshape(-1), edge_src_lig.reshape(-1)])
        edge_dst = torch.cat([edge_dst_rec.reshape(-1), edge_dst_lig.reshape(-1)])

        # combine graphs
        edge_index = [edge_src, edge_dst]
        edge_indices = torch.stack(edge_index, dim=1)
        edge_attr = e[edge_indices[:, 0], edge_indices[:, 1]]

        return edge_index, edge_attr

    def get_random_graph(self, x, e, sep, num_self, num_cross):
        """cross graph from the complex pose"""

        # distance matrix
        d = torch.norm((x[:, None, :] - x[None, :, :]), dim=-1)

        # make sure the knn not exceed the size
        rec_len = sep
        lig_len = x.size(0) - sep

        # self and cross
        num_self_lig = num_self
        num_cross_lig = num_cross
        num_self_rec = num_self
        num_cross_rec = num_cross

        if num_self_lig > lig_len:
            num_self_lig = lig_len
        if num_cross_lig > rec_len:
            num_cross_lig = rec_len
        if num_self_rec > rec_len:
            num_self_rec = rec_len
        if num_cross_rec > lig_len:
            num_cross_rec = lig_len

        # intra and inter topk
        nbhd_ranking_ii, nbhd_indices_ii = d[..., :sep, :sep].topk(num_self_rec, dim=-1, largest=False)
        nbhd_ranking_jj, nbhd_indices_jj = d[..., sep:, sep:].topk(num_self_lig, dim=-1, largest=False)
        nbhd_indices_ij = self.sample_indices(d[..., :sep, sep:], num_cross_rec)
        nbhd_indices_ji = self.sample_indices(d[..., sep:, :sep], num_cross_lig)

        # edge src and dst
        edge_src_rec = torch.arange(start=0, end=rec_len, device=self.device)[..., None].repeat(1, num_self_rec+num_cross_rec)
        edge_src_lig = torch.arange(start=rec_len, end=rec_len+lig_len, device=self.device)[..., None].repeat(1, num_self_lig+num_cross_lig)
        edge_dst_rec = torch.cat([nbhd_indices_ii, nbhd_indices_ij + rec_len], dim=1)
        edge_dst_lig = torch.cat([nbhd_indices_ji, nbhd_indices_jj + rec_len], dim=1)
        edge_src = torch.cat([edge_src_rec.reshape(-1), edge_src_lig.reshape(-1)])
        edge_dst = torch.cat([edge_dst_rec.reshape(-1), edge_dst_lig.reshape(-1)])

        # combine graphs
        edge_index = [edge_src, edge_dst]
        edge_indices = torch.stack(edge_index, dim=1)
        edge_attr = e[edge_indices[:, 0], edge_indices[:, 1]]

        return edge_index, edge_attr

    def get_knn_and_sample_graph(self, x, e):
        knn_indices, sampled_points_indices = get_knn_and_sample(x)
        indices = torch.cat([knn_indices, sampled_points_indices], dim=-1)
        n_points, n_samples = indices.shape

        # edge src and dst
        edge_src = torch.arange(start=0, end=n_points, device=self.device)[..., None].repeat(1, n_samples).reshape(-1)
        edge_dst = indices.reshape(-1)

        # combine graphs
        edge_index = [edge_src, edge_dst]
        edge_indices = torch.stack(edge_index, dim=1)
        edge_attr = e[edge_indices[:, 0], edge_indices[:, 1]]

        return edge_index, edge_attr

    def get_positional_embed(self, rec_len, lig_len):
        """
        edge positional embedding (one-hot) for different chains of the complex.
        [1, 0] for intra-chain edges;
        [0, 1] for inter-chain edges.
        """
        # chain embedding
        total_len = rec_len + lig_len
        chains = torch.zeros(total_len, total_len, device=self.device)
        chains[:rec_len, rec_len:] = 1
        chains[rec_len:, :rec_len] = 1
        chains = F.one_hot(chains.long(), num_classes=2).float()

        # residue embedding
        rmax = 32
        rec = torch.arange(0, rec_len, device=self.device)
        lig = torch.arange(0, lig_len, device=self.device) 
        total = torch.cat([rec, lig], dim=0)
        pairs = total[None, :] - total[:, None]
        pairs = torch.clamp(pairs, min=-rmax, max=rmax)
        pairs = pairs + rmax 
        pairs[:rec_len, rec_len:] = 2*rmax + 1
        pairs[rec_len:, :rec_len] = 2*rmax + 1 
        relpos = F.one_hot(pairs, num_classes=2*rmax+2).float()

        return torch.cat([relpos, chains], dim=-1)
        
    def inertia(self, x):
        inner = (x ** 2).sum(dim=-1)
        inner = inner[..., None, None] * torch.eye(3, device=self.device)[None, ...]
        outer = x[..., None, :] * x[..., None]
        inertia = (inner - outer)
        return inertia.sum(dim=0)

    def fa_rep(self, distance):
        return torch.where(distance <= 3.0, 100.0 * torch.exp(-distance**2), torch.tensor(0., device=distance.device))

    def get_spatial_embed(self, coord):
        dist, omega, theta, phi = get_coords6d(coord)

        mask = dist < 22.0
        
        num_omega_bins = 24
        num_theta_bins = 24
        num_phi_bins = 12
        omega_bin = self.get_bins(omega, -180.0, 180.0, num_omega_bins)
        theta_bin = self.get_bins(theta, -180.0, 180.0, num_theta_bins)
        phi_bin = self.get_bins(phi, 0.0, 180.0, num_phi_bins)

        def mask_mat(mat, num_bins):
            mat[~mask] = 0
            mat.fill_diagonal_(0)
            return mat

        omega_bin = mask_mat(omega_bin, num_omega_bins)
        theta_bin = mask_mat(theta_bin, num_theta_bins)
        phi_bin = mask_mat(phi_bin, num_phi_bins)

        # to onehot
        omega = F.one_hot(omega_bin, num_classes=num_omega_bins).float() 
        theta = F.one_hot(theta_bin, num_classes=num_theta_bins).float() 
        phi = F.one_hot(phi_bin, num_classes=num_phi_bins).float() 
        
        return torch.cat([omega, theta, phi], dim=-1)

    def get_bins(self, x, min_bin, max_bin, num_bins):
        # Coords are [... L x 3 x 3], where it's [N, CA, C] x 3 coordinates.
        boundaries = torch.linspace(
            min_bin,
            max_bin,
            num_bins - 1,
            device=x.device,
        )
        bins = torch.sum(x.unsqueeze(-1) > boundaries, dim=-1)  # [..., L, L]
        return bins

    def sample_indices(self, matrix, num_samples):
        n, m = matrix.shape
        # Generate random permutations of indices for each row
        permuted_indices = torch.argsort(torch.rand(n, m, device=self.device), dim=1)

        # Select the first num_samples indices from each permutation
        sampled_indices = permuted_indices[:, :num_samples]

        return sampled_indices
            
#----------------------------------------------------------------------------
# Testing

if __name__ == '__main__':
    conf = ModelConfig(
        lm_embed_dim=1280,
        positional_embed_dim=68,
        spatial_embed_dim=60,
        contact_embed_dim=1,
        node_dim=24,
        edge_dim=12,
        inner_dim=24,
        depth=2,
    )
    model = Score_Net(conf)
    rec_x = torch.randn(40, 1280)
    lig_x = torch.randn(5, 1280)
    rec_pos = torch.randn(40, 3, 3)
    lig_pos = torch.randn(5, 3, 3)
    t = torch.tensor([0.5])
    contact_matrix = torch.zeros(45, 45)
    position_matrix = torch.zeros(45, 45, 68)
    out = model(rec_x, lig_x, rec_pos, lig_pos, t, contact_matrix, position_matrix)
    print(out)

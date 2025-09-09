import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch import Tensor
import copy
from pathlib import Path
import os
from torch_geometric.utils import subgraph
from model.FiLM import DomainFiLM
import numpy as np
import ast
try:
    from sklearn.decomposition import PCA
except ImportError: 
    PCA = None

def _iter_params(model, require_grad_only=True): 
    for p in model.parameters():  
        if require_grad_only and not p.requires_grad:
            continue
        yield p

def flatten_grads(model: nn.Module, require_grad_only = True):
    grads = []
    for p in _iter_params(model, require_grad_only):  
        if p.grad is None:
            grads.append(torch.zeros_like(p).reshape(-1))
        else:
            grads.append(p.grad.detach().reshape(-1))
    if not grads:
        return torch.empty(0, device=next(model.parameters()).device)
    return torch.cat(grads, dim=0)


def unflatten_grads(flattened, model, require_grad_only=True):
    unflattened = []
    offset = 0
    for p in _iter_params(model, require_grad_only):
        numel = p.numel()
        grad_shape = p.shape
        unflattened.append(flattened[offset:offset+numel].view(grad_shape))
        offset += numel
    return unflattened


class GraphProbContrastLoss(nn.Module):
    def __init__(self, mask_ratio,
                       recon_weight,
                       neigh_weight,
                       detach_embed):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.recon_weight = recon_weight
        self.neigh_weight = neigh_weight
        self.detach_embed = detach_embed
    
    def forward(self, data, embed):
        x, edge_index = data.x, data.edge_index
        device = x.device
        num_nodes, feat_dim = x.size()
        num_mask = max(1, int(self.mask_ratio * num_nodes))
        mask_idx = torch.randperm(num_nodes, device=device)[:num_mask]
        x_masked = x[mask_idx]
        h_masked = embed[mask_idx]
        if self.detach_embed:
            h_masked = h_masked.detach()
        recon = F.linear(h_masked, weight = torch.randn(feat_dim, h_masked.size(1), device=device) * 0.01)
        if edge_index.numel() > 0:
            row, col = edge_index
            deg = torch.bincount(row, minlength=num_nodes).float().clamp(min=1.0)
            neigh_mean = torch.zeros_like(embed)
            neigh_mean.index_add_(0, row, embed[col])
            neigh_mean = neigh_mean / deg.unsqueeze(-1)
        else:
            neigh_mean = embed
        recon_pred = neigh_mean[mask_idx][:,:feat_dim] if neigh_mean.size(1) >= feat_dim else F.pad(neigh_mean[mask_idx], (0, feat_dim - neigh_mean.size(1)))
        recon_loss = F.mse_loss(recon, recon_pred, reduction='mean')

        if edge_index.numel() == 0:
            neigh_loss = torch.tensor(0.0, device=device)
        else:
            row, col = edge_index
            diff = embed[row] - embed[col]
            neigh_loss = diff.pow(2).sum(dim=-1).mean()
        return self.recon_weight * recon_loss + self.neigh_weight * neigh_loss

class ConvProjection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        de_cfg = cfg.Fingerprint.DE
        self.d_e = cfg.Fingerprint.compressed_dim
        hidden_channels = de_cfg.hidden_channels
        num_conv_layers = de_cfg.num_conv_layers
        kernel_size = de_cfg.kernel_size
        padding = de_cfg.padding
        use_maxpool = de_cfg.use_maxpool
        pool_size = de_cfg.pool_size
        pool_stride = de_cfg.pool_stride
        adaptive_pool_size = de_cfg.adaptive_pool_size
        mlp_hidden_dims = de_cfg.mlp_hidden_dims
        mlp_dropout = de_cfg.mlp_dropout
        conv_layers = []
        in_channels = 1
        for i in range(num_conv_layers):
            out_channels = hidden_channels * (2 ** i)
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.BatchNorm2d(out_channels))
            if i < num_conv_layers - 1 and use_maxpool:
                conv_layers.append(nn.MaxPool2d(pool_size, pool_stride))
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((adaptive_pool_size, adaptive_pool_size))
        final_conv_channels = hidden_channels * (2 ** (num_conv_layers - 1))
        mlp_input_dim = final_conv_channels * adaptive_pool_size * adaptive_pool_size        

        mlp_layers = []
        prev_dim = mlp_input_dim
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(mlp_dropout)
            ])
            prev_dim = hidden_dim
        mlp_layers.append(nn.Linear(prev_dim, self.d_e))
        
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, delta_theta):

        x = delta_theta.unsqueeze(0).unsqueeze(0)
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        e = self.mlp(x)
        return e.squeeze(0) 

class DomainEmbeddingExtractor:
    def __init__(self, frozen_backbone:nn.Module, cfg):
        self.frozen_backbone = frozen_backbone
        self.cfg = cfg
        pretrain_ds_names = cfg.pretrain.pretrain_datasets  
        if cfg.Fingerprint.task == 'node_cls':
            if cfg.Fingerprint.loss_type == 'contrastive':
                self.prob_loss = GraphProbContrastLoss(
                    mask_ratio=cfg.Fingerprint.contrast_loss.probe_mask_ratio,
                    recon_weight=cfg.Fingerprint.contrast_loss.probe_recon_weight,
                    neigh_weight=cfg.Fingerprint.contrast_loss.probe_neigh_weight,
                    detach_embed=cfg.Fingerprint.detach_embed
                )
            elif cfg.Fingerprint.loss_type == 'ce':
                self.prob_loss = nn.CrossEntropyLoss()
            else:
                raise ValueError(f"Unknown loss type {cfg.Fingerprint.loss_type} for Fingerprint")

        self.d_e = cfg.Fingerprint.compressed_dim
        self.device = self._device() 
        if self.cfg.Fingerprint.DE_type == 'conv':
            self.projection = ConvProjection(cfg).to(self.device)

        if isinstance(pretrain_ds_names, list):
            ds_names_str = '_'.join(pretrain_ds_names) # dsname1_dsname2_...
        else:
            ds_names_str = '_'.join(ast.literal_eval(str(pretrain_ds_names)))

        fingerprint_dir = os.path.join(cfg.dirs.fingerprint_storage, ds_names_str)
        self.cache_dir = Path(fingerprint_dir)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cached = False
        self._e = None
        self._B = None
        self._theta0 = None
        self._try_load_cache()  

    def _pth(self, name):
        return self.cache_dir/ name if self.cache_dir else None

    def _try_load_cache(self):
        if not self.cache_dir: 
            return
        cache_files = ['projection_state.pt', 'PreDomainEmbedding.pt', 'theta0.pt', 'delta_matrices.pt']
        if self.cfg.Fingerprint.DE_type == 'conv':
            if all((self._pth(f).exists() for f in cache_files)):
                # Load projection network state
                projection_state = torch.load(self._pth('projection_state.pt'), weights_only=False)
                self.projection.load_state_dict(projection_state)
                
                # Load embeddings and theta0
                self._e = torch.load(self._pth('PreDomainEmbedding.pt'), weights_only=False)
                self._theta0 = torch.load(self._pth('theta0.pt'), weights_only=False)
                self._delta_matrices = torch.load(self._pth('delta_matrices.pt'), weights_only=False)  # domain embeddings of pretraining graphs
                
                # Restore backbone to theta0
                self.frozen_backbone.load_state_dict(self._theta0, strict=False)
                self._cached = True
        elif self.cfg.Fingerprint.DE_type == 'pca':
            if (self._pth('B.pt').exists() and 
                self._pth('PreDomainEmbedding.pt').exists() and 
                self._pth('theta0.pt').exists()):
                self._B = torch.load(self._pth('B.pt'), weights_only=False)
                self._e = torch.load(self._pth('PreDomainEmbedding.pt'), weights_only=False)
                self._theta0 = torch.load(self._pth('theta0.pt'), weights_only=False)
                self.frozen_backbone.load_state_dict(self._theta0, strict=False)
                self._cached = True
        else:
            raise ValueError(f"Unknown Domain Embedder Type {self.cfg.Fingerprint.DE_type} for Domain Embeddings")


    def _device(self):
        if self.cfg.Fingerprint.device is not None:
            return torch.device(self.cfg.Fingerprint.device)
        try:
            return next(self.frozen_backbone.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _maybe_cache_path(self, pretrain_mode:str ,name:str): # persist fingerprints to disk
        if self.cfg.dirs.fingerprint_storage is None:
            return None
        os.makedirs(self.cfg.dirs.fingerprint_storage, exist_ok=True)
        return os.path.join(self.cfg.dirs.fingerprint_storage, pretrain_mode, name)
    
    def _save_cache(self):
        if not self.cache_dir: 
            return
        if self.cfg.Fingerprint.DE_type == 'conv':
            torch.save(self.projection.state_dict(), self._pth('projection_state.pt'))
            torch.save(self._e.cpu(), self._pth('PreDomainEmbedding.pt'))
            torch.save(self._theta0, self._pth('theta0.pt'))
        elif self.cfg.Fingerprint.DE_type == 'pca':
            torch.save(self._B.cpu(), self._pth('B.pt'))
            torch.save(self._e.cpu(), self._pth('PreDomainEmbedding.pt'))
            torch.save(self._theta0, self._pth('theta0.pt'))
        if self._delta_matrices is not None:
            torch.save([dm.cpu() for dm in self._delta_matrices], self._pth('delta_matrices.pt'))
        if hasattr(self, '_padded_delta_matrices') and self._padded_delta_matrices is not None:
            torch.save([dm.cpu() for dm in self._padded_delta_matrices], self._pth('padded_delta_matrices.pt'))
        if hasattr(self, '_original_shapes'):
            torch.save(self._original_shapes, self._pth('original_shapes.pt'))
        if hasattr(self, '_d_c_max'):
            torch.save(self._d_c_max, self._pth('d_c_max.pt'))
        self._cached = True

    @torch.no_grad()
    def _embed_nodes(self, data: Data) -> Tensor:
        self.frozen_backbone.eval()
        data = data.to(self.device)
        h, g = self.frozen_backbone(data)
        return h.detach(), g.detach()

    def _domain_subgraph(self, data:Data, idx: int):
        graph_mask = (data.batch == idx) 
        edge_mask = graph_mask[data.edge_index[0]] & graph_mask[data.edge_index[1]]
        node_idx = graph_mask.nonzero(as_tuple=False).view(-1) 
        if node_idx.numel() == 0:
            raise ValueError(f"Domain {idx} has no nodes")
        edge_index_i, _ = subgraph(node_idx, data.edge_index, relabel_nodes=True) 
        x_i = data.x[graph_mask]
        y_i = data.y[graph_mask]
        xe_i = data.xe[edge_mask] if hasattr(data, 'xe') else None
        return x_i, edge_index_i, node_idx, y_i, x_i.shape[0], xe_i
    
    def _sample_nodes(self, x, edge_index, y, max_nodes):
        num_nodes = x.shape[0]
        if num_nodes <= max_nodes:
            return x, edge_index, y
        perm = torch.randperm(num_nodes, device=x.device)[:max_nodes] 
        new_idx = torch.full((num_nodes,), -1, dtype=torch.long, device=x.device)
        new_idx[perm] = torch.arange(max_nodes, device=x.device)
        row,col = edge_index
        keep_mask = (new_idx[row] >= 0) & (new_idx[col] >= 0)  

        row = new_idx[row[keep_mask]]
        col = new_idx[col[keep_mask]]

        return x[perm], torch.stack([row, col], dim=0), y[perm]
    
    def _probe_grad4domain(self, data, H_all, domain_idx):
        model = self.frozen_backbone.to(self.device)
        model.train(False)

        if self._theta0 is None:
            self._theta0 = {k:v.clone().cpu() for k, v in self.frozen_backbone.state_dict().items()}


        x_i, edge_index_i, node_idx, y_i, num_nodes, xe_i = self._domain_subgraph(data, domain_idx)
        
        x_i, edge_index_i, y_i = self._sample_nodes(x_i, edge_index_i, y_i, self.cfg.Fingerprint.max_nodes if self.cfg.Fingerprint.get('max_nodes', None) is not None else num_nodes)

        x_i = x_i.to(self.device)
        edge_index_i = edge_index_i.to(self.device)
        y_i = y_i.to(self.device)
        xe_i = xe_i.to(self.device) if xe_i is not None else None
        domain_graph = Data(x=x_i,
                            edge_index=edge_index_i,
                            y=y_i,
                            xe=xe_i,
                            batch=torch.zeros(x_i.shape[0], dtype=torch.long, device=self.device))
        H_i, _ = model(domain_graph)  # [N_i, num_classes]

        if self.cfg.Fingerprint.loss_type == 'ce':
            loss = self.prob_loss(H_i, y_i)
        elif self.cfg.Fingerprint.loss_type == 'contrastive':
            loss = self.prob_loss(domain_graph, H_i)
        else:
            raise ValueError(f"Unknown loss type {self.cfg.Fingerprint.loss_type} for Fingerprint")
        
        loss.requires_grad = True
        model.zero_grad(set_to_none=True)
        loss.backward()

        if self.cfg.Fingerprint.DE_type == 'pca':
            grad_vec = flatten_grads(model, self.cfg.Fingerprint.require_grad_only).detach()

            delta = -self.cfg.Fingerprint.probe_lr * grad_vec  # scale by probe_lr
            del model
            return delta.cpu()
        elif self.cfg.Fingerprint.DE_type == 'conv':
            grad_matrix = None
            for name, param in model.named_parameters():
                if 'weight' in name and param.grad is not None:
                    grad_matrix = param.grad.detach().clone()
                    break
            if grad_matrix is None:
                grad_vec = flatten_grads(model, self.cfg.Fingerprint.require_grad_only).detach()
                d = x_i.shape[1]
                d_c = int(y_i.max().item()) + 1
                expected_size = d * d_c
                if grad_vec.numel() >= expected_size:
                    grad_matrix = grad_vec[:expected_size].view(d, d_c)
                else:
                    padded = torch.zeros(expected_size, device=self.device)
                    padded[:grad_vec.numel()] = grad_vec
                    grad_matrix = padded.reshape(d, d_c)
            else:
                if grad_matrix.shape[0] != x_i.shape[1]:
                    grad_matrix = grad_matrix.T
            
            delta_matrix = -self.cfg.Fingerprint.probe_lr * grad_matrix
            del model
            return delta_matrix.cpu()

    def _fit_pca(self, deltas):
        M, d_theta = deltas.shape 
        device = deltas.device


        if PCA is not None and M > 10:
            pca = PCA(n_components = self.d_e,
                      whiten=self.cfg.Fingerprint.pca_whiten,
                      svd_solver = 'full' if self.cfg.Fingerprint.pca_svd_full else 'randomized',
                      random_state=self.cfg.Fingerprint.random_state)

            comps = pca.fit_transform(deltas.cpu().numpy())

            if self.cfg.Fingerprint.l2_normalize:
                comps = comps / (np.linalg.norm(comps, axis=1, keepdims=True) + 1e-8)

            B = torch.from_numpy(pca.components_).to(device, dtype = deltas.dtype)  # [d_e, d_theta]
            e = torch.from_numpy(comps).to(device, dtype = deltas.dtype)
            return B,e
        else: # Use torch SVD
            mean = deltas.mean(dim=0, keepdim=True)
            Xc  =deltas - mean  
            U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)  
            B = Vh[:self.d_e]
            e = (U[:, :self.d_e] * S[:self.d_e])
            if self.cfg.Fingerprint.pca_whiten:
                e = e / (S[:self.d_e] + 1e-8)
            if self.cfg.Fingerprint.l2_normalize:
                e = F.normalize(e, p=2, dim=-1)
            return B, e


    def compute_fingerprints_pca(self, data):
        if self._cached:
            return self._e, self._B

        data = data.to(self.device)
        M = int(data.batch.max().item()) + 1 
        H_all, _  = self._embed_nodes(data) 

        deltas = []
        for i in range(M): 
            delta_i = self._probe_grad4domain(data, H_all, i)
            deltas.append(delta_i)
        deltas = torch.stack(deltas, dim=0) 
        B, e = self._fit_pca(deltas)
        ds_names_str = '_'.join(data.name_dict.keys()) 
        path_B = self._maybe_cache_path(ds_names_str, "B.pt")
        path_e = self._maybe_cache_path(ds_names_str, "PreDomainEmbedding.pt")
        self._B, self._e= B, e
        self._save_cache()
        return e, B
    
    def compute_fingerprints_conv(self, data):
        if self._cached:
            return self._e, None
        data = data.to(self.device)
        M = int(data.batch.max().item()) + 1
        delta_matrices = []  
        original_shapes = []
        for i in range(M):
            delta_i = self._probe_grad4domain(data, None, i)
            delta_matrices.append(delta_i)
            original_shapes.append(delta_i.shape)

        d = delta_matrices[0].shape[0]
        d_c_max = max(delta.shape[1] for delta in delta_matrices)
        padding_strategy = self.cfg.Fingerprint.DE.get('padding_strategy', 'zero')
        padding_noise_std = self.cfg.Fingerprint.DE.get('padding_noise_std', 0.01)
        padded_deltas = []
        for i, delta in enumerate(delta_matrices):
            d_c_i = delta.shape[1]
            if d_c_i< d_c_max:
                if padding_strategy == 'zero':
                    padding = torch.zeros(d, d_c_max - d_c_i, device=delta.device)
                elif padding_strategy == 'noise':
                    padding = torch.randn(d, d_c_max - d_c_i, device=delta.device) * padding_noise_std
                elif padding_strategy == 'repeat_last':
                    # Repeat the last column
                    last_col = delta[:, -1:].repeat(1, d_c_max - d_c_i)
                    padding = last_col
                else:
                    padding = torch.zeros(d, d_c_max-d_c_i, device=delta.device)
                delta_padded = torch.cat([delta, padding], dim=1)
            else:
                delta_padded = delta
            padded_deltas.append(delta_padded)
        
        self._delta_matrices = delta_matrices
        self._padded_delta_matrices = padded_deltas
        self.original_shapes = original_shapes
        self._d_c_max = d_c_max
        
        self._train_projection(padded_deltas)  

        embeddings = []
        for delta_padded in padded_deltas:
            delta_padded = delta_padded.to(self.device)
            with torch.no_grad():
                e_i = self.projection(delta_padded)
                if self.cfg.Fingerprint.l2_normalize:
                    e_i = F.normalize(e_i, p=2, dim=-1)
                embeddings.append(e_i)

        self._e = torch.stack(embeddings, dim=0)  

        self._save_cache()
        return self._e, None

    def _train_projection(self, delta_matrices):
        de_cfg = self.cfg.Fingerprint.DE
        num_epochs = de_cfg.train_epochs
        lr = de_cfg.train_lr
        diversity_weight = de_cfg.diversity_weight

        optimizer = torch.optim.Adam(self.projection.parameters(), lr=lr)

        deltas = [d.to(self.device) for d in delta_matrices]
        M = len(deltas)

        orig_dists = torch.zeros(M,M, device=self.device)

        for i in range(M):
            for j in range(i + 1, M):
                orig_dists[i,j] = torch.norm(deltas[i] - deltas[j], p="fro")
                orig_dists[j,i] = orig_dists[i,j]
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            embeddings = []
            for delta in deltas:
                e = self.projection(delta)
                if self.cfg.Fingerprint.l2_normalize:
                    e = F.normalize(e, p=2, dim=-1)
                embeddings.append(e)
            
            proj_dists = torch.zeros(M, M, device=self.device)
            for i in range(M):
                for j in range(i + 1, M):
                    proj_dists[i,j] = torch.norm(embeddings[i] - embeddings[j], p=2)
                    proj_dists[j,i] = proj_dists[i,j]
            
            loss = F.mse_loss(proj_dists, orig_dists)

            if M < self.d_e:
                E = torch.stack(embeddings, dim=0)  
                gram = E @ E.T
                diversity_loss = -torch.logdet(gram + 1e-6 * torch.eye(M, device=self.device))
                loss = loss + diversity_weight * diversity_loss
            loss.requires_grad = True
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}")


    def get_cached(self):
        if self.cfg.Fingerprint.DE_type == 'pca':
            if not self._cached:
                return None
            return self._e, self._B
        elif self.cfg.Fingerprint.DE_type == 'conv':
            if not self._cached:
                return None, None
            return self._e, None            

    @torch.no_grad()
    def fingerprint_unseen_pca(self, data_new: Data):
        if self._B is None or self._theta0 is None:
            raise RuntimeError('Compute domain embedding on pre‑training corpus first.')

        self.frozen_backbone.load_state_dict(self._theta0)
        self.frozen_backbone.to(self.device).train(False)
        data_new = data_new.to(self.device)
        edge_index_new = data_new.edge_index
        x = data_new.x
        H, _ = self.frozen_backbone(data_new)
        y = data_new.y
        if self.cfg.Fingerprint.loss_type == 'ce':
            loss = self.prob_loss(H, y)
        elif self.cfg.Fingerprint.loss_type == 'contrastive':
            loss = self.prob_loss(data_new, H)
        self.frozen_backbone.zero_grad(set_to_none=True)
        loss.backward()
        grad = flatten_grads(self.frozen_backbone, self.cfg.Fingerprint.require_grad_only).detach()
        delta = -self.cfg.Fingerprint.probe_lr * grad.cpu()
        e_new = (self._B @ delta).to(self.device)
        if self.cfg.Fingerprint.l2_normalize:
            e_new = F.normalize(e_new, p=2, dim=-1)
        return e_new

    def fingerprint_unseen_conv(self, data_new: Data):
        if self._theta0 is None:
            raise RuntimeError('Compute domain embedding on pre‑training corpus first.')
        
        self.frozen_backbone.load_state_dict(self._theta0)
        self.frozen_backbone.to(self.device).train(False)
        data_new = data_new.to(self.device)

        if not hasattr(data_new, 'batch'):
            data_new.batch = torch.zeros(data_new.x.shape[0], dtype=torch.long, device=self.device)
        

        H, _ = self.frozen_backbone(data_new)

        y = data_new.y
        if self.cfg.Fingerprint.loss_type == 'ce':
            loss = self.prob_loss(H, y)
        elif self.cfg.Fingerprint.loss_type == 'contrastive':
            loss = self.prob_loss(data_new, H)
        
        self.frozen_backbone.zero_grad(set_to_none=True)
        loss.requires_grad = True
        loss.backward()
        grad_matrix = None
        for name, param in self.frozen_backbone.named_parameters():
            if 'weight' in name and param.grad is not None:
                grad_matrix = param.grad.detach().clone()
                break
        if grad_matrix is None:
            grad_vec = flatten_grads(self.frozen_backbone, self.cfg.Fingerprint.require_grad_only).detach()
            d = data_new.x.shape[1]
            d_c = int(data_new.y.max().item()) + 1
            expected_size = d * d_c
            if grad_vec.numel() >= expected_size:
                grad_matrix = grad_vec[:expected_size].view(d, d_c)
            else:
                padded = torch.zeros(expected_size, device=self.device)
                padded[:grad_vec.numel()] = grad_vec
                grad_matrix = padded.reshape(d, d_c)
        else:
            if grad_matrix.shape[0] != data_new.x.shape[1]:
                grad_matrix = grad_matrix.T
        
        delta_new = -self.cfg.Fingerprint.probe_lr * grad_matrix

        if hasattr(self, '_d_c_max'):
            d = delta_new.shape[0]
            d_c_new = delta_new.shape[1]
            if d_c_new < self._d_c_max:
                padding = torch.zeros(d, self._d_c_max-d_c_new, device=self.device)
                delta_new = torch.cat([delta_new, padding], dim=1)
            elif d_c_new > self._d_c_max:
                delta_new = delta_new[:, :self._d_c_max]
        
        e_new = self.projection(delta_new)
        if self.cfg.Fingerprint.l2_normalize:
            e_new = F.normalize(e_new, p=2, dim=-1)
        return e_new


class DomainEmbedder(nn.Module):
    def __init__(self, frozen_backboneGNN, cfg):  
        super().__init__()
        self.frozen_backboneGNN = frozen_backboneGNN
        self.cfg = cfg
        self.dm_extractor = DomainEmbeddingExtractor(frozen_backboneGNN, cfg)
        self.dm_film = DomainFiLM(cfg)

    def forward(self, data, device):
        B = None
        if self.cfg.Fingerprint.DE_type == 'pca':
            e, B = self.dm_extractor.compute_fingerprints_pca(data)
            e = e.to(device)
            B = B.to(device)
        elif self.cfg.Fingerprint.DE_type == 'conv':
            e, _ = self.dm_extractor.compute_fingerprints_conv(data)
            e = e.to(device)
        gamma_f, beta_f, gamma_l, beta_l = self.dm_film(e)
        return e, (gamma_f, beta_f, gamma_l, beta_l), B
    
    @torch.no_grad()
    def fingerprint_unseen(self, data_new: Data):
        return self.dm_extractor.fingerprint_unseen_pca(data_new) if self.cfg.Fingerprint.DE_type == 'pca' else self.dm_extractor.fingerprint_unseen_conv(data_new)




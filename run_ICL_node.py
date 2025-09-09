#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from icl.icl import PrototypeInContextLearner
from utils.logging import logger
import torch_geometric.transforms as T
from torch_geometric.utils import degree

class GraphAwarePrototypeLearner(PrototypeInContextLearner):    
    def __init__(self, args):
        super().__init__(args)
        self.label_propagation = LabelPropagation(num_layers=3, alpha=0.9)  
        
    def compute_graph_aware_prototypes(self, 
                                      features: torch.Tensor,
                                      labels: torch.Tensor,
                                      edge_index: torch.Tensor,
                                      unique_labels: torch.Tensor,
                                      mask: torch.Tensor = None) -> torch.Tensor:
        if not self.args.no_lp:
            n_nodes = features.shape[0]
            n_classes = len(unique_labels)
            
            # Initialize soft labels
            soft_labels = torch.zeros(n_nodes, n_classes, device=features.device)
            
            if mask is not None:
                for idx, label in enumerate(unique_labels):
                    class_mask = mask & (labels == label)
                    if class_mask.sum() > 0:
                        soft_labels[class_mask, idx] = 1.0
            
            propagated_labels = self.label_propagation(soft_labels, edge_index)
            
            prototypes = []
            for idx, label in enumerate(unique_labels):
                class_weights = propagated_labels[:, idx]
                
                weight_threshold = 0.1   # default 0.1
                valid_mask = class_weights > weight_threshold
                
                if valid_mask.sum() > 0:
                    weighted_features = features[valid_mask] * class_weights[valid_mask].unsqueeze(-1)
                    prototype = weighted_features.sum(dim=0) / class_weights[valid_mask].sum()
                else:
                    class_mask = labels == label
                    if mask is not None:
                        class_mask = class_mask & mask
                    if class_mask.sum() > 0:
                        prototype = features[class_mask].mean(dim=0)
                    else:
                        prototype = torch.zeros_like(features[0])
                
                prototypes.append(prototype)
        else:
            row, col = edge_index
            degrees = degree(col, features.size(0), dtype=features.dtype)
            degrees = degrees / (degrees.mean() + 1e-6)
            prototypes = []
            for label in unique_labels:
                class_mask = labels == label
                if mask is not None:
                    class_mask = class_mask & mask
                
                if class_mask.sum() > 0:
                    class_features = features[class_mask]
                    class_degrees = degrees[class_mask]
                    
                    weights = F.softmax(class_degrees, dim=0)
                    prototype = (class_features * weights.unsqueeze(1)).sum(dim=0)
                else:
                    prototype = torch.zeros_like(features[0])
                
                prototypes.append(prototype)                    
        
        prototypes = torch.stack(prototypes, dim=0)
        prototypes = F.normalize(prototypes, p=2, dim=-1)
        
        return prototypes
    
    def adaptive_distance(self, 
                         query_features,
                         prototypes,
                         confidence = None) -> torch.Tensor:
        query_norm = F.normalize(query_features, p=2, dim=-1)
        proto_norm = F.normalize(prototypes, p=2, dim=-1)
        
        cos_sim = torch.mm(query_norm, proto_norm.t())
        
        eucl_dist = torch.cdist(query_norm, proto_norm, p=2)
        eucl_sim = 1.0 / (1.0 + eucl_dist)  
        
        query_var = query_features.var(dim=-1, keepdim=True)
        alpha = torch.sigmoid(query_var)  
        
        combined_sim = alpha * cos_sim + (1 - alpha) * eucl_sim
        
        if confidence is not None:
            combined_sim = combined_sim * confidence.unsqueeze(0)
        
        return combined_sim
    
    def semi_supervised_refinement(self,
                                   features: torch.Tensor,
                                   initial_predictions: torch.Tensor,
                                   edge_index: torch.Tensor,
                                   support_mask: torch.Tensor,
                                   num_iterations: int = 5):
        n_nodes = features.shape[0]
        n_classes = initial_predictions.max().item() + 1
        if not self.args.no_lp:
            pseudo_labels = torch.zeros(n_nodes, n_classes, device=features.device)
            for i in range(n_nodes):
                if support_mask[i]:
                    pseudo_labels[i, initial_predictions[i]] = 1.0
                else:
                    pseudo_labels[i, initial_predictions[i]] = 0.8
            
            for _ in range(num_iterations):
                propagated = self.label_propagation(pseudo_labels, edge_index)
                
                pseudo_labels[~support_mask] = 0.9 * propagated[~support_mask] + 0.1 * pseudo_labels[~support_mask]
                
                pseudo_labels = F.softmax(pseudo_labels, dim=-1)
            
            refined_predictions = pseudo_labels.argmax(dim=-1)
        else:
            edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=n_nodes)
            refined_predictions = initial_predictions.clone()
            
            for _ in range(num_iterations):
                new_predictions = refined_predictions.clone()
                
                for node in torch.where(~support_mask)[0]:
                    neighbors = edge_index_with_loops[1][edge_index_with_loops[0] == node]
                    
                    if len(neighbors) > 0:
                        neighbor_preds = refined_predictions[neighbors]
                        
                        unique_preds, counts = torch.unique(neighbor_preds, return_counts=True)
                        majority_pred = unique_preds[counts.argmax()]
                        
                        if torch.rand(1).item() > 0.3:  
                            new_predictions[node] = majority_pred
                
                refined_predictions = new_predictions            
        
        return refined_predictions
    
    @torch.no_grad()
    def enhanced_prototype_inference(self,
                                    graph_data,
                                    k_shot: int = 5,
                                    seed: int = 42,
                                    use_graph_aware: bool = True,
                                    use_adaptive_distance: bool = True,
                                    use_refinement: bool = True) -> Dict:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        logger.info("Running enhanced prototype-based inference")
        logger.info(f"Settings: graph_aware={use_graph_aware}, adaptive={use_adaptive_distance}, refinement={use_refinement}")
        
        domain_embedding = self.compute_domain_embedding(graph_data)
        gamma_f, beta_f, gamma_l, beta_l = self.domain_embedder.dm_film(domain_embedding.unsqueeze(0))
        gamma_f, beta_f = gamma_f.squeeze(0), beta_f.squeeze(0)
        gamma_l, beta_l = gamma_l.squeeze(0), beta_l.squeeze(0)
        
        H, _ = self.backbone_gnn.encode(
            graph_data.x,
            graph_data.edge_index,
            graph_data.xe if hasattr(graph_data, 'xe') else None,
            graph_data.batch if hasattr(graph_data, 'batch') else None
        )
        
        z_all = gamma_f * H + beta_f
        z_all = F.normalize(z_all, p=2, dim=-1)
        
        labels = graph_data.y
        unique_labels = torch.unique(labels).sort()[0]
        num_classes = len(unique_labels)
        
        if hasattr(graph_data, 'train_mask') and hasattr(graph_data, 'test_mask'):
            train_mask = graph_data.train_mask
            test_mask = graph_data.test_mask
        else:
            n_nodes = labels.shape[0]
            perm = torch.randperm(n_nodes)
            split_idx = int(0.8 * n_nodes)
            train_mask = torch.zeros(n_nodes, dtype=torch.bool, device=self.device)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool, device=self.device)
            train_mask[perm[:split_idx]] = True
            test_mask[perm[split_idx:]] = True
        
        support_mask = torch.zeros_like(train_mask)
        support_labels = []
        
        
        for class_label in unique_labels:
            class_train_mask = train_mask & (labels == class_label)
            class_train_indices = class_train_mask.nonzero(as_tuple=False).squeeze(-1)
            
            if class_train_indices.numel() > 0:
                n_samples = min(k_shot, len(class_train_indices))
                perm = torch.randperm(len(class_train_indices))[:n_samples]
                selected = class_train_indices[perm]
                support_mask[selected] = True
                support_labels.extend([class_label.item()] * n_samples)
        
        logger.info(f"Support set: {support_mask.sum().item()} nodes from {num_classes} classes")
        
        if use_graph_aware:
            prototypes = self.compute_graph_aware_prototypes(
                z_all, labels, graph_data.edge_index, unique_labels, support_mask
            )
        else:
            support_features = z_all[support_mask]
            support_labels_tensor = labels[support_mask]
            prototypes = self.compute_prototypes(support_features, support_labels_tensor, unique_labels)
        
        if use_adaptive_distance:
            similarities = self.adaptive_distance(z_all, prototypes)
        else:
            similarities = self.prototype_distance(z_all, prototypes, 'cosine')
        
        initial_predictions = similarities.argmax(dim=1)
        
        predicted_labels = torch.zeros_like(labels)
        for i, label in enumerate(unique_labels):
            mask = initial_predictions == i
            predicted_labels[mask] = label
        
        if use_refinement and graph_data.edge_index.shape[1] > 0:
            refined_predictions = self.semi_supervised_refinement(
                z_all, predicted_labels, graph_data.edge_index, support_mask, num_iterations=5
            )
            predicted_labels[~support_mask] = refined_predictions[~support_mask]
        
        test_predictions = predicted_labels[test_mask].cpu().numpy()
        test_true = labels[test_mask].cpu().numpy()
        
        accuracy = (test_predictions == test_true).mean()
        
        per_class_acc = {}
        for class_label in unique_labels:
            class_mask = test_true == class_label.item()
            if class_mask.sum() > 0:
                class_acc = (test_predictions[class_mask] == test_true[class_mask]).mean()
                per_class_acc[class_label.item()] = float(class_acc)
        
        test_similarities = similarities[test_mask]
        test_probs = F.softmax(test_similarities / 0.1, dim=1)
        confidences = test_probs.max(dim=1)[0].cpu().numpy()
        
        results = {
            'accuracy': float(accuracy),
            'per_class_accuracy': per_class_acc,
            'mean_confidence': float(confidences.mean()),
            'num_test_samples': test_mask.sum().item(),
            'num_classes': num_classes,
            'k_shot': k_shot,
            'use_graph_aware': use_graph_aware,
            'use_adaptive_distance': use_adaptive_distance,
            'use_refinement': use_refinement,
            'label_propagation': not self.args.no_lp
        }
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Mean confidence: {confidences.mean():.4f}")
        
        return results

class LabelPropagation(MessagePassing):    
    def __init__(self, num_layers: int = 3, alpha: float = 0.9):
        super().__init__(aggr='add')
        
        self.num_layers = num_layers
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        x_orig = x.clone()
        
        for _ in range(self.num_layers):
            x = self.propagate(edge_index, x=x, norm=norm)
            x = self.alpha * x + (1 - self.alpha) * x_orig
        
        return x
    
    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='generated_files/output/G-Align/Aug13-0:14-97cc0c8c/final_gfm_model.pt')
    parser.add_argument('--dataset', type=str, default='cora')  
    parser.add_argument('--k_shot', type=int, default=5)  # 
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--gpu_id', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--norm_feat', action='store_true', default=False)
    parser.add_argument('--add_loop', action='store_true', default=False)
    parser.add_argument('--undirected', action='store_true', default=False)
    parser.add_argument('--no_lp', action='store_true', default=False)
    args = parser.parse_args()
    learner = GraphAwarePrototypeLearner(args)
    
    if args.dataset not in learner.cfg['_ds_meta_data'].keys():
        if args.dataset == 'computers':
            learner.cfg['_ds_meta_data'][args.dataset] = ('pyg, Amazon.Computers')
        elif args.dataset == 'cora':
            learner.cfg['_ds_meta_data'][args.dataset] = ('pyg, Planetoid.Cora')
        elif args.dataset == 'ogbn-products':
            learner.cfg['_ds_meta_data'][args.dataset] = ('ogb.nodeproppred, PygNodePropPredDataset.ogbn-products')
        elif args.dataset == 'Roman-empire':
            learner.cfg['_ds_meta_data'][args.dataset] = ('pyg, HeterophilousGraphDataset.Roman-empire')
        elif args.dataset == 'usa':
            learner.cfg['_ds_meta_data'][args.dataset] = ('pyg, Airports.USA')
        elif args.dataset == 'paris':
            learner.cfg['_ds_meta_data'][args.dataset] = ('pyg, CityNetwork.paris')
        elif args.dataset == 'facebookpagepage':
            learner.cfg['_ds_meta_data'][args.dataset] = ('pyg, Social.FacebookPagePage')
        elif args.dataset == 'flickr':
            learner.cfg['_ds_meta_data'][args.dataset] = ('pyg, AttributedGraphDataset.Flickr')
        elif args.dataset == 'email':
            learner.cfg['_ds_meta_data'][args.dataset] = ('pyg, Email.EmailEUCore')
        elif args.dataset == 'twitch-de':
            learner.cfg['_ds_meta_data'][args.dataset] = ('pyg, Twitch.DE')
        elif args.dataset == 'reddit':
            learner.cfg['_ds_meta_data'][args.dataset] = ('pyg, Social.Reddit')
        elif args.dataset == 'blogcatalog':
            learner.cfg['_ds_meta_data'][args.dataset] = ('pyg, AttributedGraphDataset.BlogCatalog')
        elif args.dataset == 'deezereurope':
            learner.cfg['_ds_meta_data'][args.dataset] = ('pyg, Deezer.DeezerEurope')
        elif args.dataset == 'physics':
            learner.cfg['_ds_meta_data'][args.dataset] = ('pyg, Coauthor.Physics')
        elif args.dataset == 'weibo':
            learner.cfg['_ds_meta_data'][args.dataset] = ('pyg, AttributedGraphDataset.TWeibo')
        elif args.dataset == 'twitter':
            learner.cfg['_ds_meta_data'][args.dataset] = ('pyg, AttributedGraphDataset.Twitter')
        elif args.dataset == 'facebook':
            learner.cfg['_ds_meta_data'][args.dataset] = ('pyg, AttributedGraphDataset.Facebook')
        elif args.dataset == 'fm':
            learner.cfg['_ds_meta_data'][args.dataset] = ('pyg, Social.LastFMAsia')
    graph_data = learner.load_downstream_graph(args.dataset)
    if args.norm_feat:
        graph_data = T.NormalizeFeatures()(graph_data)
    if args.add_loop:
        graph_data = T.AddSelfLoops()(graph_data)
    if args.undirected:
        graph_data = T.ToUndirected()(graph_data)
    logger.info("="*60)
    logger.info("In-Context Node Classification")
    logger.info("="*60)
    
    configs = [
        (False, False, False),  
        (True, False, False),   
        (False, True, False),   
        (False, False, True),  
        (True, True, False),   
        (True, True, True),    
    ]
    
    config_names = [
        "Baseline",
        "Graph-aware",
        "Adaptive distance",
        "Refinement",
        "Graph + Adaptive",
        "All enhancements"
    ]
    
    results_summary = []
    
    for (use_graph, use_adaptive, use_refine), name in zip(configs, config_names):
        logger.info(f"\nTesting: {name}")
        
        accuracies = []
        for run in range(min(5, args.n_runs)):
            results = learner.enhanced_prototype_inference(
                graph_data,
                k_shot=args.k_shot,
                seed=42+run,
                use_graph_aware=use_graph,
                use_adaptive_distance=use_adaptive,
                use_refinement=use_refine
            )
            accuracies.append(results['accuracy'])
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        results_summary.append({
            'config': name,
            'mean_acc': mean_acc,
            'std_acc': std_acc
        })
        
        print(f"  {name}: {mean_acc:.4f} ± {std_acc:.4f}")
    
    print("\n" + "="*60)
    print("SUMMARY - Best Configuration:")
    print("="*60)
    
    best_config = max(results_summary, key=lambda x: x['mean_acc'])
    print(f"Best: {best_config['config']}")
    print(f"Accuracy: {best_config['mean_acc']:.4f} ± {best_config['std_acc']:.4f}")


if __name__ == "__main__":
    main()
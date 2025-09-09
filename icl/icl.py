import torch
import torch.nn as nn
import torch.nn.functional as F
import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from utils.logging import logger
from data_process.data import SingleGraphDataset
from data_process.datahelper import refine_dataset, span_node_and_edge_idx, filter_unnecessary_attrs, SentenceEncoder
from model.base import BackboneGNN
from model.fingerprint import DomainEmbedder
import numpy as np
from typing import Dict
from torch_geometric.data import Data
import os.path as osp
from tqdm import tqdm
import argparse
from sklearn.decomposition import PCA
from torch_geometric.utils import k_hop_subgraph, subgraph
from torch_geometric.loader import NeighborLoader
from utils.utils import extract_classes_subgraph

class PrototypeInContextLearner:
    
    def __init__(self, args):
        self.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else "cpu")
        self.model_path = args.model_path
        self.args = args
        
        logger.info(f"Loading pretrained model from {self.model_path}")
        self.model_state = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        self.cfg = self.model_state['config']
        self.L_max = self.model_state['L_max']
        self.pretrain_datasets = self.model_state['pretrain_datasets']
        self.pretrain_tasks = self.model_state.get('pretrain_tasks', [])
        self._resolve_config_paths()
        self._setup_model()
            
    def _resolve_config_paths(self):
        import os
        from omegaconf import OmegaConf
        
        project_root = os.getcwd()
        
        try:
            config_dict = OmegaConf.to_container(self.cfg, resolve=False)
            
            if 'dirs' in config_dict and config_dict['dirs']:
                dirs = config_dict['dirs']
                
                if 'fingerprint_storage' in dirs:
                    dirs['fingerprint_storage'] = os.path.join(project_root, 'generated_files', 'fingerprint')
                
                if 'data_storage' in dirs:
                    dirs['data_storage'] = os.path.join(project_root, 'datasets')
                
                if 'output' in dirs:
                    model_name = 'G-Align'
                    if 'model' in config_dict and 'name' in config_dict['model']:
                        model_name = config_dict['model']['name']
                    dirs['output'] = os.path.join(project_root, 'generated_files', 'output', model_name)
            
            self.cfg = OmegaConf.create(config_dict)
            
        except Exception as e:
            logger.warning(f"Failed to resolve config paths: {e}. Using fallback paths.")
            from omegaconf import DictConfig
            if not hasattr(self.cfg, 'dirs'):
                self.cfg.dirs = DictConfig({})
            
            self.cfg.dirs.fingerprint_storage = os.path.join(project_root, 'generated_files', 'fingerprint')
            self.cfg.dirs.data_storage = os.path.join(project_root, 'datasets')
            self.cfg.dirs.output = os.path.join(project_root, 'generated_files', 'output', 'G-Align')

    def _setup_model(self):
        input_dim = self.model_state['combined_graphs_info']['num_features']
        
        self.backbone_gnn = BackboneGNN(in_dim=input_dim, num_classes=self.L_max, cfg=self.cfg)
        self.backbone_gnn.load_state_dict(self.model_state['backbone_state_dict'])
        self.backbone_gnn.to(self.device).eval()
        logger.info(f"Loaded backbone with {self.backbone_gnn.n_layers} layers")
        
        self.frozen_backbone = BackboneGNN(in_dim=input_dim, num_classes=self.L_max, cfg=self.cfg)
        self.frozen_backbone.n_layers = 1
        self.frozen_backbone.readout_proj = False
        
        if 'frozen_backbone_state_dict' in self.model_state:
            self.frozen_backbone.load_state_dict(self.model_state['frozen_backbone_state_dict'], strict=False)
        else:
            state_dict = {}
            for name, param in self.backbone_gnn.state_dict().items():
                if 'gnns.0' in name or 'bns.0' in name:
                    state_dict[name] = param
            self.frozen_backbone.load_state_dict(state_dict, strict=False)
        
        self.frozen_backbone.to(self.device).eval()
        
        self.domain_embedder = DomainEmbedder(self.frozen_backbone, self.cfg)
        
        if self.cfg.Fingerprint.DE_type == 'conv' and 'domain_embedder_projection_state' in self.model_state:
            self.domain_embedder.dm_extractor.projection.load_state_dict(
                self.model_state['domain_embedder_projection_state']
            )
            
            if 'domain_embedder_delta_matrices' in self.model_state:
                delta_matrices = self.model_state['domain_embedder_delta_matrices']
                if isinstance(delta_matrices, list) and len(delta_matrices) > 0:
                    d_c_max = max(dm.shape[1] for dm in delta_matrices)
                    self.domain_embedder.dm_extractor._d_c_max = d_c_max
        
        self.domain_embedder.dm_extractor._theta0 = self.model_state.get('domain_embedder_theta0')
        self.domain_embedder.dm_extractor._e = self.model_state.get('domain_embedder_e')
        if self.cfg.Fingerprint.DE_type == 'pca':
            self.domain_embedder.dm_extractor._B = self.model_state.get('domain_embedder_B')
        
        self.domain_embedder.dm_extractor._cached = True
        
        film_state = {k.replace('de.dm_film.', ''): v for k, v in self.model_state['model_state_dict'].items() 
                     if 'de.dm_film' in k}
        if film_state:
            self.domain_embedder.dm_film.load_state_dict(film_state)
        
        self.domain_embedder.to(self.device).eval()
        
        logger.info("Model components initialized successfully")
    
    def load_downstream_graph(self, dataset_name: str, data_path: str = None) -> Data:
        logger.info(f"Loading downstream dataset: {dataset_name}")
        
        if data_path is None:
            data_path = self.cfg.dirs.data_storage
        
        llm_encoder = SentenceEncoder(self.cfg.llm_name, batch_size=self.cfg.llm_b_size)
        
        dataset = SingleGraphDataset(
            self.cfg, 
            name=dataset_name,
            llm_encoder=llm_encoder,
            load_text=False
        )
        
        dataset = refine_dataset(dataset)
        dataset = span_node_and_edge_idx(dataset)
        dataset = filter_unnecessary_attrs(dataset)
        
        pca_x = self.dimension_align(dataset)
        
        # Handle edge attributes
        if hasattr(dataset.data, 'xe'):
            xe = dataset.data.xe
            if xe.dim() == 1:
                xe = xe.unsqueeze(1)
            if xe.shape[1] == 1 and pca_x.shape[1] > 1:
                xe = xe.expand(-1, pca_x.shape[1])
        else:
            num_edges = dataset.edge_index.shape[1]
            xe = torch.zeros(num_edges, pca_x.shape[1])
        
        graph_data = Data(
            x=pca_x,
            edge_index=dataset.edge_index,
            y=dataset.labels,
            xe=xe,
            train_mask=dataset.train_mask,
            test_mask=dataset.test_mask,
            val_mask=dataset.val_mask
        )

        if dataset_name == 'blogcatalog':
            graph_data = extract_classes_subgraph(graph_data, target_classes = [0,2,3,4,5])
        if dataset_name == 'ogbn-products':
            if graph_data.y.dim() > 1:
                graph_data.y = graph_data.y.squeeze()
            num_sample_per_class = self.args.k_shot * 10  
            sampled_nodes = []
            
            for class_id in range(47):  
                class_mask =  (graph_data.y == class_id)
                class_nodes = torch.where(class_mask)[0]
                if len(class_nodes) > 0:
                    n_sample = min(num_sample_per_class, len(class_nodes))
                    sampled = class_nodes[torch.randperm(len(class_nodes))[:n_sample]]
                    sampled_nodes.append(sampled)
            if len(sampled_nodes) == 0:
                raise ValueError("No nodes were sampled!")  
            sampled_nodes = torch.cat(sampled_nodes)
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(sampled_nodes, num_hops=1, edge_index=graph_data.edge_index, num_nodes=graph_data.x.shape[0], relabel_nodes=True)
            graph_data = Data(x=graph_data.x[subset],edge_index=edge_index,y=graph_data.y[subset],xe=torch.zeros(edge_index.shape[1], dtype=torch.long))

        if dataset_name == 'weibo':

            if graph_data.y.dim() > 1:
                graph_data.y = graph_data.y.squeeze()
            num_sample_per_class = self.args.k_shot * 10  
            sampled_nodes = []
            
            for class_id in range(8):  
                class_mask =  (graph_data.y == class_id)
                class_nodes = torch.where(class_mask)[0]
                if len(class_nodes) > 0:
                    n_sample = min(num_sample_per_class, len(class_nodes))
                    sampled = class_nodes[torch.randperm(len(class_nodes))[:n_sample]]
                    sampled_nodes.append(sampled)
            if len(sampled_nodes) == 0:
                raise ValueError("No nodes were sampled!")  
            sampled_nodes = torch.cat(sampled_nodes)
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(sampled_nodes, num_hops=1, edge_index=graph_data.edge_index, num_nodes=graph_data.x.shape[0], relabel_nodes=True)

            graph_data = Data(x=graph_data.x[subset],edge_index=edge_index,y=graph_data.y[subset],xe=torch.zeros(edge_index.shape[1], dtype=torch.long))


        if dataset_name == 'Roman-empire':
            train_mask = graph_data.train_mask[:,0]
            val_mask = graph_data.val_mask[:,0]
            test_mask = graph_data.test_mask[:,0]
            graph_data.train_mask = train_mask
            graph_data.val_mask = val_mask
            graph_data.test_mask = test_mask



        if not hasattr(graph_data, 'batch'):
            graph_data.batch = torch.zeros(graph_data.x.shape[0], dtype=torch.long)
        

        return graph_data.to(self.device)

    def dimension_align(self, ds):
        unify_dim = self.cfg.unify_dim if self.cfg.unify_dim else 64
        pca_cache_path = osp.join(ds.processed_dir, f"pca_{unify_dim}.pt")
        
        if osp.exists(pca_cache_path):
            pca_x = torch.load(pca_cache_path, weights_only=False)
        else:
            if ds.data.xn.shape[1] == unify_dim:
                pca_x = ds.data.xn.clone()
            else:
                x_np = ds.data.xn.cpu().numpy()
                pca = PCA(n_components=unify_dim)
                projected = pca.fit_transform(x_np)
                pca_x = torch.from_numpy(projected).float()
                torch.save(pca_x, pca_cache_path)
        return pca_x

    @torch.no_grad()
    def compute_domain_embedding(self, graph_data: Data) -> torch.Tensor:
        logger.info("Computing domain embedding for new graph...")
        
        if not hasattr(graph_data, 'batch') or graph_data.batch is None:
            graph_data.batch = torch.zeros(graph_data.x.shape[0], dtype=torch.long, device=self.device)
        
        if self.domain_embedder.dm_extractor._theta0 is not None:
            self.frozen_backbone.load_state_dict(self.domain_embedder.dm_extractor._theta0, strict=False)
        
        domain_embedding = self.domain_embedder.fingerprint_unseen(graph_data)
        
        logger.info(f"Domain embedding shape: {domain_embedding.shape}")
        return domain_embedding
    
    def compute_prototypes(self, features: torch.Tensor, labels: torch.Tensor, 
                          unique_labels: torch.Tensor) -> torch.Tensor:
        prototypes = []
        
        for class_label in unique_labels:
            class_mask = (labels == class_label)
            if class_mask.sum() > 0:
                class_features = features[class_mask]
                prototype = class_features.mean(dim=0)
            else:
                prototype = torch.zeros_like(features[0])
            
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes, dim=0)
        
        prototypes = F.normalize(prototypes, p=2, dim=-1)
        
        return prototypes
    
    def prototype_distance(self, query_features: torch.Tensor, 
                          prototypes: torch.Tensor,
                          distance_metric: str = 'euclidean') -> torch.Tensor:
        if distance_metric == 'euclidean':
            distances = torch.cdist(query_features, prototypes, p=2)
            return -distances  
        
        elif distance_metric == 'cosine':
            query_norm = F.normalize(query_features, p=2, dim=-1)
            proto_norm = F.normalize(prototypes, p=2, dim=-1)
            similarities = torch.mm(query_norm, proto_norm.t())
            return similarities
        
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
    
    @torch.no_grad()
    def prototype_inference(self, 
                           graph_data: Data, 
                           k_shot: int = 5,
                           seed: int = 42,
                           distance_metric: str = 'cosine',
                           use_label_prototypes: bool = False) -> Dict:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        logger.info(f"Performing {k_shot}-shot prototype-based learning")
        logger.info(f"Distance metric: {distance_metric}")
        
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
        
        logger.info(f"Number of classes: {num_classes}")
        logger.info(f"Unique labels: {unique_labels.tolist()}")
        
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
        
        support_indices = []
        support_labels = []
        
        for class_label in unique_labels:
            class_train_mask = train_mask & (labels == class_label)
            class_train_indices = class_train_mask.nonzero(as_tuple=False).squeeze(-1)
            
            if class_train_indices.numel() == 0:
                logger.warning(f"Class {class_label} has no training samples!")
                continue
            
            if len(class_train_indices) < k_shot:
                logger.warning(f"Class {class_label} has only {len(class_train_indices)} train samples")
                selected = class_train_indices
            else:
                perm = torch.randperm(len(class_train_indices))[:k_shot]
                selected = class_train_indices[perm]
            
            support_indices.extend(selected.tolist())
            support_labels.extend([class_label.item()] * len(selected))
        
        support_indices = torch.tensor(support_indices, device=self.device)
        support_labels = torch.tensor(support_labels, device=self.device)
        
        logger.info(f"Support set: {len(support_indices)} nodes")
        
        support_features = z_all[support_indices]
        prototypes = self.compute_prototypes(support_features, support_labels, unique_labels)
        
        logger.info(f"Computed {len(prototypes)} class prototypes")
        
        if use_label_prototypes and hasattr(self, 'E_label'):
            E_label = self.model_state['model_state_dict'].get('E_lab')
            if E_label is not None:
                E_label = E_label.to(self.device)
                U_base = E_label[:num_classes]
                U_aligned = gamma_l * U_base + beta_l
                U_aligned = F.normalize(U_aligned, p=2, dim=-1)
                
                alpha = 0.5 
                prototypes = alpha * prototypes + (1 - alpha) * U_aligned
                prototypes = F.normalize(prototypes, p=2, dim=-1)
                logger.info("Combined feature and label prototypes")
        
        test_indices = test_mask.nonzero(as_tuple=False).squeeze(-1)
        
        if test_indices.numel() == 0:
            logger.error("No test samples found!")
            return {'accuracy': 0.0, 'error': 'No test samples'}
        
        predictions = []
        confidences = []
        true_labels = []
        distances_all = []
        
        logger.info(f"Evaluating on {len(test_indices)} test nodes...")
        
        batch_size = 100
        for batch_start in tqdm(range(0, len(test_indices), batch_size), desc="Inference"):
            batch_end = min(batch_start + batch_size, len(test_indices))
            batch_indices = test_indices[batch_start:batch_end]
            
            query_features = z_all[batch_indices]
            
            distances = self.prototype_distance(query_features, prototypes, distance_metric)
            
            if distance_metric == 'euclidean':
                pred_indices = distances.argmax(dim=1)
                probs = F.softmax(distances / 0.1, dim=1)  # Temperature scaling
            else:
                pred_indices = distances.argmax(dim=1)
                probs = F.softmax(distances / 0.1, dim=1)
            
            for i, (idx, pred_idx) in enumerate(zip(batch_indices, pred_indices)):
                predicted_label = unique_labels[pred_idx].item()
                confidence = probs[i, pred_idx].item()
                true_label = labels[idx].item()
                
                predictions.append(predicted_label)
                confidences.append(confidence)
                true_labels.append(true_label)
                distances_all.append(distances[i].cpu().numpy())
        
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        confidences = np.array(confidences)
        
        accuracy = (predictions == true_labels).mean()
        
        per_class_acc = {}
        for class_label in unique_labels:
            class_mask = true_labels == class_label.item()
            if class_mask.sum() > 0:
                class_acc = (predictions[class_mask] == true_labels[class_mask]).mean()
                per_class_acc[class_label.item()] = float(class_acc)
        
        # Prediction distribution
        pred_distribution = {}
        for label in unique_labels:
            pred_distribution[label.item()] = int((predictions == label.item()).sum())
        
        results = {
            'accuracy': float(accuracy),
            'per_class_accuracy': per_class_acc,
            'mean_confidence': float(confidences.mean()),
            'predictions': predictions,
            'true_labels': true_labels,
            'confidences': confidences,
            'num_test_samples': len(test_indices),
            'num_classes': num_classes,
            'k_shot': k_shot,
            'prediction_distribution': pred_distribution,
            'distance_metric': distance_metric,
            'use_label_prototypes': use_label_prototypes
        }
        
        logger.info(f"Overall Accuracy: {accuracy:.4f}")
        logger.info(f"Mean Confidence: {confidences.mean():.4f}")
        logger.info(f"Prediction distribution: {pred_distribution}")
        
        return results
    
    def evaluate_multiple_runs(self,
                              graph_data: Data,
                              k_shot: int = 5,
                              n_runs: int = 10,
                              distance_metric: str = 'cosine',
                              use_label_prototypes: bool = False) -> Dict:
        logger.info(f"Running {n_runs} evaluations with different support sets")
        logger.info(f"Settings: {k_shot}-shot, {distance_metric} distance")
        
        all_accuracies = []
        all_confidences = []
        all_per_class = []
        
        for run in range(n_runs):
            results = self.prototype_inference(
                graph_data, 
                k_shot=k_shot, 
                seed=42+run,
                distance_metric=distance_metric,
                use_label_prototypes=use_label_prototypes
            )
            
            if 'error' not in results:
                all_accuracies.append(results['accuracy'])
                all_confidences.append(results['mean_confidence'])
                all_per_class.append(results['per_class_accuracy'])
        
        if not all_accuracies:
            return {'error': 'All runs failed'}
        
        # Aggregate per-class accuracies
        mean_per_class = {}
        if all_per_class:
            all_classes = set()
            for pc in all_per_class:
                all_classes.update(pc.keys())
            
            for class_id in all_classes:
                class_accs = [pc.get(class_id, 0) for pc in all_per_class]
                mean_per_class[class_id] = {
                    'mean': float(np.mean(class_accs)),
                    'std': float(np.std(class_accs))
                }
        
        return {
            'mean_accuracy': float(np.mean(all_accuracies)),
            'std_accuracy': float(np.std(all_accuracies)),
            'mean_confidence': float(np.mean(all_confidences)),
            'std_confidence': float(np.std(all_confidences)),
            'per_class_accuracy': mean_per_class,
            'all_accuracies': all_accuracies,
            'k_shot': k_shot,
            'n_runs': n_runs,
            'distance_metric': distance_metric,
            'use_label_prototypes': use_label_prototypes
        }


def main():
    parser = argparse.ArgumentParser(description="G-Align Prototype-based In-Context Learning")
    parser.add_argument('--model_path', type=str, default='generated_files/output/G-Align/final_gfm_model.pt',
                       help='Path to pretrained model checkpoint')
    parser.add_argument('--dataset', type=str, default='cora',
                       help='Downstream dataset name')
    parser.add_argument('--k_shot', type=int, default=5,
                       help='Number of support examples per class')
    parser.add_argument('--n_runs', type=int, default=10,
                       help='Number of evaluation runs')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--distance_metric', type=str, default='cosine',
                       choices=['euclidean', 'cosine'],
                       help='Distance metric for prototypes')
    parser.add_argument('--use_label_prototypes', action='store_true',
                       help='Combine feature and label prototypes')
    parser.add_argument('--compare_metrics', action='store_true',
                       help='Compare different distance metrics')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize the prototype-based learner
    logger.info("="*60)
    logger.info("G-Align Prototype-based In-Context Learning")
    logger.info("="*60)
    
    learner = PrototypeInContextLearner(args)
    
    # Load downstream dataset
    graph_data = learner.load_downstream_graph(args.dataset)
    logger.info(f"Loaded {args.dataset} dataset:")
    logger.info(f"  Nodes: {graph_data.x.shape[0]}")
    logger.info(f"  Edges: {graph_data.edge_index.shape[1]}")
    logger.info(f"  Features: {graph_data.x.shape[1]}")
    logger.info(f"  Classes: {int(graph_data.y.max().item()) + 1}")
    
    if args.compare_metrics:
        # Compare different configurations
        logger.info("\n" + "="*60)
        logger.info("Comparing Different Configurations")
        logger.info("="*60)
        
        configs = [
            ('cosine', False),
            ('cosine', True),
            ('euclidean', False),
            ('euclidean', True)
        ]
        
        for metric, use_label in configs:
            label_str = "with label prototypes" if use_label else "without label prototypes"
            logger.info(f"\nTesting {metric} distance {label_str}:")
            
            results = learner.evaluate_multiple_runs(
                graph_data,
                k_shot=args.k_shot,
                n_runs=min(5, args.n_runs),
                distance_metric=metric,
                use_label_prototypes=use_label
            )
            
            if 'error' not in results:
                print(f"  Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
                print(f"  Mean Confidence: {results['mean_confidence']:.4f}")
    
    else:
        logger.info("\n" + "="*60)
        logger.info(f"Single Run Evaluation ({args.k_shot}-shot)")
        logger.info("="*60)
        
        single_results = learner.prototype_inference(
            graph_data, 
            k_shot=args.k_shot,
            seed=args.seed,
            distance_metric=args.distance_metric,
            use_label_prototypes=args.use_label_prototypes
        )
        
        if 'error' not in single_results:
            print(f"\nSingle Run Results:")
            print(f"  Accuracy: {single_results['accuracy']:.4f}")
            print(f"  Confidence: {single_results['mean_confidence']:.4f}")
            print(f"  Test Samples: {single_results['num_test_samples']}")
            print(f"  Distance Metric: {single_results['distance_metric']}")
            
            print(f"\nPer-Class Accuracy:")
            for class_id, acc in single_results['per_class_accuracy'].items():
                print(f"  Class {class_id}: {acc:.4f}")
        
        if args.n_runs > 1:
            logger.info("\n" + "="*60)
            logger.info(f"Multiple Runs Evaluation ({args.n_runs} runs)")
            logger.info("="*60)
            
            multi_results = learner.evaluate_multiple_runs(
                graph_data,
                k_shot=args.k_shot,
                n_runs=args.n_runs,
                distance_metric=args.distance_metric,
                use_label_prototypes=args.use_label_prototypes
            )
            
            if 'error' not in multi_results:
                print(f"\nMultiple Runs Results ({args.n_runs} runs):")
                print(f"  Mean Accuracy: {multi_results['mean_accuracy']:.4f} ± {multi_results['std_accuracy']:.4f}")
                print(f"  Mean Confidence: {multi_results['mean_confidence']:.4f} ± {multi_results['std_confidence']:.4f}")
                print(f"  Distance Metric: {multi_results['distance_metric']}")
                print(f"  Use Label Prototypes: {multi_results['use_label_prototypes']}")
                
                print(f"\n  Per-Class Accuracy (mean ± std):")
                for class_id, stats in multi_results['per_class_accuracy'].items():
                    print(f"    Class {class_id}: {stats['mean']:.4f} ± {stats['std']:.4f}")
                
                # Show distribution
                print(f"\n  Accuracy Distribution:")
                for i, acc in enumerate(multi_results['all_accuracies']):
                    print(f"    Run {i+1}: {acc:.4f}")
    
    logger.info("\n" + "="*60)
    logger.info("Evaluation Complete!")
    logger.info("="*60)

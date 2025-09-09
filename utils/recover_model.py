import torch
import os
import os.path as osp
from pathlib import Path
import copy
from omegaconf import DictConfig, OmegaConf
import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from model.base import BackboneGNN
from model.fingerprint import DomainEmbedder
from model.pt_model import GFM
from data_process.data import CombineDataset
from data_process.task_constructor import train_task_constructor
from data_process.datahelper import refine_dataset, span_node_and_edge_idx, filter_unnecessary_attrs
from utils.logging import logger
import hydra


def recover_model_from_checkpoint():
    
    checkpoint_dir = "generated_files/checkpoints/G-Align/Aug11-17:00-9929422f/checkpoints"
    output_dir = "generated_files/output/G-Align/Aug11-17:00-9929422f"
    fingerprint_dir = "generated_files/fingerprint/pubmed_arxiv_wikics"
    
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_path = osp.join(checkpoint_dir, "gfm-epoch=2914-val_loss=1.1416.ckpt")

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    cfg = checkpoint['hyper_parameters']['cfg']
    L_max = checkpoint['hyper_parameters']['L_max']
    
    project_root = os.getcwd()
    if not hasattr(cfg, 'dirs'):
        cfg.dirs = DictConfig({})
    cfg.dirs.fingerprint_storage = osp.join(project_root, 'generated_files', 'fingerprint')
    cfg.dirs.data_storage = osp.join(project_root, 'datasets')
    cfg.dirs.output = output_dir
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    comb_graphs_info = checkpoint['hyper_parameters'].get('comb_pretrained_graphs')
    
    input_dim = 64 
    
    backbone_gnn = BackboneGNN(in_dim=input_dim, num_classes=L_max, cfg=cfg)
    
    frozen_backbone = BackboneGNN(in_dim=input_dim, num_classes=L_max, cfg=cfg)
    frozen_backbone.n_layers = 1
    frozen_backbone.readout_proj = False
    
    theta0_path = osp.join(fingerprint_dir, "theta0.pt")
    if osp.exists(theta0_path):
        theta0 = torch.load(theta0_path, weights_only=False)
        frozen_backbone.load_state_dict(theta0, strict=False)
        logger.info("Loaded theta0 for frozen backbone")
    
    domain_embedder = DomainEmbedder(frozen_backboneGNN=frozen_backbone, cfg=cfg)
    
    if cfg.Fingerprint.DE_type == 'conv':
        projection_state_path = osp.join(fingerprint_dir, "projection_state.pt")
        if osp.exists(projection_state_path):
            projection_state = torch.load(projection_state_path, weights_only=False)
            domain_embedder.dm_extractor.projection.load_state_dict(projection_state)
            logger.info("Loaded projection network state")
        
        delta_matrices_path = osp.join(fingerprint_dir, "delta_matrices.pt")
        d_c_max_path = osp.join(fingerprint_dir, "d_c_max.pt")
        if osp.exists(delta_matrices_path) and osp.exists(d_c_max_path):
            delta_matrices = torch.load(delta_matrices_path, weights_only=False)
            d_c_max = torch.load(d_c_max_path, weights_only=False)
            domain_embedder.dm_extractor._delta_matrices = delta_matrices
            domain_embedder.dm_extractor._d_c_max = d_c_max
            logger.info("Loaded delta matrices and d_c_max")
    
    e_path = osp.join(fingerprint_dir, "PreDomainEmbedding.pt")
    if osp.exists(e_path):
        domain_embedder.dm_extractor._e = torch.load(e_path, weights_only=False)
        logger.info("Loaded pre-computed domain embeddings")
    
    domain_embedder.dm_extractor._theta0 = theta0 if 'theta0' in locals() else None
    domain_embedder.dm_extractor._cached = True
    
    model_state_dict = checkpoint['state_dict']
    
    backbone_state = {}
    for key, value in model_state_dict.items():
        if key.startswith('GNNEnc.'):
            new_key = key.replace('GNNEnc.', '')
            backbone_state[new_key] = value
    
    backbone_gnn.load_state_dict(backbone_state)
    logger.info("Loaded trained backbone state")
    
    film_state = {}
    for key, value in model_state_dict.items():
        if 'de.dm_film.' in key:
            new_key = key.replace('de.dm_film.', '')
            film_state[new_key] = value
    
    if film_state:
        domain_embedder.dm_film.load_state_dict(film_state)
        logger.info("Loaded FiLM network state")
    
    combined_graphs_info = {
        'num_nodes': 200761,  
        'num_features': 64,
        'num_graphs': 3,  
        'ptr': torch.tensor([0, 19717, 189060, 200761]),  
        'name_dict': {'pubmed': 0, 'arxiv': 1, 'wikics': 2}
    }
    
    final_model_state = {
        'model_state_dict': model_state_dict,
        'backbone_state_dict': backbone_gnn.state_dict(),
        'frozen_backbone_state_dict': frozen_backbone.state_dict(),
        'domain_embedder_theta0': domain_embedder.dm_extractor._theta0,
        'domain_embedder_e': domain_embedder.dm_extractor._e,
        'config': cfg,
        'L_max': L_max,
        'pretrain_datasets': ['pubmed', 'arxiv', 'wikics'],
        'pretrain_tasks': ['pubmed_link', 'pubmed_node', 'arxiv', 'wikics'],
        'combined_graphs_info': combined_graphs_info
    }
    
    if cfg.Fingerprint.DE_type == 'conv':
        final_model_state['domain_embedder_projection_state'] = domain_embedder.dm_extractor.projection.state_dict()
        if hasattr(domain_embedder.dm_extractor, '_delta_matrices'):
            final_model_state['domain_embedder_delta_matrices'] = domain_embedder.dm_extractor._delta_matrices
    elif cfg.Fingerprint.DE_type == 'pca':
        # Load B matrix for PCA
        B_path = osp.join(fingerprint_dir, "B.pt")
        if osp.exists(B_path):
            final_model_state['domain_embedder_B'] = torch.load(B_path, weights_only=False)
    
    final_model_path = osp.join(output_dir, "final_gfm_model.pt")
    torch.save(final_model_state, final_model_path)
    
    logger.info(f"Successfully saved final model to: {final_model_path}")
    logger.info(f"Model file size: {os.path.getsize(final_model_path) / 1024 / 1024:.2f} MB")
    
    try:
        test_load = torch.load(final_model_path, map_location='cpu', weights_only=False)
        logger.info("Verification: Model can be loaded successfully!")
        logger.info(f"Model contains keys: {list(test_load.keys())}")
    except Exception as e:
        logger.error(f"Failed to verify model loading: {e}")
    
    return final_model_path

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("Recovering model from checkpoint...")
    logger.info("="*60)
    
    final_path = recover_model_from_checkpoint()
    
    logger.info("="*60)
    logger.info("Recovery complete!")
    logger.info(f"Final model saved at: {final_path}")
    logger.info("="*60)
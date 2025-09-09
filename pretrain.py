import torch
import torch.nn as nn
import torch.nn.functional as F
import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False) 
from omegaconf import DictConfig
from utils.logging import timer
import hydra
from utils.logging import logger
from utils.exp import init_exp
from utils.utils import seed_setting
from data_process.data import CombineDataset
import wandb
from omegaconf import OmegaConf
from data_process.task_constructor import train_task_constructor, UnifiedTaskConstructor
from data_process.datahelper import refine_dataset, span_node_and_edge_idx, filter_unnecessary_attrs
from model.base import BackboneGNN, FlexibleBackboneGNN, BackboneGNN2
from model.fingerprint import DomainEmbedder
from model.pt_model import GFM, DataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import os
import os.path as osp
import copy


def create_backbone_and_frozen_copy(input_dim, L_max, cfg, pretrain_device):
    n_layers_train = cfg.Fingerprint.n_layers  
    n_layers_fingerprint = cfg.Fingerprint.n_layers_fingerprint if hasattr(cfg.Fingerprint, 'n_layers_fingerprint') else 1
    
    logger.info(f"Creating backbone with {n_layers_train} layers for training")
    logger.info(f"Creating frozen backbone with {n_layers_fingerprint} layers for fingerprinting")
    
    if cfg.Fingerprint.get('use_flexible_backbone', True):
        backbone_gnn = FlexibleBackboneGNN(
            in_dim=input_dim, 
            num_classes=L_max, 
            cfg=cfg,
            fingerprint_layers=n_layers_fingerprint
        ).to(pretrain_device)
        
        frozen_backbone = backbone_gnn.create_fingerprint_copy().to(pretrain_device)
        
    else:
        
        backbone_gnn = BackboneGNN2(in_dim=input_dim, num_classes=L_max, cfg=cfg).to(pretrain_device)
        
        cfg_frozen = copy.deepcopy(cfg)
        cfg_frozen.Fingerprint.n_layers = n_layers_fingerprint
        
        frozen_backbone = BackboneGNN2(in_dim=input_dim, num_classes=L_max, cfg=cfg_frozen).to(pretrain_device)
        
        state_dict = backbone_gnn.get_submodel_state_dict(n_layers_fingerprint)
        frozen_backbone.load_state_dict(state_dict, strict=False)
    
    logger.info(f"Backbone GNN has {len(backbone_gnn.gnns)} conv layers")
    logger.info(f"Frozen backbone has {len(frozen_backbone.gnns)} conv layers")
    
    if len(frozen_backbone.gnns) > 0:
        with torch.no_grad():
            main_first_layer = next(backbone_gnn.gnns[0].parameters())
            frozen_first_layer = next(frozen_backbone.gnns[0].parameters())
            
            if torch.allclose(main_first_layer, frozen_first_layer):
                logger.info("First layer weights are shared between main and frozen backbone")
            else:
                logger.warning("First layer weights differ between main and frozen backbone")
    
    return backbone_gnn, frozen_backbone


def init_wandb(cfg):
    wandb.init(project=cfg.wandb.project,
               name = "Pretrain on tasks = {}".format(cfg.pretrain.pretrain_tasks),
               mode="disabled" if cfg.wandb.debug else "online",
               config=OmegaConf.to_object(cfg),)


def get_pretrain_data(cfg, pretrain_device):
    pretrain_ds_names = cfg.pretrain.pretrain_datasets
    if isinstance(pretrain_ds_names, str):
        pretrain_ds_names_lst = [a.strip() for a in pretrain_ds_names.split(",")]
    else:
        pretrain_ds_names_lst = pretrain_ds_names
    pretrain_tasks = cfg.pretrain.train_tasks  
    if isinstance(pretrain_tasks, str):
        pretrain_tasks = [a.strip() for a in pretrain_tasks.split(",")]
    pretrain_task_names = ','.join(pretrain_tasks)
    logger.info(f"Pretrain on the following tasks: {pretrain_task_names}")  
    tasks: UnifiedTaskConstructor = train_task_constructor(data_path = cfg.dirs.data_storage, cfg = cfg, pretrain_tasks = pretrain_tasks)
    pretrain_dataset_dict = {}
    data_config_lookup = cfg.data_config
    for ds_name in pretrain_ds_names_lst: 
        if ds_name not in pretrain_dataset_dict:
            data_config = data_config_lookup[ds_name]
            dataset = tasks.get_dataset(cfg, data_config)
            dataset = refine_dataset(dataset)
            dataset = span_node_and_edge_idx(dataset)
            dataset = filter_unnecessary_attrs(dataset)
            pretrain_dataset_dict[ds_name] = dataset
    
    combined_pretrained_dataset = CombineDataset(cfg=cfg, pretrain_ds_dict = pretrain_dataset_dict, pretrain_device=pretrain_device).combine_graph()
    combined_pretrained_data = combined_pretrained_dataset[0]
    return combined_pretrained_data, pretrain_tasks


@timer()
@hydra.main(config_path=f"{root}/configs", config_name="main", version_base=None)
def main(cfg:DictConfig):
    cfg = init_exp(cfg) 
    seed_setting(cfg.pretrain.seed)
    if torch.cuda.is_available() and cfg.preprocess_device == "gpu":
        pretrain_device = torch.device(f"cuda:{cfg.pretrain.gpu}")
    else:
        pretrain_device = torch.device("cpu")
    comb_graphs, pretrain_tasks = get_pretrain_data(cfg, pretrain_device)
    cfg.pretrain.pretrain_tasks = pretrain_tasks  # [pubmed_link, pubmed_node, arxiv, wikics]
    # init_wandb(cfg)

    logger.info("Starting pretraining phase...")
    L_max = int(comb_graphs.y.max().item()) + 1
    logger.info(f"Maximum label count across datasets: {L_max}")

    input_dim = comb_graphs.x.size(-1)
    if cfg.Fingerprint.n_layers == 1 and cfg.Fingerprint.readout_proj:
        backbone_gnn = BackboneGNN(in_dim=input_dim, num_classes=L_max, cfg=cfg).to(pretrain_device)

        frozen_backbone = copy.deepcopy(backbone_gnn)

        frozen_backbone.n_layers = 1
        frozen_backbone.readout_proj = False
    else:
        backbone_gnn, frozen_backbone = create_backbone_and_frozen_copy(input_dim, L_max, cfg, pretrain_device) 


    domain_embedder = DomainEmbedder(frozen_backboneGNN=frozen_backbone, cfg=cfg).to(pretrain_device)
    

    if domain_embedder.dm_extractor._cached and cfg.Fingerprint.n_layers == 1 and cfg.Fingerprint.readout_proj:
        _theta0 = domain_embedder.dm_extractor._theta0
        backbone_gnn.load_state_dict(_theta0, strict=False)
    elif domain_embedder.dm_extractor._cached and cfg.Fingerprint.n_layers != 1 and cfg.Fingerprint.readout_proj:
        _theta0 = domain_embedder.dm_extractor._theta0
        backbone_gnn.load_state_dict(_theta0, strict=False)
    

    model = GFM(cfg = cfg,
                L_max= L_max,
                comb_pretrained_graphs= comb_graphs,
                backboneGNN=backbone_gnn,
                domain_embedder=domain_embedder)
    k_shot = cfg.pretrain.k_shot
    m_way = min(cfg.pretrain.m_way, L_max)
    t_query = cfg.pretrain.t_query
    n_episodes = cfg.pretrain.n_eqisodes
    data_module = DataModule(data=comb_graphs, k=k_shot, m=m_way, n=n_episodes, t=t_query)

    checkpoint_dir = osp.join(cfg.dirs.checkpoint_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.dirs.output, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir,
                                          filename = "gfm-{epoch:02d}-{val_loss:.4f}",
                                          monitor="val_loss",
                                          mode="min",
                                          save_top_k=3,
                                          save_last=True,
                                          verbose=True)
    
    early_stopping_callback = EarlyStopping(monitor="val_loss",        
                                            min_delta=cfg.pretrain.min_delta,
                                            patience=cfg.pretrain.patience,
                                            verbose=True,
                                            mode="min")
    wandb_logger = WandbLogger(project=cfg.wandb.project,
                               name = "Pretrain on tasks = {}, n_layers = {}".format(cfg.pretrain.pretrain_tasks, cfg.Fingerprint.n_layers),
                               save_dir=cfg.dirs.temp,
                               mode="disabled" if cfg.wandb.debug else "online",
                               config=OmegaConf.to_object(cfg)) if not cfg.wandb.debug else None
    
    trainer = pl.Trainer(max_epochs=cfg.pretrain.pretrain_epochs,
                         accelerator="gpu" if torch.cuda.is_available() and cfg.pretrain.gpu is not None else "cpu",
                         devices=[cfg.pretrain.gpu] if torch.cuda.is_available() and cfg.pretrain.gpu is not None else "auto",
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback, early_stopping_callback],
                         log_every_n_steps=cfg.pretrain.log_every_n_steps,
                         check_val_every_n_epoch=cfg.pretrain.check_val_every_n_epoch,
                         enable_progress_bar= True,
                         deterministic=True)
    
    logger.info("Starting training...")

    trainer.fit(model, datamodule=data_module)

    final_model_path = osp.join(cfg.dirs.output, "final_gfm_model.pt")

    if not domain_embedder.dm_extractor._cached:
        logger.info("Computing domain embeddings...")
        with torch.no_grad():
            comb_graphs = comb_graphs.to(pretrain_device)
            if cfg.Fingerprint.DE_type == 'pca':
                e, B = domain_embedder.dm_extractor.compute_fingerprints_pca(comb_graphs)
            elif cfg.Fingerprint.DE_type == 'conv':
                e, _ = domain_embedder.dm_extractor.compute_fingerprints_conv(comb_graphs)

    model_state = {
        'model_state_dict': model.state_dict(),
        'backbone_state_dict': backbone_gnn.state_dict(),  # Trained backbone
        'frozen_backbone_state_dict': frozen_backbone.state_dict(),  # Frozen initial backbone θ₀
        'domain_embedder_B': domain_embedder.dm_extractor._B,
        'domain_embedder_theta0': domain_embedder.dm_extractor._theta0,
        'domain_embedder_e': domain_embedder.dm_extractor._e,
        'config': cfg,
        'L_max': L_max,
        'pretrain_datasets': cfg.pretrain.pretrain_datasets,
        'pretrain_tasks': pretrain_tasks,
        'combined_graphs_info': {
            'num_nodes': comb_graphs.x.shape[0],
            'num_features': comb_graphs.x.shape[1],
            'num_graphs': int(comb_graphs.ptr.numel() - 1),
            'ptr': comb_graphs.ptr,
            'name_dict': comb_graphs.name_dict
        }        
    }
    if cfg.Fingerprint.DE_type == 'conv':
        model_state['domain_embedder_projection_state'] = domain_embedder.dm_extractor.projection.state_dict()
        model_state['domain_embedder_delta_matrices'] = domain_embedder.dm_extractor._delta_matrices


    torch.save(model_state, final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")
    logger.info("Running final validation...")
    test_results = trainer.test(model, datamodule=data_module)    

    logger.info("Pretraining completed successfully!")
    logger.info(f"Final test results: {test_results}")
    
    wandb.finish()
    return model, final_model_path


if __name__ == "__main__":
    main()   






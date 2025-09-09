from omegaconf import OmegaConf
from datetime import datetime
from uuid import uuid4
import os
from utils.config import save_config

def generate_unique_id(cfg):
    cur_time = datetime.now().strftime("%b%-d-%-H:%M-")
    given_uid = cfg.get("uid")
    uid = given_uid if given_uid else cur_time + str(uuid4()).split("-")[0]
    return uid

def init_exp(cfg):
    OmegaConf.set_struct(cfg, False)  # allow cfg to be modified
    cfg.uid = generate_unique_id(cfg)
    for directory in cfg.dirs.values():
        os.makedirs(directory, exist_ok=True)
    cfg_out_file = cfg.dirs.output + "hydra_cfg.yaml"
    save_config(cfg, cfg_out_file, as_global=True)
    return cfg
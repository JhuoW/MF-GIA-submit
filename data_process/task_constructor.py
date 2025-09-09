import copy
import data_process.datahelper as datahelper
from data_process.data import SingleGraphDataset
import os.path as osp
from data_process.datahelper import SentenceEncoder

name2dataset = {"arxiv": SingleGraphDataset, "cora": SingleGraphDataset, "pubmed": SingleGraphDataset,
                 "wikics": SingleGraphDataset, "amazon-ratings": SingleGraphDataset}


class UnifiedTaskConstructor:
    def __init__(self, tasks: list[str], encoder: datahelper.SentenceEncoder, task_config_lookup: dict,
                 data_config_lookup: dict, root="cache_data", batch_size=256, sample_size=-1):
        self.root = root
        self.tasks = tasks
        self.encoder = encoder
        self.task_config_lookup = task_config_lookup
        self.data_config_lookup = data_config_lookup
        self.batch_size = batch_size
        self.sample_size = sample_size
        # with open("data/low_resource_split.json", "r") as f:
        #     self.lr_class_split = json.load(f)

        self.dataset = {}  # keyed by base dataset names e.g. cora, pubmed and not cora-link
        self.dataset_split = {}  # keyed by dataset names and task level e.g. cora_e2e_link
        self.preprocess_storage = {}  # keyed by dataset names and task level e.g. cora_e2e_link
        self.datamanager = {}
        self.edges = {}
        self.datasets = {"train": [], "valid": [],
                         "test": []}  # train a list of Dataset, valid/test a list of DataWithMeta
        self.stage_names = {"train": [], "valid": [], "test": []}

    def construct_exp(self):
        val_task_index_lst = []
        val_pool_mode = []
        for task in self.tasks:
            config = self.task_config_lookup[task]
            config = copy.deepcopy(config)
            val_task_index_lst.append(self.construct_task(config))
            val_pool_mode.append(config["eval_pool_mode"])
        return val_task_index_lst, val_pool_mode

    def construct_task(self, config):
        """
        Datasets in a task are described in config["eval_set_constructs"] that describe the stage (train/valid/test)
        of the dataset.
        """
        val_task_index = []
        for stage_config in config["eval_set_constructs"]:
            if "dataset" not in stage_config:
                stage_config["dataset"] = config["dataset"]
            dataset_name = stage_config["dataset"]

            assert dataset_name in self.data_config_lookup

            dataset_config = self.data_config_lookup[dataset_name]

            stage_ind = self.add_dataset(stage_config, dataset_config)

            if stage_config["stage"] == "valid":
                val_task_index.append(stage_ind)
        return val_task_index


    def get_dataset(self, cfg, dataset_config):
        dataset_name = dataset_config["dataset_name"]
        if dataset_name not in self.dataset:
            self.dataset[dataset_name] = name2dataset[dataset_name](cfg, 
                                                                    name = dataset_name, 
                                                                    llm_encoder = self.encoder, 
                                                                    load_text = False if dataset_name in ["cora", "pubmed", "amazon-ratings"] else True)
        return self.dataset[dataset_name]

def train_task_constructor(data_path, cfg, pretrain_tasks):
    llm_encoder = SentenceEncoder(cfg.llm_name, batch_size=cfg.llm_b_size)
    if isinstance(pretrain_tasks, str):
        task_names = [a.strip() for a in pretrain_tasks.split(",")]
    else:
        task_names = pretrain_tasks
    
    root = data_path
    if cfg.llm_name != "ST":
        root = f"{data_path}_{cfg.llm_name}"
    
    tasks = UnifiedTaskConstructor(task_names, 
                                   llm_encoder, 
                                   task_config_lookup = None,
                                   data_config_lookup = None,
                                   root=root,
                                   batch_size=cfg.batch_size,
                                   sample_size=cfg.train_sample_size)

    return tasks

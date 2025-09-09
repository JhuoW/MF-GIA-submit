from hydra.utils import instantiate
from torch_geometric.loader import DataLoader as PyGDataLoader
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import torch
from itertools import accumulate
from torch_geometric.data import Data, Dataset
from data_process.datahelper import SentenceEncoder, read_knowledge_graph
from torch_geometric.data import InMemoryDataset
import os.path as osp
from utils.utils import pth_safe_load
from torch_geometric.datasets import WikiCS
from ogb.nodeproppred import PygNodePropPredDataset
from abc import ABC
import json
import functools
from torch_geometric.transforms import ToUndirected

class SingleGraphDataset(InMemoryDataset, ABC):
    def __init__(self, cfg, 
                       name, 
                       transform = None,   
                       pre_transform = None, 
                       llm_encoder = SentenceEncoder,  
                       load_text = False): 
        self.cfg = cfg
        self.ds_name = name
        self.unify_dim = cfg.unify_dim if cfg.unify_dim else 50
        self.data_source, ds_alias = cfg['_ds_meta_data'][self.ds_name].split(", ")
        components = ds_alias.split(".") 

        if len(components) == 2:
            base_source, name = components
        else:
            raise ValueError(f"Unexpected ds_alias format: {ds_alias}")

        root = osp.join(cfg.dirs.data_storage, self.data_source, ds_alias)
        self.llm_encoder = llm_encoder
        self.load_text = load_text
        super().__init__(root=root, transform=transform, pre_transform=pre_transform)

        if load_text: 
            self.texts = torch.load(self.processed_paths[1], weights_only=False)  # load text features

        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        if load_text:
            self.side_data = pth_safe_load(self.processed_paths[2]) if osp.exists(self.processed_paths[2]) else None

        self.num_nodes = self.data.num_nodes
        self.num_feats = self.data.num_features
        self.edge_index = self.data.edge_index
        self.labels = self.data.y
        self.train_mask = self.data.train_mask if hasattr(self.data, 'train_mask') else None
        self.val_mask = self.data.val_mask if hasattr(self.data, 'val_mask') else None
        self.test_mask = self.data.test_mask if hasattr(self.data, 'test_mask') else None
        if not load_text:
            self.pca_feat = self.data.pca_x

    def data2vec(self, data:list[str]):
        if self.llm_encoder is None:
            raise NotImplementedError("LLM encoder is not defined")
        if data is None:
            return None
        embeddings = self.llm_encoder.encode(data).cpu()
        return embeddings        


    @property
    def raw_file_names(self):
        return []
    
    def download(self):
        pass

    @property
    def processed_file_names(self):
        if self.load_text:
            return ["geometric_data_processed.pt", "texts.pkl", "data.pt"]
        return ['data.pt']

    def text2feature(self, texts):
        if isinstance(texts[0], str):
            return self.data2vec(texts)
        return [self.text2feature(text) for text in texts]

    def add_text_emb(self, data_list, text_emb):
        data_list[0].node_text_feat = text_emb[0] 
        data_list[0].edge_text_feat = text_emb[1] 
        data_list[0].class_node_text_feat = text_emb[3]  
        return self.collate(data_list)

    @staticmethod
    def _build_adj(edge_index, num_nodes):
        row, col = edge_index
        data = torch.ones(row.size(0))
        import scipy.sparse as sp
        coo = sp.coo_matrix((np.ones(row.size(0)), (row.cpu().numpy(), col.cpu().numpy())),
                            shape=(num_nodes, num_nodes))
        return sp.csr_matrix(coo)

    def gen_data(self):
        ds_alias = self.cfg['_ds_meta_data'][self.ds_name].split(", ")[1] 
        components = ds_alias.split(".")  
        parent, child = components  
        if self.ds_name == 'wikics':

            dataset =  WikiCS(root = self.root)
            with open(osp.join(self.root, "metadata.json")) as json_file:
                raw_data = json.load(json_file)
            node_info = raw_data['nodes']
            label_info = raw_data['labels']
            node_text_lst = []
            label_text_lst = []
            for node in node_info:
                node_feature = ((
                        "feature node. wikipedia entry name: " + node["title"] + ". entry content: " + functools.reduce(
                    lambda x, y: x + " " + y, node["tokens"])).lower().strip())
                node_text_lst.append(node_feature)
            for label in label_info.values():
                label_feature = (("prompt node. wikipedia entry category: " + label).lower().strip())
                label_text_lst.append(label_feature)
            edge_text = ["feature edge. wikipedia page link"]
            prompt_text = ["prompt node. node classification of wikipedia entry category"]
            prompt_edge_text = ["prompt edge."]
            prompt_text_map = {"e2e_node": {"noi_node_text_feat": ["noi_node_text_feat", [0]],
                                            "class_node_text_feat": ["class_node_text_feat", torch.arange(len(label_text_lst))],
                                            "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]}}
            return ([dataset.data], 
                    [
                        node_text_lst, 
                        edge_text, 
                        prompt_text, 
                        label_text_lst, 
                        prompt_edge_text,
                    ], 
                    prompt_text_map,)
        elif self.ds_name == 'arxiv':
            dataset = PygNodePropPredDataset(name = "ogbn-arxiv", root = self.root, transform = ToUndirected())
            split = dataset.get_idx_split()
            def from_split_to_mask(split, length):
                mask = torch.zeros(length, dtype = torch.bool)
                mask[split] = True
                return mask
            dataset.data.train_mask = from_split_to_mask(split['train'], dataset.data.num_nodes)
            dataset.data.val_mask = from_split_to_mask(split['valid'], dataset.data.num_nodes)
            dataset.data.test_mask = from_split_to_mask(split['test'], dataset.data.num_nodes)

            nodeidx2paperid = pd.read_csv(osp.join(self.root, "nodeidx2paperid.csv.gz"), index_col = "node idx")
            titleabs_url = "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv"
            titleabs = pd.read_csv(titleabs_url, sep = "\t", names = ["paper id", "title", "abstract"], index_col = "paper id")

            titleabs = nodeidx2paperid.join(titleabs, on = "paper id")
            text = (
                    "feature node. paper title and abstract: "
                    + titleabs["title"]
                    + ". "
                    + titleabs["abstract"]
                )
            
            node_text_list = text.values
            feat_node_texts = node_text_list.tolist()

            categories_desc = open(osp.join(self.root, "arxiv_CS_categories.txt"), "r").readlines()

            state = 0
            result = {"id": [], "name": [], "description": []}

            for line in categories_desc:
                if state == 0:
                    assert line.strip().startswith("cs.")
                    category = (
                        "arxiv " + " ".join(line.strip().split(" ")[0].split(".")).lower()
                    )  # e. g. cs lo
                    name = line.strip()[7:-1]  # e. g. Logic in CS
                    result["id"].append(category)
                    result["name"].append(name)
                    state = 1
                    continue
                elif state == 1:
                    description = line.strip()
                    result["description"].append(description)
                    state = 2
                    continue
                elif state == 2:
                    state = 0
                    continue
            arxiv_cs_taxonomy = pd.DataFrame(result)
            mapping_file = osp.join(self.root, "labelidx2arxivcategeory.csv.gz")
            arxiv_categ_vals = pd.merge(
                pd.read_csv(mapping_file),
                arxiv_cs_taxonomy,
                left_on="arxiv category",
                right_on="id",
            )
            text = (
                "prompt node. literature category and description: "
                + arxiv_categ_vals["name"]
                + ". "
                + arxiv_categ_vals["description"]
            )
            label_text_lst = text.values
            class_node_texts = label_text_lst.tolist()
            or_labeled_text = []
            not_and_labeled_text = []
            for i in range(len(arxiv_categ_vals)):
                for j in range(len(arxiv_categ_vals)):
                    c1 = arxiv_categ_vals.iloc[i]
                    c2 = arxiv_categ_vals.iloc[j]
                    txt = (
                            "prompt node. literature category and description: not "
                            + c1["name"]
                            + ". "
                            + c1["description"]
                            + " and not "
                            + c2["name"]
                            + ". "
                            + c2["description"]
                        )
                    not_and_labeled_text.append(txt)
                    txt = (
                            "prompt node. literature category and description: either "
                            + c1["name"]
                            + ". "
                            + c1["description"]
                            + " or "
                            + c2["name"]
                            + ". "
                            + c2["description"]
                        )
                    or_labeled_text.append(txt)
                                
            logic_node_texts = or_labeled_text + not_and_labeled_text
            feat_edge_texts = ["feature edge. citation"]
            noi_node_texts = ["prompt node. node classification of literature category"]
            prompt_edge_texts = [
                                    "prompt edge.",
                                    "prompt edge. edge for query graph that is our target",
                                    "prompt edge. edge for support graph that is an example",
                                ]
            prompt_text_map = {
                "e2e_node": {
                    "noi_node_text_feat": ["noi_node_text_feat", [0]],
                    "class_node_text_feat": [
                        "class_node_text_feat",
                        torch.arange(len(class_node_texts)),
                    ],
                    "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]],
                },
                "lr_node": {
                    "noi_node_text_feat": ["noi_node_text_feat", [0]],
                    "class_node_text_feat": [
                        "class_node_text_feat",
                        torch.arange(len(class_node_texts)),
                    ],
                    "prompt_edge_text_feat": ["prompt_edge_text_feat", [0, 1, 2]],
                },
                "logic_e2e": {
                    "noi_node_text_feat": ["noi_node_text_feat", [0]],
                    "class_node_text_feat": [
                        "class_node_text_feat",
                        torch.arange(
                            len(class_node_texts), len(class_node_texts) + len(logic_node_texts)
                        ),
                    ],
                    "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]],
                },
            }
            return (
                [dataset[0]],
                [
                    feat_node_texts,
                    feat_edge_texts,
                    noi_node_texts,
                    class_node_texts,
                    # class_node_texts + logic_node_texts,
                    prompt_edge_texts,
                ],
                prompt_text_map,
            )
            


    def process(self):
        if not self.load_text:  # for cora, pubmed
            if self.data_source not in ['ogb.nodeproppred']:
                ds_alias = self.cfg['_ds_meta_data'][self.ds_name].split(", ")[1]  # Planetoid.Cora   PygNodePropPredDataset.ogbn-products
                components = ds_alias.split(".")
                # save data into datasets/pyg/Planetoid.Cora/...
                if len(components) == 2:
                    parent, child = components   # parent = Planetoid, child = Cora 
                    # torch_geometric.datasets.planetoid.Planetoid
                    if child not in ['FacebookPagePage',  'EmailEUCore', 'Reddit', 'DeezerEurope', 'LastFMAsia']:
                        dataset = getattr(__import__("torch_geometric.datasets", fromlist=[parent]), parent)(
                            root = self.root, name = child
                        )
                    else: # parent = Social, child=FacebookPagePage
                        from torch_geometric.datasets import LastFMAsia, FacebookPagePage, EmailEUCore, Reddit, DeezerEurope
                        dataset = FacebookPagePage(root=self.root)
                else:
                    dataset = instantiate({"_target_": f"torch_geometric.datasets.{ds_alias}", "root": self.root})
                data = dataset[0]
                num_nodes = data.num_nodes
                pca_cache_path = osp.join(self.processed_dir, f"pca_{self.unify_dim}.pt")
                if osp.exists(pca_cache_path):
                    pca_x = torch.load(pca_cache_path, weights_only=False)  # already a float tensor
                else:
                    if dataset.num_features == self.unify_dim:
                        pca_x = data.x.clone()
                    else:
                        if self.ds_name not in ['flickr','twitter']:
                            x_np = data.x.cpu().numpy()
                        else:
                            x_np = data.x.cpu().to_dense().numpy()
                        pca = PCA(n_components=self.unify_dim)
                        projected = pca.fit_transform(x_np)  # (N, unify_dim)
                        pca_x = torch.from_numpy(projected).float()
                    torch.save(pca_x, pca_cache_path)
                
                data.pca_x = pca_x
                data.num_classes = dataset.num_classes  # store for convenience
                data.num_features = dataset.num_features
                data.y = data.y
                data.edge_index = data.edge_index
                data_list = [data]
                data, slices = self.collate(data_list)
                torch.save((data, slices), self.processed_paths[0])
            else:
                ds_alias = self.cfg['_ds_meta_data'][self.ds_name].split(", ")[1]  # PygNodePropPredDataset.ogbn-products
                components = ds_alias.split(".")  
                parent, child = components  # # PygNodePropPredDataset, ogbn-products

                dataset = PygNodePropPredDataset(name = child, root = self.root, transform = ToUndirected())
                data = dataset[0]
                num_nodes = data.num_nodes
                pca_cache_path = osp.join(self.processed_dir, f"pca_{self.unify_dim}.pt")
                if osp.exists(pca_cache_path):
                    pca_x = torch.load(pca_cache_path, weights_only=False)  # already a float tensor
                else:
                    if dataset.num_features == self.unify_dim:
                        pca_x = data.x.clone()
                    else:
                        x_np = data.x.cpu().numpy()
                        pca = PCA(n_components=self.unify_dim)
                        projected = pca.fit_transform(x_np)  # (N, unify_dim)
                        pca_x = torch.from_numpy(projected).float()
                    torch.save(pca_x, pca_cache_path)
                
                data.pca_x = pca_x
                data.num_classes = dataset.num_classes  # store for convenience
                data.num_features = dataset.num_features
                data.y = data.y
                data.edge_index = data.edge_index
                data_list = [data]
                data, slices = self.collate(data_list)
                torch.save((data, slices), self.processed_paths[0])
        else:  # for arxiv, wikics
            if self.llm_encoder.model is None:
                self.llm_encoder.get_model()
            data_list, texts, side_data = self.gen_data()
            texts_emb = self.text2feature(texts)  
            torch.save(texts, self.processed_paths[1]) 
            if side_data is not None:
                torch.save(side_data, self.processed_paths[2])
            else:
                torch.save("No side datat", self.processed_paths[2])

            data, slices = self.add_text_emb(data_list, texts_emb)
            print("Saving...")
            torch.save((data, slices), self.processed_paths[0])

    def dataloader(self, batch_size=1, shuffle=False, **kwargs):
        return PyGDataLoader(self, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def get_task_map(self):
        return self.side_data

    def get_edge_list(self, mode="e2e"):
        if mode == "e2e_node":
            return {"f2n": [1, 0], "n2f": [3, 0], "n2c": [2, 0], "c2n": [4, 0]}
        elif mode == "lr_node":
            return {"f2n": [1, 0]}
        elif mode == "e2e_link":
            return {"f2n": [1, 0], "n2f": [3, 0], "n2c": [2, 0], "c2n": [4, 0]}

    def get_prompt_text_feat(self, task_name):

        task_map = self.get_task_map()
        if task_name not in task_map:
            raise NotImplementedError(
                "Task " + task_name + " is not implemented for " + self.name + " dataset the implemented tasks are "
                + str(
                    task_map.keys()))
        feat_ind = task_map[task_name]
        prompt_feats = {}
        for k in feat_ind:
            prompt_feats[k] = getattr(self.data, feat_ind[k][0])[feat_ind[k][1]]
        return prompt_feats


class KGDataset(InMemoryDataset, ABC):
    def __init__(self, cfg, 
                       name,  # [fb15k237, wn18rr]
                       transform = None,   
                       pre_transform = None, 
                       llm_encoder = SentenceEncoder, 
                       load_text = False):
        self.cfg = cfg
        self.ds_name = name  
        self.unify_dim = cfg.unify_dim if cfg.unify_dim else 50
        self.data_source, ds_alias = cfg['_ds_meta_data'][self.ds_name].split(", ")        
        components = ds_alias.split(".") 
        base_source, name = components
        self.ds_alias = ds_alias # FB15K237, WN18RR
        # datasets/ofa/KnowledgeGraph.FB15K237
        root = osp.join(cfg.dirs.data_storage, self.data_source, ds_alias)
        self.llm_encoder = llm_encoder
        self.load_text = load_text
        super().__init__(root=root, transform=transform, pre_transform=pre_transform)
        if load_text: # for arxiv, wikics
            self.texts = torch.load(self.processed_paths[1], weights_only=False)  # load text features

        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        if load_text:
            self.side_data = pth_safe_load(self.processed_paths[2]) if osp.exists(self.processed_paths[2]) else None

        self.num_nodes = self.data.num_nodes
        self.num_feats = self.data.num_features
        # self.num_classes = int(self.data.y.max().item() + 1) if self.data.y is not None else None
        # self.adj = self._build_adj(self.data.edge_index, self.num_nodes)
        self.edge_index = self.data.edge_index
        self.labels = self.data.y
        self.train_mask = self.data.train_mask if hasattr(self.data, 'train_mask') else None
        self.val_mask = self.data.val_mask if hasattr(self.data, 'val_mask') else None
        self.test_mask = self.data.test_mask if hasattr(self.data, 'test_mask') else None
        if not load_text:
            self.pca_feat = self.data.pca_x        

    def data2vec(self, data:list[str]):
        if self.llm_encoder is None:
            raise NotImplementedError("LLM encoder is not defined")
        if data is None:
            return None
        embeddings = self.llm_encoder.encode(data).cpu()
        return embeddings        

    @property
    def raw_file_names(self):
        # We delegate to Planetoid, so we can leave this empty to force download in process()
        return []
    
    def download(self):
        # No-op because underlying dataset class will handle its own download in process()
        pass

    @property
    def processed_file_names(self):
        if self.load_text:
            return ["geometric_data_processed.pt", "texts.pkl", "data.pt"]
        return ['data.pt']

    def text2feature(self, texts):
        if isinstance(texts[0], str):
            return self.data2vec(texts)
        return [self.text2feature(text) for text in texts]

    def add_text_emb(self, data_list, text_emb):
        data_list[0].node_text_feat = text_emb[0]
        data_list[0].edge_text_feat = text_emb[1]
        data_list[0].class_node_text_feat = text_emb[2]
        return self.collate(data_list)

    def get_idx_split(self):
        return self.side_data[0]

    def get_task_map(self):
        return self.side_data[-1]
    
    def get_edge_list(self, mode="e2e"):
        if mode == "e2e_link":
            return {"f2n": [1, 0], "n2f": [3, 0], "n2c": [2, 0], "c2n": [4, 0]}
        elif mode == "lr_link":
            return {"f2n": [1, 0], "n2f": [3, 0]}
    
    def gen_data(self):
        names = ["train", "valid", "test"]
        name_dict = {n: osp.join(self.root, n + ".txt") for n in names}
        # self.root = datasets/ofa/KnowledgeGraph.FB15K237
        return read_knowledge_graph(self.root, name_dict, self.ds_alias)
    
    def get_prompt_text_feat(self, task_name):
        task_map = self.get_task_map()
        if task_name not in task_map:
            raise NotImplementedError(
                "Task " + task_name + " is not implemented for " + self.name + " dataset the implemented tasks are "
                + str(
                    task_map.keys()))
        feat_ind = task_map[task_name]
        prompt_feats = {}
        for k in feat_ind:
            prompt_feats[k] = getattr(self.data, feat_ind[k][0])[feat_ind[k][1]]
        return prompt_feats
    
    def process(self):
        if self.llm_encoder.model is None:
            self.llm_encoder.get_model()
        data_list, texts, side_data = self.gen_data()
        texts_emb = self.text2feature(texts)  # 所有节点的文本用大模型转为embedding 特征
        torch.save(texts, self.processed_paths[1]) 
        if side_data is not None:
            torch.save(side_data, self.processed_paths[2])
        else:
            torch.save("No side data", self.processed_paths[2])

        data, slices = self.add_text_emb(data_list, texts_emb)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])


class CombineDataset:
    def __init__(self, cfg, pretrain_ds_dict, pretrain_device):
        self.cfg = cfg
        self.pretrain_ds_dict = pretrain_ds_dict
        self.pretrain_ds_list = list(self.pretrain_ds_dict.values())
        self.pretrain_ds_names = list(self.pretrain_ds_dict.keys())

        self.ds_name_dict = {name: i for i, name in enumerate(self.pretrain_ds_names)}  # dataset name to id

        self.to(pretrain_device)
    
    def to(self, device):
        for ds in self.pretrain_ds_list:
            ds.to(device)

    def dimension_align(self, ds):
        unify_dim = self.cfg.unify_dim if self.cfg.unify_dim else 50
        pca_cache_path = osp.join(ds.processed_dir, f"pca_{unify_dim}.pt")
        if osp.exists(pca_cache_path):
            pca_x = torch.load(pca_cache_path, weights_only=False)
        else:
            if ds.num_features == unify_dim:
                pca_x = ds.data.xn.clone()
            else:
                x_np = ds.data.xn.cpu().numpy()
                pca = PCA(n_components=unify_dim)
                projected = pca.fit_transform(x_np)
                pca_x  = torch.from_numpy(projected).float()
                torch.save(pca_x, pca_cache_path)
        return pca_x

    def combine_graph(self):
        node_counts = [ds.num_nodes for ds in self.pretrain_ds_list]

        node_offsets = [0] + list(accumulate(node_counts))[:-1]  # 每个图的开始节点 [0, 3, 8]

        label_counts = [ds.num_classes for ds in self.pretrain_ds_list]
        label_offsets = [0] + list(accumulate(label_counts))[:-1]  

        pca_x = torch.cat([self.dimension_align(ds) for ds in self.pretrain_ds_list], dim=0) 

        xe    = torch.cat([ds.data.xe for ds in self.pretrain_ds_list], dim=0).unsqueeze(1) if hasattr(self.pretrain_ds_list[0].data, 'xe') else None
        y = torch.cat([ds.labels + lo for ds, lo in zip(self.pretrain_ds_list, label_offsets)], dim=0)

        # shift and concatenate edge_index
        edge_indices = []
        for ds, no in zip(self.pretrain_ds_list, node_offsets):
            edge_indices.append(ds.edge_index + no)
        edge_index = torch.cat(edge_indices, dim=1)

        if self.cfg.pretrain.use_original_mask:
            def cat_mask(attr):
                masks = [getattr(ds, attr, None) for ds in self.pretrain_ds_list]
                masks = [m if (m is not None) else torch.zeros(ds.num_nodes, dtype=torch.bool) for ds, m in zip(self.pretrain_ds_list, masks) if m is not None]
                return torch.cat(masks, dim = 0)

            train_mask = cat_mask("train_mask")
            val_mask = cat_mask("val_mask")
            test_mask = cat_mask("test_mask")

        batch = torch.cat([torch.full((cnt,), i, dtype=torch.long) for i, cnt in enumerate(node_counts)], dim=0)  # graph id

        # ptr = [0, n0, n0+n1, n0+n1+n2, ..., total_nodes]
        ptr = torch.tensor(node_offsets + [sum(node_counts)], dtype=torch.long)
    
        # build the combined dataset
        data = Data(x = pca_x,
                    edge_index = edge_index,
                    y = y, 
                    xe = xe, 
                    train_mask = train_mask if self.cfg.pretrain.use_original_mask else None,
                    val_mask = val_mask if self.cfg.pretrain.use_original_mask else None,
                    test_mask = test_mask if self.cfg.pretrain.use_original_mask else None,
                    batch = batch,
                    ptr = ptr,
                    name_dict = self.ds_name_dict)
        
        return [data]



class EpisodeDataset(Dataset):
    def __init__(self, data: Data, k: int, m: int, n: int, t: int = 1):  # m-way, k-shot, t-query
        super().__init__()
        self.data, self.k, self.m, self.n, self.t = data, k, m, n, t
        self.M = int(data.ptr.numel() - 1) 
        self.batch = data.batch
        self.y = data.y
    
    def __len__(self):
        return self.M * self.n  

    def _sample_episode(self, i): 
        node_idx = (self.batch == i).nonzero(as_tuple = False).view(-1)  
        labels = self.y[node_idx]  
        classes = labels.unique()

        m = min(self.m, classes.numel())
        picked = classes[torch.randperm(classes.numel())[:m]] 
        support, query = [],[]
        for c in picked:
            idx_c = node_idx[labels == c]
            idx_c = idx_c[torch.randperm(idx_c.size(0))] 
            k = min(self.k, idx_c.size(0)//2) 
            support.append(idx_c[:k])
            query.append(idx_c[k:k+self.t])  
        return {
            'graph': i,
            'classes': picked,   # (m, )
            'support': torch.cat(support),  # (m*k, )
            'query': torch.cat(query)     # (m*t,)
        }
    def __getitem__(self, idx):  
        graph_id = idx // self.n   
        return self._sample_episode(graph_id)

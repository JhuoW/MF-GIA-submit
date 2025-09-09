# adapted from github.com/Zehong-Wang/GFT

import torch
from sentence_transformers import SentenceTransformer
from transformers import (LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel)
from tqdm.autonotebook import trange
import gc
import os.path as osp
import torch_geometric as pyg
import pandas as pd
import json

class SentenceEncoder:
    def __init__(self, name, root="cache_data/pretrained_llm_model", batch_size=1, multi_gpu=False, gpu_id = 0):
        self.name = name
        self.root = root
        self.device, self.gpu_ids = get_available_devices(gpu_id=gpu_id)
        self.batch_size = batch_size
        self.multi_gpu = multi_gpu
        self.model = None
        self.tokenizer = None
    
    def get_model(self):
        if self.name == "ST": # sentence-transformers
            print("Loading SentenceTransformer model ulti-qa-distilbert-cos-v1...")
            self.model = SentenceTransformer("multi-qa-distilbert-cos-v1", device=self.device, cache_folder=self.root)
            self.encode = self.ST_encode
        elif self.name == 'llama2_7b':
            model_name = "meta-llama/Llama-2-7b-hf"
            print(f"Loading Llama model {model_name}...")
            self.model = LlamaForCausalLM.from_pretrained(model_name, cache_dir=self.root).to(self.device)
            tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=self.root)
            tokenizer.padding_side = "right"
            tokenizer.truncation_side = 'right'
            self.tokenizer = tokenizer
            self.encode = self.llama_encode
                
        elif self.name == "llama2_13b":
            model_name = "meta-llama/Llama-2-13b-hf"
            print(f"Loading Llama model {model_name}...")
            model = LlamaForCausalLM.from_pretrained(model_name, cache_dir=self.root)
            self.model = model.to(self.device)
            tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=self.root)
            tokenizer.padding_side = "right"
            tokenizer.truncation_side = 'right'
            self.tokenizer = tokenizer
            self.encode = self.llama_encode           
        elif self.name == "e5":
            model_name = "intfloat/e5-large-v2"
            print(f"Loading E5 model {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.root)
            model = AutoModel.from_pretrained(model_name, cache_dir=self.root)
            self.model = model.to(self.device)
            self.tokenizer = tokenizer
            self.encode = self.e5_encode

        elif self.name == "roberta":
            print("Loading SentenceTransformer model roberta-base-nli-stsb-mean-tokens...")
            self.model = SentenceTransformer("sentence-transformers/roberta-base-nli-stsb-mean-tokens",
                device=self.device, cache_folder=self.root, )
            self.encode = self.ST_encode
        else:
            raise ValueError(f"Unknown language model: {self.name}.")

    def encode(self, texts, to_tensor=True):
        raise NotImplementedError("Not define llm encoder yet.")

    def ST_encode(self, sentences, to_tensor = True):
        if self.multi_gpu:
            pool = self.model.start_multi_process_pool()
            embeddings = self.model.encode_multi_process(sentences, pool=pool, batch_size=self.batch_size, )
            embeddings = torch.from_numpy(embeddings)
        else:
            embeddings = self.model.encode(sentences, 
                                           batch_size=self.batch_size, 
                                           show_progress_bar=True, 
                                           convert_to_tensor=to_tensor, 
                                           convert_to_numpy= not to_tensor)
        return embeddings
    
    def llama_encode(self, texts, to_tensor=True):

        # Add EOS token for padding
        self.tokenizer.pad_token = self.tokenizer.eos_token
        all_embeddings = []
        with torch.no_grad():
            for start_index in trange(0, len(texts), self.batch_size, desc="Batches", disable=False, ):
                sentences_batch = texts[start_index: start_index + self.batch_size]
                input_ids = self.tokenizer(sentences_batch, return_tensors="pt", padding="longest", truncation=True,
                                           max_length=500).input_ids.to(self.device)
                transformer_output = self.model(input_ids, return_dict=True, output_hidden_states=True)["hidden_states"]
                # No gradients on word_embeddings
                word_embeddings = transformer_output[-1].detach()
                sentence_embeddings = word_embeddings.mean(dim=1)
                all_embeddings.append(sentence_embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        if not to_tensor:
            all_embeddings = all_embeddings.numpy()

        return all_embeddings

    def e5_encode(self, texts, to_tensor=True):
        def average_pool(last_hidden_states, attention_mask):
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        all_embeddings = []
        with torch.no_grad():
            for start_index in trange(0, len(texts), self.batch_size, desc="Batches", disable=False, ):
                sentences_batch = texts[start_index: start_index + self.batch_size]
                batch_dict = self.tokenizer(sentences_batch, padding="longest", truncation=True, return_tensors='pt')
                for item, value in batch_dict.items():
                    batch_dict[item] = value.to(self.device)
                outputs = self.model(**batch_dict)
                embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                embeddings = embeddings.detach()
                all_embeddings.append(embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        if not to_tensor:
            all_embeddings = all_embeddings.numpy()

        return all_embeddings

    def flush_model(self):
        if self.model is not None:
            self.model = None
        if self.tokenizer is not None:
            self.tokenizer = None

        gc.collect()
        torch.cuda.empty_cache()

def get_available_devices(gpu_id=0):
    r"""Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')  # you can change it to 0,1,2,3
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids


def refine_dataset(dataset):
    # works for molecule graphs
    if dataset.data.get("node_embs") is not None:
        dataset.data.node_text_feat = dataset.data.node_embs
        dataset.data.node_embs = None
    if dataset.data.get("edge_embs") is not None:
        dataset.data.edge_text_feat = dataset.data.edge_embs
        dataset.data.edge_embs = None
    if dataset.data.get("pretrain_edge_index") is not None:
        dataset.data.edge_index = dataset.data.pretrain_edge_index
        dataset.data.pretrain_edge_index = None
    if dataset.data.get("y") is not None:
        dataset.data.y.squeeze_() 
    return dataset

def span_node_and_edge_idx(dataset):
    # Define node index
    if dataset.data.x.ndim == 1:
        return dataset

    if dataset.ds_name in ['cora', 'pubmed', 'amazon-ratings', 'computers', 'ogbn-products', 'Roman-empire', 
                           'usa','paris', 'facebookpagepage', 'flickr', 'email', 'twitch-de', 'reddit', 
                           'blogcatalog', 'twitch-en', 'deezereurope', 'physics', 'weibo','twitter','facebook', 'fm']:
        dataset.data.xn = dataset.data.x
    elif dataset.ds_name in ['arxiv', 'wikics', 'fb15k237', 'wn18rr']:
        dataset.data.xn = dataset.data.node_text_feat



    # Define edge index
    if dataset.ds_name in ['cora', 'pubmed', 'amazon-ratings', 'computers', 'ogbn-products', 'Roman-empire', 
                           'usa','paris', 'facebookpagepage','flickr', 'email', 'twitch-de', 'reddit', 
                           'blogcatalog', 'twitch-en', 'deezereurope', 'physics','weibo','twitter', 'facebook', 'fm']:
        num_edge_types = 1
    elif dataset.ds_name in ['arxiv', 'wikics']:
        num_edge_types = dataset.data.edge_text_feat.shape[0]  # 1个边类型
    num_edges = dataset.data.edge_index.shape[1]   

    if num_edge_types == 1:
        dataset.data.xe = torch.zeros([num_edges], dtype=torch.long)  # 如果只有一个边类型，那么所有边的特征初始化为0
    else:
        dataset.data.xe = dataset.data.edge_types
    return dataset

def filter_unnecessary_attrs(dataset):
    keys = ['x',
            'xn',
            'xe',
            'edge_index',
            'y'
            ]
    for k, v in dataset.data.to_dict().items():
        if k not in keys:
            dataset.data[k] = None
    return dataset



def gen_entities(root, name):
    if name == "KnowledgeGraph.WN18RR":
        entity2id = {}
        entity_lst = []
        text_lst = []
        with open(osp.join(root, "entity2text.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                tmp = line.strip().split("\t")
                entity_lst.append(tmp[0])
                text_lst.append(tmp[1])

        entity2id = {entity: i for i, entity in enumerate(entity_lst)}
    elif name == "KnowledgeGraph.FB15K237":
        entity_lst = []
        text_lst = []
        with open(osp.join(root, "entity2wikidata.json"), "r") as f:
            data = json.load(f)

        for k in data:
            # print(data[k])
            entity_lst.append(k)
            text_lst.append("entity names: " + data[k]["label"] + ", entity alternatives: " + ", ".join(
                data[k]["alternatives"]) + ". entity descriptions:" + data[k]["description"] if data[k][
                                                                                                    "description"] is
                                                                                                not None else "None")

        entity2id = {entity: i for i, entity in enumerate(entity_lst)}
    else:
        raise NotImplementedError("Dataset " + name + " is not implemented.")
    return entity_lst, text_lst, entity2id


def read_knowledge_graph(root, files, name):
    entity_lst, text_lst, entity2id = gen_entities(root, name)
    relation2id = {}

    converted_triplets = {}
    rel_list = []
    rel = len(relation2id)

    for file_type, file_path in files.items():

        edges = []
        edge_types = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split("\n")[:-1]]
        unknown_entity = 0
        for triplet in file_data:
            if triplet[0] not in entity2id:
                text_lst.append("entity names: Unknown")
                entity_lst.append(triplet[0])
                entity2id[triplet[0]] = len(entity2id)
                unknown_entity += 1
            if triplet[2] not in entity2id:
                text_lst.append("entity names: Unknown")
                entity_lst.append(triplet[2])
                entity2id[triplet[2]] = len(entity2id)
                unknown_entity += 1
            if triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel_list.append(triplet[1])
                rel += 1

            edges.append([entity2id[triplet[0]], entity2id[triplet[2]], ])
            edge_types.append(relation2id[triplet[1]])
        print(unknown_entity)
        converted_triplets[file_type] = [edges, edge_types]

    all_edge_index = torch.concat([torch.tensor(converted_triplets["train"][0]).T, torch.tensor(converted_triplets["valid"][0]).T, torch.tensor(converted_triplets["test"][0]).T], dim=1)
    all_edge_types = torch.concat([torch.tensor(converted_triplets["train"][1]), torch.tensor(converted_triplets["valid"][1]), torch.tensor(converted_triplets["test"][1])], dim=0)

    # My setting
    new_data = pyg.data.data.Data(x=torch.zeros([len(text_lst), 1]),
        edge_index=all_edge_index, edge_types=all_edge_types)

    node_text = ["feature node. entity and entity description: " + ent for ent in text_lst]

    edge_text = ["feature edge. relation between two entities."]

    prompt_edge_text = ["prompt edge", "prompt edge. edge for query graph that is our target",
                        "prompt edge. edge for support graph that is an example"]
    prompt_node_text = ["prompt node. relation type prediction between the connected entities.", ]
    label_text = ["prompt node. relation between two entities. " + relation for relation in rel_list]

    prompt_text_map = {"e2e_link": {"noi_node_text_feat": ["noi_node_text_feat", [0]],
                                    "class_node_text_feat": ["class_node_text_feat", torch.arange(len(label_text))],
                                    "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]},
                       "lr_link": {"noi_node_text_feat": ["noi_node_text_feat", [0]],
                                   "class_node_text_feat": ["class_node_text_feat", torch.arange(len(label_text))],
                                   "prompt_edge_text_feat": ["prompt_edge_text_feat", [0, 1, 2]]}}

    return ([new_data], [node_text, edge_text, label_text, prompt_edge_text, prompt_node_text],
            [converted_triplets, rel_list, prompt_text_map],)

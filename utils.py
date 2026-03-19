import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import RelGraphConv

def calculate_rank(score, target, filter_list):
    score_target = score[target]
    score[filter_list] = score_target - 1
    rank = np.sum(score > score_target) + np.sum(score == score_target) // 2 + 1
    score[target] += 1
    return rank

def get_topk_indices(score, target, filter_list, topk):
    score_target = score[target]
    score[filter_list] = score_target - 1
    score[target] += 1
    sorted_indices = np.argsort(score)[-topk:][::-1]
    return sorted_indices

def metrics(rank):
    mr = np.mean(rank)
    mrr = np.mean(1 / rank)
    hit10 = np.sum(rank < 11) / len(rank)
    hit3 = np.sum(rank < 4) / len(rank)
    hit1 = np.sum(rank < 2) / len(rank)
    return mr, mrr, hit10, hit3, hit1

def build_kg(dataset, num_ent, num_rel):
    path = "./data/{}/train.txt".format(dataset)
    f = open(path, 'r')
    e1, e2, rels = [], [], []
    entity_map = {}
    for line in f.readlines()[1:]:
        h, t, r = line[:-1].split('\t')
        e1.append(int(h))
        e2.append(int(t))
        rels.append(int(r))
        entity_map[int(h)] = 1
        entity_map[int(t)] = 1
    for i in range(0, num_ent):
        e1.append(i)
        e2.append(i)
        rels.append(num_rel)
    graph = dgl.graph((e1, e2))
    return graph, rels

def load_entities(file_path):
    """
    从 entities2Id 文件中读取内容，返回 id->name 的映射字典
    文件格式：Name ID
    """
    id2name = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            name, idx = line.strip().split()
            id2name[int(idx)] = name
    return id2name


def get_names_from_ids(id_list, id2name):
    """
    根据数字 id_list 返回对应的名称列表
    """
    return [id2name.get(i, f"<未知ID:{i}>") for i in id_list]

class AReLU(nn.Module):
    def __init__(self, alpha=0.90, beta=2.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha]))
        self.beta = nn.Parameter(torch.tensor([beta]))

    def forward(self, input):
        alpha = torch.clamp(self.alpha, min=0.01, max=0.99)
        beta = 1 + torch.sigmoid(self.beta)

        return F.relu(input) * beta - F.relu(-input) * alpha


if __name__ == "__main__":
    file_path = "data/MKG-Y/entity2id.txt"
    id2name = load_entities(file_path)

    # 示例：输入一个数字列表
    input_ids = [6610, 414, 14279, 2668]
    names = get_names_from_ids(input_ids, id2name)
    print("输入ID：", input_ids)
    print("对应名称：", names)
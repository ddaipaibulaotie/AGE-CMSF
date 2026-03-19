import json
import torch

from collections import Counter, defaultdict

def load_ent_map(dataset):
    ent_map = {}
    if dataset == "DB15K":
        f = open("data/{}/entities.txt".format(dataset), "r")
        for line in f.readlines():
            ent = line.replace('\n', '')
            ent_map[ent] = ent
    elif dataset == "MDKG":
        f = open("data/{}/entity2id.txt".format(dataset), "r")
        for line in f.readlines():
            ent, id = line[:-1].split('\t')
            ent_map[ent] = int(id)
    else:
        f = open("data/{}/entity2id.txt".format(dataset), "r")
        for line in f.readlines():
            ent, id = line[:-1].split(' ')
            ent_map[ent] = int(id)
    return ent_map


def get_entity_visual_tokens(dataset, max_num):
    if dataset == "MDKG":
        return get_entity_visual_tokens_mdkg(dataset, max_num)
    tokenized_result = json.load(open("tokens/{}-visual.json".format(dataset), "r"))
    token_dict = defaultdict(list)
    entity_dict = load_ent_map(dataset)
    for i in range(8192 + 1):
        token_dict[i] = []
    for entity in tokenized_result:
        token_count = Counter(tokenized_result[entity])
        selected_tokens = token_count.most_common(max_num)
        for (token, num) in selected_tokens:
            token_dict[token].append(entity)
    
    num_count = [(k, len(token_dict[k])) for k in token_dict]
    num_count = sorted(num_count, key=lambda x: -x[1])
    token_ids = list(token_dict.keys())
    # print(token_ids)
    entity_to_token = defaultdict(list)
    for i in range(len(token_ids)):
        for ent in token_dict[token_ids[i]]:
            entity_to_token[entity_dict[ent]].append(i)
    # json.dump(entity_to_token, open("{}-tokens-{}.json".format(dataset, max_num), "w"))
    entid_tokens = []
    ent_key_mask = []
    for i in range(len(entity_dict)):
        key = str(i) if dataset == "DB15K" else i
        if entity_to_token[key] != []:
            entid_tokens.append(entity_to_token[key])
            ent_key_mask.append(([False] * max_num))
        else:
            entid_tokens.append([8192] * max_num)
            ent_key_mask.append(([True] * max_num))
    return torch.LongTensor(entid_tokens), torch.BoolTensor(ent_key_mask).to(torch.device('cpu'))
    # return torch.LongTensor(entid_tokens), torch.BoolTensor(ent_key_mask).cuda()


def get_entity_visual_tokens_with_limit_db15k(dataset, max_num, max_img=None):
    assert max_img is not None
    tokenized_result = json.load(open("tokens/{}-visual.json".format(dataset), "r"))
    token_dict = defaultdict(list)
    entity_dict = load_ent_map(dataset)
    for id in tokenized_result.keys():
        tokenized_result[id] = tokenized_result[id][0: min(len(tokenized_result[id]), 196 * max_img)]
    for i in range(8192 + 1):
        token_dict[i] = []
    for entity in tokenized_result:
        token_count = Counter(tokenized_result[entity])
        selected_tokens = token_count.most_common(max_num)
        for (token, num) in selected_tokens:
            token_dict[token].append(entity)
    
    num_count = [(k, len(token_dict[k])) for k in token_dict]
    num_count = sorted(num_count, key=lambda x: -x[1])
    token_ids = list(token_dict.keys())
    # print(token_ids)
    entity_to_token = defaultdict(list)
    for i in range(len(token_ids)):
        for ent in token_dict[token_ids[i]]:
            entity_to_token[entity_dict[ent]].append(i)
    # json.dump(entity_to_token, open("{}-tokens-{}.json".format(dataset, max_num), "w"))
    entid_tokens = []
    ent_key_mask = []
    for i in range(len(entity_dict)):
        if entity_to_token[str(i)] != []:
            entid_tokens.append(entity_to_token[str(i)])
            ent_key_mask.append(([False] * max_num))
        else:
            entid_tokens.append([8192] * max_num)
            ent_key_mask.append(([True] * max_num))
    return torch.LongTensor(entid_tokens), torch.BoolTensor(ent_key_mask).to(torch.device('cpu'))
    # return torch.LongTensor(entid_tokens), torch.BoolTensor(ent_key_mask).cuda()


def get_entity_visual_tokens_with_limit(dataset, max_num, max_img=None):
    if dataset == "DB15K":
        return get_entity_visual_tokens_with_limit_db15k(dataset, max_num, max_img)
    if dataset == "MDKG":
        return get_entity_visual_tokens_with_limit_mdkg(dataset, max_num, max_img)
    assert max_img is not None
    tokenized_result = json.load(open("tokens/{}-visual-v2.json".format(dataset), "r"))
    token_dict = defaultdict(list)
    entity_dict = load_ent_map(dataset)
    for id in tokenized_result.keys():
        tokenized_result[id] = tokenized_result[id][0: min(len(tokenized_result[id]), 196 * max_img)]
    for i in range(8192 + 1):
        token_dict[i] = []
    for entity in tokenized_result:
        token_count = Counter(tokenized_result[entity])
        selected_tokens = token_count.most_common(max_num)
        for (token, num) in selected_tokens:
            token_dict[token].append(entity)
    
    num_count = [(k, len(token_dict[k])) for k in token_dict]
    num_count = sorted(num_count, key=lambda x: -x[1])
    token_ids = list(token_dict.keys())
    entity_to_token = defaultdict(list)
    for i in range(len(token_ids)):
        for ent in token_dict[token_ids[i]]:
            entity_to_token[entity_dict[ent]].append(i)
    entid_tokens = []
    ent_key_mask = []
    for i in range(len(entity_dict)):
        if entity_to_token[i] != []:
            entid_tokens.append(entity_to_token[i])
            ent_key_mask.append(([False] * max_num))
        else:
            entid_tokens.append([8192] * max_num)
            ent_key_mask.append(([True] * max_num))
    return torch.LongTensor(entid_tokens), torch.BoolTensor(ent_key_mask).to(torch.device('cpu'))
    # return torch.LongTensor(entid_tokens), torch.BoolTensor(ent_key_mask).cuda()


def get_entity_textual_tokens_db15K(dataset, max_num):
    tokenized_result = json.load(open("tokens/{}-textual.json".format(dataset), "r"))
    token_dict = defaultdict(list)
    entity_dict = load_ent_map(dataset)
    for i in range(30522 + 1):
        token_dict[i] = []
    for entity in tokenized_result:
        token_count = Counter(tokenized_result[entity])
        selected_tokens = token_count.most_common(max_num)
        for (token, num) in selected_tokens:
            token_dict[token].append(entity)
    num_count = [(k, len(token_dict[k])) for k in token_dict]
    num_count = sorted(num_count, key=lambda x: -x[1])
    token_ids = list(token_dict.keys())
    # print(token_ids)
    entity_to_token = defaultdict(list)
    for i in range(len(token_ids)):
        for ent in token_dict[token_ids[i]]:
            entity_to_token[entity_dict[ent]].append(i)
    entid_tokens = []
    ent_key_mask = []
    for i in range(len(entity_dict)):
        if entity_to_token[str(i)] == max_num:
            entid_tokens.append(entity_to_token[str(i)])
            ent_key_mask.append(([False] * max_num))
        else:
            s = entity_to_token[str(i)]
            entid_tokens.append(s + [14999] * (max_num - len(s)))
            ent_key_mask.append(([False] * len(s) + [True] * (max_num - len(s))))
    return torch.LongTensor(entid_tokens), torch.BoolTensor(ent_key_mask).to(torch.device('cpu'))
    # return torch.LongTensor(entid_tokens), torch.BoolTensor(ent_key_mask).cuda()

def get_entity_textual_tokens(dataset, max_num):
    if dataset == "DB15K":
        return get_entity_textual_tokens_db15K(dataset, max_num)
    if dataset == "MDKG":
        return get_entity_textual_tokens_mdkg(dataset, max_num)
    tokenized_result = json.load(open("tokens/{}-textual.json".format(dataset), "r"))
    token_dict = defaultdict(list)
    entity_dict = load_ent_map(dataset)
    for i in range(30522 + 1):
        token_dict[i] = []
    for entity in tokenized_result:
        token_count = Counter(tokenized_result[entity])
        selected_tokens = token_count.most_common(max_num)
        for (token, num) in selected_tokens:
            token_dict[token].append(entity)
    num_count = [(k, len(token_dict[k])) for k in token_dict]
    num_count = sorted(num_count, key=lambda x: -x[1])
    token_ids = list(token_dict.keys())
    entity_to_token = defaultdict(list)
    for i in range(len(token_ids)):
        for ent in token_dict[token_ids[i]]:
            entity_to_token[entity_dict[ent]].append(i)
    entid_tokens = []
    ent_key_mask = []
    for i in range(len(entity_dict)):
        if entity_to_token[i] == max_num:
            entid_tokens.append(entity_to_token[i])
            ent_key_mask.append(([False] * max_num))
        else:
            s = entity_to_token[i]
            entid_tokens.append(s + [14999] * (max_num - len(s)))
            ent_key_mask.append(([False] * len(s) + [True] * (max_num - len(s))))
    return torch.LongTensor(entid_tokens), torch.BoolTensor(ent_key_mask).to(torch.device('cpu'))
    # return torch.LongTensor(entid_tokens), torch.BoolTensor(ent_key_mask).cuda()


def get_entity_textual_tokens_FB15K237(dataset, max_num):
    assert dataset == "FB15K-237"
    tokenized_result = json.load(open("tokens/{}-textual.json".format(dataset), "r"))
    token_dict = defaultdict(list)
    entity_dict = load_ent_map(dataset)
    for i in range(30522 + 1):
        token_dict[i] = []
    for entity in tokenized_result:
        token_count = Counter(tokenized_result[entity])
        selected_tokens = token_count.most_common(max_num)
        for (token, num) in selected_tokens:
            token_dict[token].append(entity)
    num_count = [(k, len(token_dict[k])) for k in token_dict]
    num_count = sorted(num_count, key=lambda x: -x[1])
    token_ids = list(token_dict.keys())
    entity_to_token = defaultdict(list)
    for i in range(len(token_ids)):
        for ent in token_dict[token_ids[i]]:
            entity_to_token[entity_dict[ent]].append(i)
    entid_tokens = []
    ent_key_mask = []
    for i in entity_dict:
        if entity_to_token[i] == max_num:
            entid_tokens.append(entity_to_token[ent])
            ent_key_mask.append(([False] * max_num))
        else:
            s = entity_to_token[i]
            entid_tokens.append(s + [14999] * (max_num - len(s)))
            ent_key_mask.append(([False] * len(s) + [True] * (max_num - len(s))))
    return torch.LongTensor(entid_tokens), torch.BoolTensor(ent_key_mask).to(torch.device('cpu'))
    # return torch.LongTensor(entid_tokens), torch.BoolTensor(ent_key_mask).cuda()


def get_entity_visual_tokens_FB15K237(dataset, max_num):
    assert dataset == "FB15K-237"
    tokenized_result = json.load(open("tokens/{}-visual.json".format(dataset), "r"))
    token_dict = defaultdict(list)
    entity_dict = load_ent_map(dataset)
    for i in range(8192 + 1):
        token_dict[i] = []
    for entity in tokenized_result:
        token_count = Counter(tokenized_result[entity])
        selected_tokens = token_count.most_common(max_num)
        for (token, num) in selected_tokens:
            token_dict[token].append(entity)
    
    num_count = [(k, len(token_dict[k])) for k in token_dict]
    num_count = sorted(num_count, key=lambda x: -x[1])
    token_ids = list(token_dict.keys())
    # print(token_ids)
    entity_to_token = defaultdict(list)
    for i in range(len(token_ids)):
        for ent in token_dict[token_ids[i]]:
            entity_to_token[entity_dict[ent]].append(i)
    # json.dump(entity_to_token, open("{}-tokens-{}.json".format(dataset, max_num), "w"))
    entid_tokens = []
    ent_key_mask = []
    for i in entity_dict:
        key = i
        if entity_to_token[key] != []:
            entid_tokens.append(entity_to_token[key])
            ent_key_mask.append(([False] * max_num))
        else:
            entid_tokens.append([8192] * max_num)
            ent_key_mask.append(([True] * max_num))
    return torch.LongTensor(entid_tokens), torch.BoolTensor(ent_key_mask).to(torch.device('cpu'))
    # return torch.LongTensor(entid_tokens), torch.BoolTensor(ent_key_mask).cuda()


# =========================
# 原始词表/码本 id 对齐版（不改原函数）
# =========================

from collections import Counter, defaultdict
import json, torch

def _pad_and_mask(seq, max_num, pad_val):
    s = list(seq)
    if len(s) >= max_num:
        return s[:max_num], [False] * max_num
    pad_len = max_num - len(s)
    return s + [pad_val] * pad_len, ([False] * len(s) + [True] * pad_len)

def get_entity_textual_tokens_mdkg(dataset, max_num, vocab_size=6762, pad_idx=6761):
    """
    适配：tokens/{dataset}-textual.json 的 key 为实体名称，value 为【原始 BERT token id 列表】。
    将原始大 id 压缩成 0..(vocab_size-2) 的范围（最后一个索引 pad_idx 作为 PAD）。
    返回：
      entid_tokens: LongTensor[num_entities, max_num]（值 ∈ [0, vocab_size-1]）
      ent_key_mask: BoolTensor[num_entities, max_num]
    """
    assert vocab_size == pad_idx + 1, "要求 vocab_size = pad_idx + 1，例如 15000 / 14999"
    PAD_TXT = pad_idx

    # 读取：名称 -> 原始 token id 序列
    tokenized_result = json.load(open("tokens/{}-textual.json".format(dataset), "r"))
    # 名称 -> 实体 id
    entity_dict = load_ent_map(dataset)

    # 1) 统计全局频次，选出前 vocab_size-1 个 id 作为“压缩词表”
    global_counter = Counter()
    for name, toks in tokenized_result.items():
        if name not in entity_dict:
            continue
        # 这里累加原始 id 的出现次数（你也可以用 set(toks) 统计“是否出现过”，按需调整）
        global_counter.update(toks)

    # 选 top-(vocab_size-1) 个原始 id
    top_raw = [tid for tid, _ in global_counter.most_common(vocab_size - 1)]

    # 构造 raw_id -> compact_id 的映射（0..vocab_size-2），其余映射到 PAD
    raw2cmp = {tid: i for i, tid in enumerate(top_raw)}

    # 2) 构建“倒排”思路与 per-entity top-K，但注意要把原始 id 映射成压缩 id
    # 统计每个实体自身的词频，选出该实体 top-K 的原始 id，然后映射为 compact id
    num_ents = len(entity_dict)
    entid_tokens, ent_key_mask = [], []

    # 为了按 id 顺序输出，先做 id->name 的反查
    id2name = {v: k for k, v in entity_dict.items()}

    for i in range(num_ents):
        name = id2name[i]
        toks = tokenized_result.get(name, [])
        cnt = Counter(toks)
        # 该实体内的 top-K 原始 id
        per_ent_top = [tid for tid, _ in cnt.most_common(max_num)]
        # 映射到压缩 id（不在映射内的 → PAD）
        mapped = [raw2cmp.get(tid, PAD_TXT) for tid in per_ent_top]
        fixed, mask = _pad_and_mask(mapped, max_num, PAD_TXT)
        entid_tokens.append(fixed)
        ent_key_mask.append(mask)

    return torch.LongTensor(entid_tokens), torch.BoolTensor(ent_key_mask).to(torch.device('cpu'))


def get_entity_visual_tokens_mdkg(dataset, max_num):
    """
    视觉：tokens/{dataset}-visual.json 的 key 为实体名称，value 为【原始视觉 codebook id 列表】。
    规则：
      - 预填充 token_dict[0..8192]（8192 为 PAD，占位保留），确保索引空间与码本一致；
      - 对每个实体统计其视觉 tokens 的词频，选 top-K（K=max_num）的【原始 token id】；
      - 输出 shape=[num_entities, max_num]，用 PAD=8192 右填充。
    """
    PAD_VIS = 8192
    CODEBOOK = 8192
    tokenized_result = json.load(open("tokens/{}-visual.json".format(dataset), "r"))  # name -> [tok ids]
    entity_dict = load_ent_map(dataset)  # name -> id

    token_dict = defaultdict(list)
    for i in range(CODEBOOK + 1):  # 0..8192（8192 作为 PAD 占位）
        token_dict[i] = []

    for name, toks in tokenized_result.items():
        if name not in entity_dict:
            continue
        cnt = Counter(toks)
        selected = cnt.most_common(max_num)
        for (tok_id, _c) in selected:
            if 0 <= tok_id <= CODEBOOK:
                token_dict[tok_id].append(name)

    token_ids = list(token_dict.keys())  # 0..8192
    entity_to_token = defaultdict(list)  # eid -> [原始 codebook id]
    for tok_id in token_ids:
        for name in token_dict[tok_id]:
            eid = entity_dict[name]
            entity_to_token[eid].append(tok_id)

    num_ents = len(entity_dict)
    entid_tokens, ent_key_mask = [], []
    for i in range(num_ents):
        seq = entity_to_token.get(i, [])
        fix, msk = _pad_and_mask(seq, max_num, PAD_VIS)
        entid_tokens.append(fix)
        ent_key_mask.append(msk)
    return torch.LongTensor(entid_tokens), torch.BoolTensor(ent_key_mask).to(torch.device('cpu'))


def get_entity_visual_tokens_with_limit_mdkg(dataset, max_num, max_img=None):
    """
    与上类似，但先把每个实体的视觉 tokens 限制到 196*max_img（贴合 DB15K 的每图 196 patch）。
    """
    assert max_img is not None, "请提供 max_img"
    PAD_VIS = 8192
    CODEBOOK = 8192
    tokenized_result = json.load(open("tokens/{}-visual.json".format(dataset), "r"))  # name -> [tok ids]
    entity_dict = load_ent_map(dataset)

    # 限制每实体的 token 数量
    limited = {}
    for name, toks in tokenized_result.items():
        limited[name] = toks[: min(len(toks), 196 * max_img)]

    token_dict = defaultdict(list)
    for i in range(CODEBOOK + 1):
        token_dict[i] = []

    for name, toks in limited.items():
        if name not in entity_dict:
            continue
        cnt = Counter(toks)
        selected = cnt.most_common(max_num)
        for (tok_id, _c) in selected:
            if 0 <= tok_id <= CODEBOOK:
                token_dict[tok_id].append(name)

    token_ids = list(token_dict.keys())
    entity_to_token = defaultdict(list)
    for tok_id in token_ids:
        for name in token_dict[tok_id]:
            eid = entity_dict[name]
            entity_to_token[eid].append(tok_id)

    num_ents = len(entity_dict)
    entid_tokens, ent_key_mask = [], []
    for i in range(num_ents):
        seq = entity_to_token.get(i, [])
        fix, msk = _pad_and_mask(seq, max_num, PAD_VIS)
        entid_tokens.append(fix)
        ent_key_mask.append(msk)
    return torch.LongTensor(entid_tokens), torch.BoolTensor(ent_key_mask).to(torch.device('cpu'))

if __name__ == "__main__":
    dataset = "DB15K"
    max_token_num = 8
    a, b = get_entity_visual_tokens_FB15K237(dataset, max_token_num)
    print(a)
    
    
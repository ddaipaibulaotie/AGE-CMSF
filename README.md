# AGE-CMSF

This repository provides the official implementation of **AGE-CMSF**, a method for **Multimodal Knowledge Graph Entity Completion**.

The code corresponds to the paper:

> **Adaptive Gated Embedding and Cross Modal Semantic Fusion for Multimodal Knowledge Graph Completion**  
> (under review)

---

## Overview

AGE-CMSF is designed to improve multimodal knowledge graph entity completion by addressing two key challenges:

- Modality-internal feature imbalance via **Adaptive Gated Embedding (AGE)**
- Cross-modal semantic alignment and noise suppression via **Cross-Modal Semantic Fusion (CMSF)**

The method integrates structural, textual, and visual information to construct robust multimodal entity representations for entity completion tasks.

---

### Requirements:

```shell
numpy==1.24.2
scikit_learn==1.2.2
torch==2.0.0
tqdm==4.64.1
node2vec==0.4.6 
```

### Download:

DB15K, MKG-W and MKG-Y: https://github.com/quqxui/MMRNS


### Run Script:

train on DB15K

```shell
sh train_db15k.sh 
```

train on MKG-W

```shell
sh train_mkgw.sh 
```

train on MKG-Y

```shell
sh train_mkgy.sh 
```

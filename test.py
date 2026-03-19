#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import sys
import time
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F

from dataset import VTKG
from model import AGECMSF
from utils import calculate_rank, metrics  # we reuse existing utilities
from merge_tokens import get_entity_visual_tokens, get_entity_textual_tokens


def to_name(mapping: Dict[int, str], idx: int) -> str:
    if mapping is None:
        return str(idx)
    try:
        return mapping.get(idx, str(idx))
    except Exception:
        return str(idx)


def main():
    parser = argparse.ArgumentParser()
    # Data/model hyperparams (mirror train script defaults)
    parser.add_argument('--data', default="MKG-Y", type=str)
    parser.add_argument('--dim', default=256, type=int)
    parser.add_argument('--num_layer_enc_ent', default=1, type=int)
    parser.add_argument('--num_layer_enc_rel', default=1, type=int)
    parser.add_argument('--num_layer_dec', default=2, type=int)
    parser.add_argument('--num_head', default=8, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--dropout', default=0.01, type=float)
    parser.add_argument('--emb_dropout', default=0.9, type=float)
    parser.add_argument('--vis_dropout', default=0.4, type=float)
    parser.add_argument('--txt_dropout', default=0.4, type=float)
    parser.add_argument('--max_img_num', default=3, type=int)
    parser.add_argument('--max_vis_token', default=8, type=int)
    parser.add_argument('--max_txt_token', default=8, type=int)
    parser.add_argument('--score_function', default="tucker", type=str)

    # Evaluation settings
    parser.add_argument('--batch_size', default=256, type=int, help='Only used for optional prefill. Not needed typically.')
    parser.add_argument('--no_prefill', action='store_true', help='Skip prefill_valid step.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to .ckpt file from training.')
    parser.add_argument('--output_prefix', default=None, type=str, help='Prefix for output files. Defaults to data name + timestamp.')
    parser.add_argument('--limit', default=None, type=int, help='Optionally limit number of test triples for a quick run.')

    args = parser.parse_args()

    device = torch.device(args.device)

    # Build dataset/loader (consistent with train)
    # VTKG should expose: num_ent, num_rel, test (list of triples), filter_dict (dict for filtered settings)
    KG = VTKG(args.data, logger=None, max_vis_len=args.max_img_num)
    # Optional: run a prefill pass to set head_valid/tail_valid like training (harmless if skipped)
    if not args.no_prefill:
        try:
            from torch.utils.data import DataLoader
            loader = DataLoader(KG, batch_size=args.batch_size, shuffle=False)
            model_tmp = None  # will be defined later; we store triplets first
            prefill_buffer = []
        except Exception:
            loader = None
            prefill_buffer = []


    # Prepare multimodal token indices and masks
    visual_token_index, visual_key_mask = get_entity_visual_tokens(dataset=args.data, max_num=args.max_vis_token)
    text_token_index, text_key_mask = get_entity_textual_tokens(dataset=args.data, max_num=args.max_txt_token)

    visual_token_index = visual_token_index.to(device)
    text_token_index = text_token_index.to(device)

    # Build model (mirror train script wiring)
    model = AGECMSF(
        num_ent=KG.num_ent,
        num_rel=KG.num_rel,
        ent_vis_mask=visual_key_mask,
        ent_txt_mask=text_key_mask,
        dim_str=args.dim,
        num_head=args.num_head,
        dim_hid=args.hidden_dim,
        num_layer_enc_ent=args.num_layer_enc_ent,
        num_layer_enc_rel=args.num_layer_enc_rel,
        num_layer_dec=args.num_layer_dec,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
        vis_dropout=args.vis_dropout,
        txt_dropout=args.txt_dropout,
        visual_token_index=visual_token_index,
        text_token_index=text_token_index,
        score_function=args.score_function,
        dataset=args.data
    ).to(device)

    # Run prefill if we captured any batch-labels earlier
    if not args.no_prefill:
        try:
            if 'loader' in locals() and loader is not None:
                with torch.no_grad():
                    for batch, label in loader:
                        batch = batch.to(device)
                        label = label.to(device)
                        model.prefill_valid(batch, label)
        except Exception as e:
            # Prefill is optional; continue if dataset doesn't support it
            pass


    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_key = 'model_state_dict' if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else None
    if state_key:
        model.load_state_dict(ckpt[state_key], strict=False)
    else:
        # Fallback: assume the checkpoint is a raw state_dict
        model.load_state_dict(ckpt, strict=False)

    model.eval()

    # Try to get id->name mappings if available (optional)
    id2ent = getattr(KG, 'id2ent', None) if hasattr(KG, 'id2ent') else None
    id2rel = getattr(KG, 'id2rel', None) if hasattr(KG, 'id2rel') else None

    # Prepare outputs
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    prefix = args.output_prefix or f"{args.data}_{timestamp}"
    txt_path = f"{prefix}_test_log.txt"
    csv_path = f"{prefix}_test_results.csv"

    total_ranks = []  # collect ranks for MR/MRR/hits
    n_examples = 0

    with torch.no_grad(), open(txt_path, "w", encoding="utf-8") as ftxt, open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "query_type", "h_id", "r_id", "t_id",
            "h_name", "r_name", "t_name",
            "rank", "correct_in_top10", "top10_ids", "top10_names"
        ])

        # Compute static entity and relation embeddings once
        ent_embs, rel_embs, _ = model()

        def filtered_topk_and_rank(scores: np.ndarray, true_idx: int, filt_set: List[int], k: int = 10):
            # For rank: use project utility which handles filtered setting itself
            r = calculate_rank(scores, true_idx, filt_set)
            # For Top-k list: apply filtered protocol by masking other true entities (except the target)
            filtered_scores = scores.copy()
            for eid in filt_set:
                if eid != true_idx:
                    filtered_scores[eid] = -np.inf
            order = np.argsort(-filtered_scores)
            topk = order[:k].tolist()
            return r, topk

        test_list = KG.test if args.limit is None else KG.test[:args.limit]

        for (h, r, t) in test_list:
            # Update relation-wise probabilities (as in train/valid) before scoring
            model.contrastive_loss_relation(rel_embs, loss_flag=False)

            n_examples += 1
            # HEAD prediction: (?, r, t)
            head_query = torch.tensor([[KG.num_ent + KG.num_rel, r + KG.num_ent, t + KG.num_rel]], device=device)
            head_scores = model.score(ent_embs, rel_embs, head_query)[0].detach().cpu().numpy()
            head_filt = KG.filter_dict[(-1, r, t)]
            head_rank, head_top10 = filtered_topk_and_rank(head_scores, h, head_filt, k=10)
            total_ranks.append(head_rank)
            head_correct = int(h in head_top10)

            # Log HEAD
            hname = to_name(id2ent, h); rname = to_name(id2rel, r); tname = to_name(id2ent, t)
            top10_names = [to_name(id2ent, idx) for idx in head_top10]
            ftxt.write(f"HEAD | (?, {rname}, {tname}) -> rank={head_rank}, top10={head_top10}, correct={'✓' if head_correct else '✗'}\n")
            writer.writerow(["head", h, r, t, hname, rname, tname, head_rank, head_correct, json.dumps(head_top10), json.dumps(top10_names)])

            # TAIL prediction: (h, r, ?)
            tail_query = torch.tensor([[h + KG.num_rel, r + KG.num_ent, KG.num_ent + KG.num_rel]], device=device)
            tail_scores = model.score(ent_embs, rel_embs, tail_query)[0].detach().cpu().numpy()
            tail_filt = KG.filter_dict[(h, r, -1)]
            tail_rank, tail_top10 = filtered_topk_and_rank(tail_scores, t, tail_filt, k=10)
            total_ranks.append(tail_rank)
            tail_correct = int(t in tail_top10)

            # Log TAIL
            top10_names = [to_name(id2ent, idx) for idx in tail_top10]
            ftxt.write(f"TAIL | ({hname}, {rname}, ?) -> rank={tail_rank}, top10={tail_top10}, correct={'✓' if tail_correct else '✗'}\n")
            writer.writerow(["tail", h, r, t, hname, rname, tname, tail_rank, tail_correct, json.dumps(tail_top10), json.dumps(top10_names)])

        # Summarize
        total_ranks = np.array(total_ranks, dtype=np.float32)
        mr, mrr, hit10, hit3, hit1 = metrics(total_ranks)
        summary = f"\nSUMMARY over {n_examples} triples (both head & tail tasks counted):\n" \
                  f"MR={mr:.4f}, MRR={mrr:.6f}, Hit@10={hit10:.4f}, Hit@3={hit3:.4f}, Hit@1={hit1:.4f}\n"
        print(summary.strip())
        ftxt.write(summary)

    print(f"\nSaved detailed TXT log to: {txt_path}")
    print(f"Saved CSV results to:     {csv_path}")


if __name__ == "__main__":
    main()

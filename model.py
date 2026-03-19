import math
from utils import AReLU
from model_tool import *
from model_layer import AGELinear
import torch.nn as nn
import torch.nn.functional as F

class CrossModelAttNew(nn.Module):

    def __init__(self, feature_dim_img: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.0)

        # lazy init
        self.txt_proj = None

        # for analysis / visualization only
        self.last_attn = None

    @staticmethod
    def _flatten_img(x):
        if x.dim() == 4:  # [B, C, H, W]
            B, C, H, W = x.shape
            return x.view(B, C, H * W), ('2d', H, W)
        elif x.dim() == 3:  # [B, C, S]
            B, C, S = x.shape
            return x, ('1d', S)
        else:
            raise ValueError(f'img_feat dim must be 3 or 4, got {x.dim()}')

    @staticmethod
    def _restore_img(x_flat, meta):
        kind = meta[0]
        if kind == '2d':
            _, H, W = meta
            B, C, S = x_flat.shape
            return x_flat.view(B, C, H, W)
        else:
            return x_flat

    def _prep_text_as_BCS(self, text_feat, C_target, S_target):
        if text_feat.dim() == 3:  # [B, L, D]
            B, L, D = text_feat.shape
            if self.txt_proj is None or \
               self.txt_proj.in_features != D or \
               self.txt_proj.out_features != C_target:
                self.txt_proj = nn.Linear(D, C_target, bias=False).to(text_feat.device)

            t = self.txt_proj(text_feat)          # [B, L, C]
            t = t.transpose(1, 2).contiguous()    # [B, C, L]
            if L != S_target:
                t = F.interpolate(t, size=S_target, mode='linear', align_corners=False)
            return t

        elif text_feat.dim() == 4:  # [B, C, Ht, Wt]
            B, C_txt, Ht, Wt = text_feat.shape
            if C_txt != C_target:
                conv = nn.Conv2d(C_txt, C_target, kernel_size=1, bias=False).to(text_feat.device)
                text_feat = conv(text_feat)
            text_feat = F.interpolate(
                text_feat,
                size=(int(S_target ** 0.5), int(S_target ** 0.5)),
                mode='bilinear',
                align_corners=False
            )
            return text_feat.view(B, C_target, -1)

        elif text_feat.dim() == 3:  # [B, C_txt, St]
            B, C_txt, St = text_feat.shape
            if C_txt != C_target:
                lin = nn.Linear(C_txt, C_target, bias=False).to(text_feat.device)
                t = lin(text_feat.transpose(1, 2))
                t = t.transpose(1, 2).contiguous()
            else:
                t = text_feat
            if St != S_target:
                t = F.interpolate(t, size=S_target, mode='linear', align_corners=False)
            return t

        else:
            raise ValueError(f'Unsupported text_feat shape: {text_feat.shape}')

    def forward(self, img_feat: torch.Tensor, text_feat: torch.Tensor, return_attn: bool = False):
        # 1) image -> [B, C, S]
        img_flat, meta = self._flatten_img(img_feat)
        B, C, S = img_flat.shape

        # 2) text -> [B, C, S]
        text_BCS = self._prep_text_as_BCS(text_feat, C_target=C, S_target=S)

        # 3) channel-wise cross-attention
        q = img_flat                         # [B, C, S]
        k = text_BCS.transpose(1, 2)         # [B, S, C]
        v = text_BCS                         # [B, C, S]

        scale = S ** -0.5
        attn = torch.bmm(q, k) * scale       # [B, C, C]
        attn = self.softmax(attn)

        # save for visualization (analysis only)
        self.last_attn = attn.detach()

        out = torch.bmm(attn, v)             # [B, C, S]
        out = self.dropout(out)

        out = img_flat + self.gamma * out
        out = self._restore_img(out, meta)

        if return_attn:
            return out, attn
        return out


class AGECMSF(nn.Module):
    def __init__(
            self,
            num_ent,
            num_rel,
            ent_vis_mask,
            ent_txt_mask,
            dim_str,
            num_head,
            dim_hid,
            num_layer_enc_ent,
            num_layer_enc_rel,
            num_layer_dec,
            dropout=0.1,
            emb_dropout=0.9,
            vis_dropout=0.4,
            txt_dropout=0.1,
            visual_token_index=None,
            text_token_index=None,
            score_function="tucker",
            dataset="MKG-W"
    ):
        super(AGECMSF, self).__init__()
        self.dim_str = dim_str
        self.num_head = num_head
        self.dim_hid = dim_hid
        self.num_ent = num_ent
        self.num_rel = num_rel

        visual_tokens = torch.load("tokens/visual.pth", map_location=torch.device('cpu')) #(8193, 32)
        textual_tokens = torch.load("tokens/{}-textual.pth".format(dataset),  map_location=torch.device('cpu')) #(32000, 4096)
        structure_tokens = torch.load("tokens/{}-node2vec.pth".format(dataset), map_location=torch.device('cpu')) #(15000, 8)
        # visual_tokens = torch.load("tokens/visual.pth")
        # textual_tokens = torch.load("tokens/{}-textual.pth".format(dataset))
        # structure_tokens = torch.load("tokens/{}-node2vec.pth".format(dataset))
        self.visual_token_index = visual_token_index
        self.visual_token_embedding = nn.Embedding.from_pretrained(visual_tokens).requires_grad_(False)
        self.text_token_index = text_token_index
        self.text_token_embedding = nn.Embedding.from_pretrained(textual_tokens).requires_grad_(False)
        self.score_function = score_function

        self.visual_token_embedding.requires_grad_(False)
        self.text_token_embedding.requires_grad_(False)

        # false_ents = torch.full((self.num_ent, 1), False).cuda()
        false_ents = torch.full((self.num_ent,1),False).to(torch.device('cpu'))
        self.ent_mask = torch.cat([false_ents, false_ents, ent_vis_mask, ent_txt_mask, ent_vis_mask], dim=1)
        # self.ent_mask = torch.cat([false_ents, false_ents, ent_vis_mask, ent_txt_mask], dim = 1)

        self.ent_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.rel_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.ent_embeddings = structure_tokens.requires_grad_(False).unsqueeze(1)
        self.rel_embeddings = nn.Parameter(torch.Tensor(num_rel, 1, dim_str))
        self.lp_token = nn.Parameter(torch.Tensor(1, dim_str))

        self.str_ent_ln = nn.LayerNorm(dim_str)
        self.str_rel_ln = nn.LayerNorm(dim_str)
        self.vis_ln = nn.LayerNorm(dim_str)
        self.txt_ln = nn.LayerNorm(dim_str)

        self.embdr = nn.Dropout(p=emb_dropout)
        self.visdr = nn.Dropout(p=vis_dropout)
        self.txtdr = nn.Dropout(p=txt_dropout)

        self.pos_str_ent = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_vis_ent = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_txt_ent = nn.Parameter(torch.Tensor(1, 1, dim_str))

        self.pos_str_rel = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_vis_rel = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_txt_rel = nn.Parameter(torch.Tensor(1, 1, dim_str))

        self.pos_head = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_rel = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_tail = nn.Parameter(torch.Tensor(1, 1, dim_str))

        self.proj_ent_vis = AGELinear(visual_tokens.size(1), dim_str) #AGE
        self.proj_ent_txt = AGELinear(textual_tokens.size(1), dim_str) #AGE
        self.proj_s = nn.Linear(structure_tokens.size(1), dim_str)

        ent_encoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout,
                                                       batch_first=True)  # dim_str=256, num_head=8 dim_hid=1024, dropout=0.01
        self.ent_encoder = nn.TransformerEncoder(ent_encoder_layer, num_layer_enc_ent)  # num_layer_enc_ent = 1
        rel_encoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first=True)
        self.rel_encoder = nn.TransformerEncoder(rel_encoder_layer, num_layer_enc_rel)  # num_layer_enc_rel=1
        decoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layer_dec)  # num_layer_dec = 2

        self.num_con = 256
        self.num_vis = ent_vis_mask.shape[1]
        if self.score_function == "tucker":
            # self.tucker_decoder = TuckERLayer(dim_str, dim_str)
            self.tucker_decoder = TuckERLayerEA(dim_str, dim_str,
                                                use_ea_ent=True,  # 实体侧启用EA
                                                use_ea_rel=False,  # 关系只有单向量就关掉
                                                S_ent=64)
            self.tucker_decoder.set_masks(ent_mask=self.ent_mask, rel_mask=None)  # [B,N]，1=有效，0=padding
        else:
            pass

        self.init_weights()

        self.register_buffer('head_valid', torch.zeros(self.num_ent, self.num_rel, dtype=torch.bool))
        self.register_buffer('tail_valid', torch.zeros(self.num_ent, self.num_rel, dtype=torch.bool))

        self.bceloss = nn.BCEWithLogitsLoss()

        self.head_classifier_r = nn.Sequential(
            nn.Linear(dim_str, dim_str),
            AReLU(),
            # nn.ReLU(),
            nn.Linear(dim_str, self.num_ent)
        )
        self.tail_classifier_r = nn.Sequential(
            nn.Linear(dim_str, dim_str),
            AReLU(),
            nn.Linear(dim_str, self.num_ent)
        )

        self.cma = CrossModelAttNew(feature_dim_img=8)

    def prefill_valid(self, triplets, label):
        self.eval()
        with torch.no_grad():
            h = triplets[:, 0] - self.num_rel
            r = triplets[:, 1] - self.num_ent
            t = triplets[:, 2] - self.num_rel

            self.batch_r = r

            fill_triplets = torch.stack([h, r, t, label]).T
            for i in fill_triplets:
                if i[0] == self.num_ent:
                    self.head_valid[i[3], i[1]] = True
                    self.tail_valid[i[2], i[1]] = True
                else:
                    self.head_valid[i[0], i[1]] = True
                    self.tail_valid[i[3], i[1]] = True

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings)
        nn.init.xavier_uniform_(self.rel_embeddings)
        # nn.init.xavier_uniform_(self.proj_ent_vis.weight)
        # nn.init.xavier_uniform_(self.proj_ent_txt.weight) # 进行SE或CBAM消融后注释
        if hasattr(self.proj_ent_vis, "fc"):
            nn.init.xavier_uniform_(self.proj_ent_vis.fc.weight)
        else:
            nn.init.xavier_uniform_(self.proj_ent_vis.weight)
        nn.init.xavier_uniform_(self.ent_token)
        nn.init.xavier_uniform_(self.rel_token)
        nn.init.xavier_uniform_(self.lp_token)
        nn.init.xavier_uniform_(self.pos_str_ent)
        nn.init.xavier_uniform_(self.pos_vis_ent)
        nn.init.xavier_uniform_(self.pos_txt_ent)
        nn.init.xavier_uniform_(self.pos_str_rel)
        nn.init.xavier_uniform_(self.pos_vis_rel)
        nn.init.xavier_uniform_(self.pos_txt_rel)
        nn.init.xavier_uniform_(self.pos_head)
        nn.init.xavier_uniform_(self.pos_rel)
        nn.init.xavier_uniform_(self.pos_tail)

    def forward(self):

        ent_tkn = self.ent_token.tile(self.num_ent, 1, 1)  # (15000, 1, 256)
        rep_ent_str = self.embdr(
            self.str_ent_ln(self.proj_s(self.ent_embeddings))) + self.pos_str_ent  # (15000, 1, 256)

        entity_visual_tokens = self.visual_token_embedding(self.visual_token_index)  # (15000, 8, 32)
        entity_text_tokens = self.text_token_embedding(self.text_token_index)  # (15000, 8, 768)
        ent_vis = self.vis_ln(self.proj_ent_vis(entity_visual_tokens))  # (15000, 8, 256)
        ent_txt = self.txt_ln(self.proj_ent_txt(entity_text_tokens))  # (15000, 8, 256)
        vis_txt_fusion = self.cma(ent_vis, ent_txt)
        rep_ent_vis = self.visdr(ent_vis) + self.pos_vis_ent  # (15000, 8, 256)
        rep_ent_txt = self.txtdr(ent_txt) + self.pos_txt_ent  # (15000, 8, 256)

        ent_seq = torch.cat([ent_tkn, rep_ent_str, rep_ent_vis, rep_ent_txt, vis_txt_fusion], dim=1)  # (15000, 26, 256)
        ent_embs = self.ent_encoder(ent_seq, src_key_padding_mask=self.ent_mask)  # (15000, 18, 256)
        rep_rel_str = self.embdr(self.str_rel_ln(self.rel_embeddings))  # (169, 1, 256)

        ent_v = torch.mean(ent_embs[:, 2: 2 + self.num_vis, :], dim=1)  # (15000, 256)
        ent_t = torch.mean(ent_embs[:, 2 + self.num_vis:, ], dim=1)  # (15000, 256)

        targets_itc = torch.arange(0, self.num_ent).to(ent_t.device)
        temp = 0.5 # 温度系数
        sim_itc_tv = ent_v @ ent_t.t() / temp
        sim_itc_vt = ent_t @ ent_v.t() / temp
        itc_tv_loss = F.cross_entropy(sim_itc_tv, targets_itc)
        itc_vt_loss = F.cross_entropy(sim_itc_vt, targets_itc)
        itc_loss = 0.5 * (itc_tv_loss + itc_vt_loss)  # tensor(6.5487)


        return torch.cat([ent_embs[:, 0], self.lp_token], dim=0), rep_rel_str.squeeze(dim=1), itc_loss

    def contrastive_loss_relation(self, rel_embs, loss_flag=True):
        head_scores = self.head_classifier_r(rel_embs)
        tail_scores = self.tail_classifier_r(rel_embs)
        head_pro = torch.sigmoid(head_scores)
        tail_pro = torch.sigmoid(tail_scores)
        self.head_rel_pro = F.softmax(head_pro, dim=-1)
        self.tail_rel_pro = F.softmax(tail_pro, dim=-1)

        if loss_flag:
            head_loss = self.bceloss(head_scores, self.head_valid.T.float())
            tail_loss = self.bceloss(tail_scores, self.tail_valid.T.float())
            return 0.5 * (head_loss + tail_loss)
        else:
            return

    def score(self, emb_ent, emb_rel, triplets):
        h_seq = emb_ent[triplets[:, 0] - self.num_rel].unsqueeze(dim=1) + self.pos_head
        r_seq = emb_rel[triplets[:, 1] - self.num_ent].unsqueeze(dim=1) + self.pos_rel
        t_seq = emb_ent[triplets[:, 2] - self.num_rel].unsqueeze(dim=1) + self.pos_tail
        dec_seq = torch.cat([h_seq, r_seq, t_seq], dim=1)
        output_dec = self.decoder(dec_seq)
        rel_emb = output_dec[:, 1, :]
        ctx_emb = output_dec[triplets == self.num_ent + self.num_rel]

        if self.score_function == "tucker":
            tucker_emb = self.tucker_decoder(ctx_emb, rel_emb)
            scores = torch.mm(tucker_emb, emb_ent[:-1].transpose(1, 0))
            score = scores
        else:
            score = torch.inner(ctx_emb, emb_ent[:-1])
        return score

    def auto_reshape_tensor(self, x, target_H=None):
        """
        自动拆分第 2 维，尽量拆分成接近正方形 (H ≈ W)

        Args:
            x: 输入 Tensor (D0, D1, D2)
            target_H: 可选，指定目标 H，否则自动计算
        """
        D2 = x.size(-1)

        if target_H is None:
            # 自动计算最接近的 H（例如 256 → 16×16）
            H = int(math.sqrt(D2))
            while D2 % H != 0:
                H -= 1
            W = D2 // H
        else:
            assert D2 % target_H == 0, f"D2 ({D2}) 必须能被 target_H ({target_H}) 整除"
            H, W = target_H, D2 // target_H

        # 交换维度 + reshape
        new_shape = list(x.shape[:-1]) + [H, W]  # (D1, D0, H, W)
        return x.reshape(new_shape)

    def auto_discover_shape(self, x):
        x_new = x.reshape(
            x.shape[0],
            x.shape[1],
            x.shape[2] * x.shape[3]
        )
        return x_new


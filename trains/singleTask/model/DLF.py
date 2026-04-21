"""
Tri-subspaces disentanglement backbone for multimodal sentiment analysis.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import almt
from ...subNets.BertTextEncoder import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder


class DLF(nn.Module):
    def __init__(self, args):
        super(DLF, self).__init__()
        self.args = args
        self.use_bert = args.use_bert

        if self.use_bert:
            self.text_model = BertTextEncoder(
                use_finetune=args.use_finetune,
                transformers=args.transformers,
                pretrained=args.pretrained,
            )

        dst_feature_dims, nheads = 128, 8
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels

        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask

        # 1) Unimodal projection
        self.proj_l = nn.Conv1d(
            self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False
        )
        self.proj_a = nn.Conv1d(
            self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False
        )
        self.proj_v = nn.Conv1d(
            self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False
        )

        # 2) Modality encoders (private backbone features)
        self.encoder_l = self.get_network(self_type='l', layers=self.layers)
        self.encoder_a = self.get_network(self_type='a', layers=self.layers)
        self.encoder_v = self.get_network(self_type='v', layers=self.layers)

        # 3) Tri-subspaces heads: private, common, and pairwise-shared
        self.private_proj_l = nn.Linear(self.d_l, self.d_l)
        self.private_proj_a = nn.Linear(self.d_a, self.d_a)
        self.private_proj_v = nn.Linear(self.d_v, self.d_v)

        self.common_proj_l = nn.Linear(self.d_l, self.d_l)
        self.common_proj_a = nn.Linear(self.d_a, self.d_a)
        self.common_proj_v = nn.Linear(self.d_v, self.d_v)
        self.global_shared_proj = nn.Linear(self.d_l, self.d_l)

        self.pair_la_l = nn.Linear(self.d_l, self.d_l)
        self.pair_la_a = nn.Linear(self.d_a, self.d_l)
        self.pair_lv_l = nn.Linear(self.d_l, self.d_l)
        self.pair_lv_v = nn.Linear(self.d_v, self.d_l)
        self.pair_av_a = nn.Linear(self.d_a, self.d_l)
        self.pair_av_v = nn.Linear(self.d_v, self.d_l)

        self.pair_fuse_la = nn.Linear(self.d_l * 2, self.d_l)
        self.pair_fuse_lv = nn.Linear(self.d_l * 2, self.d_l)
        self.pair_fuse_av = nn.Linear(self.d_l * 2, self.d_l)

        # 4) Reliability estimator and semantic injection
        rel_hidden_dim = getattr(args, 'rel_hidden_dim', self.d_l)
        self.reliability_scorer = nn.Sequential(
            nn.Linear(self.d_l * 3, rel_hidden_dim),
            nn.ReLU(),
            nn.Linear(rel_hidden_dim, 3),
            nn.Sigmoid(),
        )

        self.inject_attn_l = nn.MultiheadAttention(self.d_l, self.num_heads, dropout=self.attn_dropout, batch_first=True)
        self.inject_attn_a = nn.MultiheadAttention(self.d_a, self.num_heads, dropout=self.attn_dropout_a, batch_first=True)
        self.inject_attn_v = nn.MultiheadAttention(self.d_v, self.num_heads, dropout=self.attn_dropout_v, batch_first=True)
        self.local_almt_l = almt.ALMT(args)
        self.local_almt_a = almt.ALMT(args)
        self.local_almt_v = almt.ALMT(args)

        self.inject_gate_l = nn.Sequential(nn.Linear(2, self.d_l), nn.Sigmoid())
        self.inject_gate_a = nn.Sequential(nn.Linear(2, self.d_a), nn.Sigmoid())
        self.inject_gate_v = nn.Sequential(nn.Linear(2, self.d_v), nn.Sigmoid())

        # 5) Routing and prediction heads
        router_in_dim = self.d_l * 3 + 3
        self.router = nn.Sequential(
            nn.Linear(router_in_dim, self.d_l),
            nn.ReLU(),
            nn.Linear(self.d_l, 3),
        )

        self.phi_private_l = nn.Linear(self.d_l, self.d_l)
        self.phi_private_a = nn.Linear(self.d_a, self.d_l)
        self.phi_private_v = nn.Linear(self.d_v, self.d_l)
        self.pair_proj = nn.Linear(self.d_l, self.d_l)
        self.global_proj = nn.Linear(self.d_l, self.d_l)
        self.branch_fusion_gate = nn.Sequential(
            nn.Linear(3, self.d_l // 2),
            nn.ReLU(),
            nn.Linear(self.d_l // 2, 2),
        )

        self.out_layer_s = nn.Linear(self.d_l, 1)
        self.out_layer_c = nn.Linear(self.d_l, 1)

        self.proj1 = nn.Linear(self.d_l * 2, self.d_l * 2)
        self.proj2 = nn.Linear(self.d_l * 2, self.d_l * 2)
        self.out_layer = nn.Linear(self.d_l * 2, 1)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=max(self.layers, layers),
            attn_dropout=attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask,
        )

    @staticmethod
    def _pool(seq_feat):
        return seq_feat.mean(dim=1)

    def _inject_private(self, p_m, pairwise_memory, r_m, r_other_mean, attn_layer, gate_layer, almt_module, pair_x, pair_y):
        context, _ = attn_layer(query=p_m, key=pairwise_memory, value=pairwise_memory)
        _, local_feat = almt_module(p_m, pair_x, pair_y)
        local_ctx = local_feat.unsqueeze(1).expand_as(context)
        context = context + local_ctx
        gate_input = torch.stack([r_m, r_other_mean], dim=-1)
        gate = gate_layer(gate_input).unsqueeze(1)
        return p_m + gate * context, gate

    def forward(self, text, audio, video):
        if self.use_bert:
            text = self.text_model(text)

        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)

        h_l = self.proj_l(x_l).permute(0, 2, 1)
        h_a = self.proj_a(x_a).permute(0, 2, 1)
        h_v = self.proj_v(x_v).permute(0, 2, 1)


        min_t = min(h_l.size(1), h_a.size(1), h_v.size(1))
        h_l, h_a, h_v = h_l[:, :min_t, :], h_a[:, :min_t, :], h_v[:, :min_t, :]

        # Tri-subspaces decomposition
        p_l = self.private_proj_l(h_l)
        p_a = self.private_proj_a(h_a)
        p_v = self.private_proj_v(h_v)

        c_l = self.common_proj_l(h_l)
        c_a = self.common_proj_a(h_a)
        c_v = self.common_proj_v(h_v)
        s_g = self.global_shared_proj((c_l + c_a + c_v) / 3.0)

        q_la_l = self.pair_la_l(h_l)
        q_la_a = self.pair_la_a(h_a)
        q_lv_l = self.pair_lv_l(h_l)
        q_lv_v = self.pair_lv_v(h_v)
        q_av_a = self.pair_av_a(h_a)
        q_av_v = self.pair_av_v(h_v)

        s_la = self.pair_fuse_la(torch.cat([q_la_l, q_la_a], dim=-1))
        s_lv = self.pair_fuse_lv(torch.cat([q_lv_l, q_lv_v], dim=-1))
        s_av = self.pair_fuse_av(torch.cat([q_av_a, q_av_v], dim=-1))

        pooled_l = self._pool(c_l)
        pooled_a = self._pool(c_a)
        pooled_v = self._pool(c_v)
        reliability = self.reliability_scorer(torch.cat([pooled_l, pooled_a, pooled_v], dim=-1))

        r_l, r_a, r_v = reliability[:, 0], reliability[:, 1], reliability[:, 2]
        r_other_l = (r_a + r_v) / 2.0
        r_other_a = (r_l + r_v) / 2.0
        r_other_v = (r_l + r_a) / 2.0

        mem_l = torch.cat([s_la, s_lv], dim=1)
        mem_a = torch.cat([s_la, s_av], dim=1)
        mem_v = torch.cat([s_lv, s_av], dim=1)

        p_tilde_l, g_l = self._inject_private(
            p_l, mem_l, r_l, r_other_l, self.inject_attn_l, self.inject_gate_l, self.local_almt_l, s_la, s_lv
        )
        p_tilde_a, g_a = self._inject_private(
            p_a, mem_a, r_a, r_other_a, self.inject_attn_a, self.inject_gate_a, self.local_almt_a, s_la, s_av
        )
        p_tilde_v, g_v = self._inject_private(
            p_v, mem_v, r_v, r_other_v, self.inject_attn_v, self.inject_gate_v, self.local_almt_v, s_lv, s_av
        )

        feat_pl = self._pool(p_tilde_l)
        feat_pa = self._pool(p_tilde_a)
        feat_pv = self._pool(p_tilde_v)
        feat_sg = self._pool(s_g)

        routed_feats = [
            self.phi_private_l(feat_pl),
            self.phi_private_a(feat_pa),
            self.phi_private_v(feat_pv),
        ]

        router_input = torch.cat([feat_pl, feat_pa, feat_pv, reliability], dim=-1)
        router_logits = self.router(router_input)
        router_weights = F.softmax(router_logits, dim=-1)

        fused = 0.0
        for i, feat_i in enumerate(routed_feats):
            fused = fused + router_weights[:, i:i + 1] * feat_i

        pair_repr = self.pair_proj(fused)
        global_repr = self.global_proj(feat_sg)
        branch_weights = F.softmax(self.branch_fusion_gate(reliability), dim=-1)
        pair_repr = pair_repr * branch_weights[:, 0:1]
        global_repr = global_repr * branch_weights[:, 1:2]
        final_repr = torch.cat([pair_repr, global_repr], dim=-1)

        fused_proj = self.proj2(
            F.dropout(F.relu(self.proj1(final_repr), inplace=True), p=self.output_dropout, training=self.training)
        )
        fused_proj = fused_proj + final_repr

        output = self.out_layer(fused_proj)
        logits_s = self.out_layer_s(feat_sg)
        logits_c = self.out_layer_c(pair_repr)

        return {
            'output_logit': output,
            'logits_s': logits_s,
            'logits_c': logits_c,
            'p_l': p_l,
            'p_a': p_a,
            'p_v': p_v,
            'p_tilde_l': p_tilde_l,
            'p_tilde_a': p_tilde_a,
            'p_tilde_v': p_tilde_v,
            'c_l': c_l,
            'c_a': c_a,
            'c_v': c_v,
            's_g': s_g,
            'q_la_l': q_la_l,
            'q_la_a': q_la_a,
            'q_lv_l': q_lv_l,
            'q_lv_v': q_lv_v,
            'q_av_a': q_av_a,
            'q_av_v': q_av_v,
            's_la': s_la,
            's_lv': s_lv,
            's_av': s_av,
            'reliability': reliability,
            'router_weights': router_weights,
            'inject_gate_l': g_l,
            'inject_gate_a': g_a,
            'inject_gate_v': g_v,
            'pair_repr': pair_repr,
            'global_repr': global_repr,
            'branch_weights': branch_weights,
        }

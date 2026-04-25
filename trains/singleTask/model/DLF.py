"""
Tri-subspaces disentanglement backbone for multimodal sentiment analysis.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...subNets.BertTextEncoder import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder


class TwoLayerSubspaceEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.net(x)


class SelectiveGateFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, private_feat, shared_feat):
        gate = self.gate(torch.cat([private_feat, shared_feat], dim=-1))
        fused = gate * private_feat + (1.0 - gate) * shared_feat
        return fused, gate


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

        self.proj_l = nn.Conv1d(
            self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False
        )
        self.proj_a = nn.Conv1d(
            self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False
        )
        self.proj_v = nn.Conv1d(
            self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False
        )

        self.encoder_l = self.get_network(self_type='l', layers=self.layers)
        self.encoder_a = self.get_network(self_type='a', layers=self.layers)
        self.encoder_v = self.get_network(self_type='v', layers=self.layers)

        self.private_encoder_l = TwoLayerSubspaceEncoder(self.d_l, self.d_l)
        self.private_encoder_a = TwoLayerSubspaceEncoder(self.d_a, self.d_a)
        self.private_encoder_v = TwoLayerSubspaceEncoder(self.d_v, self.d_v)

        self.common_encoder = TwoLayerSubspaceEncoder(self.d_l, self.d_l)
        self.la_encoder = TwoLayerSubspaceEncoder(self.d_l, self.d_l)
        self.lv_encoder = TwoLayerSubspaceEncoder(self.d_l, self.d_l)
        self.av_encoder = TwoLayerSubspaceEncoder(self.d_l, self.d_l)

        self.la_fuser = TwoLayerSubspaceEncoder(self.d_l, self.d_l)
        self.lv_fuser = TwoLayerSubspaceEncoder(self.d_l, self.d_l)
        self.av_fuser = TwoLayerSubspaceEncoder(self.d_l, self.d_l)

        self.gate_l = SelectiveGateFusion(self.d_l)
        self.gate_a = SelectiveGateFusion(self.d_a)
        self.gate_v = SelectiveGateFusion(self.d_v)

        self.la_attention = nn.MultiheadAttention(
            self.d_l, self.num_heads, dropout=self.attn_dropout_a, batch_first=True
        )
        self.lv_attention = nn.MultiheadAttention(
            self.d_l, self.num_heads, dropout=self.attn_dropout_v, batch_first=True
        )

        self.cross_pair_fusion = TwoLayerSubspaceEncoder(self.d_l * 2, self.d_l)
        self.final_fusion = TwoLayerSubspaceEncoder(self.d_l * 2, self.d_l)

        self.common_head = nn.Linear(self.d_l, 1)
        self.cross_head = nn.Linear(self.d_l, 1)
        self.out_layer = nn.Sequential(
            nn.Linear(self.d_l, self.d_l),
            nn.GELU(),
            nn.Dropout(self.output_dropout),
            nn.Linear(self.d_l, 1),
        )

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

    @staticmethod
    def _expand_to_length(feat, target_len):
        return feat.unsqueeze(1).expand(-1, target_len, -1)

    def _encode_backbone(self, feat, encoder):
        return encoder(feat.transpose(0, 1)).transpose(0, 1)

    def forward(self, text, audio, video):
        if self.use_bert:
            text = self.text_model(text)

        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)

        h_l = self.proj_l(x_l).permute(0, 2, 1)
        h_a = self.proj_a(x_a).permute(0, 2, 1)
        h_v = self.proj_v(x_v).permute(0, 2, 1)

        h_l = self._encode_backbone(h_l, self.encoder_l)
        h_a = self._encode_backbone(h_a, self.encoder_a)
        h_v = self._encode_backbone(h_v, self.encoder_v)

        p_l = self.private_encoder_l(h_l)
        p_a = self.private_encoder_a(h_a)
        p_v = self.private_encoder_v(h_v)

        c_l = self.common_encoder(h_l)
        c_a = self.common_encoder(h_a)
        c_v = self.common_encoder(h_v)
        common_repr = (self._pool(c_l) + self._pool(c_a) + self._pool(c_v)) / 3.0
        s_g_l = self._expand_to_length(common_repr, h_l.size(1))
        s_g_a = self._expand_to_length(common_repr, h_a.size(1))
        s_g_v = self._expand_to_length(common_repr, h_v.size(1))
        s_g = common_repr

        q_la_l = self.la_encoder(h_l)
        q_la_a = self.la_encoder(h_a)
        q_lv_l = self.lv_encoder(h_l)
        q_lv_v = self.lv_encoder(h_v)
        q_av_a = self.av_encoder(h_a)
        q_av_v = self.av_encoder(h_v)

        s_la_vec = self.la_fuser((self._pool(q_la_l) + self._pool(q_la_a)) / 2.0)
        s_lv_vec = self.lv_fuser((self._pool(q_lv_l) + self._pool(q_lv_v)) / 2.0)
        s_av_vec = self.av_fuser((self._pool(q_av_a) + self._pool(q_av_v)) / 2.0)

        s_la_l = self._expand_to_length(s_la_vec, h_l.size(1))
        s_la_a = self._expand_to_length(s_la_vec, h_a.size(1))
        s_lv_l = self._expand_to_length(s_lv_vec, h_l.size(1))
        s_lv_v = self._expand_to_length(s_lv_vec, h_v.size(1))
        s_av_a = self._expand_to_length(s_av_vec, h_a.size(1))
        s_av_v = self._expand_to_length(s_av_vec, h_v.size(1))

        s_la = s_la_vec
        s_lv = s_lv_vec
        s_av = s_av_vec

        shared_l = (s_la_l + s_lv_l) / 2.0
        shared_a = (s_la_a + s_av_a) / 2.0
        shared_v = (s_lv_v + s_av_v) / 2.0

        fused_l, gate_l = self.gate_l(p_l, shared_l)
        fused_a, gate_a = self.gate_a(p_a, shared_a)
        fused_v, gate_v = self.gate_v(p_v, shared_v)

        la_ctx, la_attn = self.la_attention(query=fused_l, key=fused_a, value=fused_a)
        lv_ctx, lv_attn = self.lv_attention(query=fused_l, key=fused_v, value=fused_v)

        cross_seq = self.cross_pair_fusion(torch.cat([la_ctx, lv_ctx], dim=-1))
        cross_repr = self._pool(cross_seq)
        final_repr = self.final_fusion(torch.cat([cross_repr, common_repr], dim=-1))

        output = self.out_layer(final_repr)
        logits_s = self.common_head(common_repr)
        logits_c = self.cross_head(cross_repr)

        return {
            'output_logit': output,
            'logits_s': logits_s,
            'logits_c': logits_c,
            'h_l': h_l,
            'h_a': h_a,
            'h_v': h_v,
            'p_l': p_l,
            'p_a': p_a,
            'p_v': p_v,
            'c_l': c_l,
            'c_a': c_a,
            'c_v': c_v,
            's_g': s_g,
            's_g_l': s_g_l,
            's_g_a': s_g_a,
            's_g_v': s_g_v,
            'q_la_l': q_la_l,
            'q_la_a': q_la_a,
            'q_lv_l': q_lv_l,
            'q_lv_v': q_lv_v,
            'q_av_a': q_av_a,
            'q_av_v': q_av_v,
            's_la': s_la,
            's_lv': s_lv,
            's_av': s_av,
            's_la_l': s_la_l,
            's_la_a': s_la_a,
            's_lv_l': s_lv_l,
            's_lv_v': s_lv_v,
            's_av_a': s_av_a,
            's_av_v': s_av_v,
            'shared_l': shared_l,
            'shared_a': shared_a,
            'shared_v': shared_v,
            'fused_l': fused_l,
            'fused_a': fused_a,
            'fused_v': fused_v,
            'gate_l': gate_l,
            'gate_a': gate_a,
            'gate_v': gate_v,
            'la_ctx': la_ctx,
            'lv_ctx': lv_ctx,
            'la_attn': la_attn,
            'lv_attn': lv_attn,
            'cross_seq': cross_seq,
            'cross_repr': cross_repr,
            'common_repr': common_repr,
            'final_repr': final_repr,
        }

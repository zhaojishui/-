import torch
from torch import nn
from almt_layer import Transformer, CrossTransformer, HhyperLearningEncoder

from einops import repeat


class ALMT(nn.Module):
    def __init__(self, args):
        super(ALMT, self).__init__()
        self.h_hyper = nn.Parameter(torch.ones(1, args.token_len, args.token_dim))#1，8，128
        self.l_encoder = Transformer(save_hidden=True, dim=args.proj_input_dim, depth=args.AHL_depth-1, heads=args.l_enc_heads, mlp_dim=args.l_enc_mlp_dim)
        self.h_hyper_layer = HhyperLearningEncoder(dim=args.token_dim, depth=args.AHL_depth, heads=args.ahl_heads, dim_head=args.ahl_dim_head, dropout=args.ahl_droup)
        self.fusion_layer = CrossTransformer(source_num_frames=args.token_len, tgt_num_frames=args.token_len, dim=args.proj_input_dim, depth=args.fusion_layer_depth, heads=args.fusion_heads, mlp_dim=args.fusion_mlp_dim)
        self.regression_layer = nn.Sequential(
            nn.Linear(args.token_dim, 1)#输入128个维度，输出1个预测结果
        )
    def forward(self, x_text, x_visual, x_audio):
        b = x_visual.size(0)#batchsize
        h_hyper = repeat(self.h_hyper, '1 n d -> b n d', b=b)#batchsize，8，128
        h_v = x_visual[:, :self.h_hyper.shape[1]]#proj_v(x_visual)形状：batchsize，序列长度，128维，然后只要前8个序列长度
        h_a = x_audio[:, :self.h_hyper.shape[1]]#audio从5维转成128维，然后只取前8个序列长度
        h_l = x_text[:, :self.h_hyper.shape[1]]#768到128，然后只取前8个序列长度

        h_t_list = self.l_encoder(h_l)
        h_hyper = self.h_hyper_layer(h_t_list, h_a, h_v, h_hyper)
        feat = self.fusion_layer(h_hyper, h_t_list[-1])[:, 0]

        output = self.regression_layer(feat)

        return output,feat


def build_model(args):
    model = ALMT(args)

    return model




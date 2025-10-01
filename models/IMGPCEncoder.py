import torch
import torch.nn as nn
import torch_scatter
import numpy as np
import math
import time
import sys
import scipy.io as scio
from .PointViT import PointTransformer
from .ImageViT import ImageTransformer
sys.path.append("..")


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.num_head
        self.attention_head_size = int(config.embed_dim / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.embed_dim, self.all_head_size)
        self.key = nn.Linear(config.embed_dim, self.all_head_size)
        self.value = nn.Linear(config.embed_dim, self.all_head_size)

        self.out = nn.Linear(config.embed_dim, config.embed_dim)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.proj_dropout = nn.Dropout(config.attention_dropout)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x_hidden_states, y_hidden_states):
        mixed_query_layer = self.query(x_hidden_states)
        mixed_key_layer = self.key(y_hidden_states)
        mixed_value_layer = self.value(y_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config.embed_dim, config.mlp_dim)
        self.fc2 = nn.Linear(config.mlp_dim, config.embed_dim)
        self.act_fn = nn.functional.gelu
        self.dropout = nn.Dropout(config.mlp_dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.attention_norm = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x, y):
        h = x
        x = self.attention_norm(x)
        y = self.attention_norm(y)
        x = self.attn(x, y)
        x = h + x

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = h + x
        return x


class IMGPCEncoder(nn.Module):
    def __init__(self, config):
        super(IMGPCEncoder, self).__init__()
        self.config = config

        self.pt_transformer = PointTransformer(config)
        self.img_transformer = ImageTransformer(config)

        self.i2p_ca_layers = nn.ModuleList()
        for _ in range(config.num_ca_layer_coarse):
            self.i2p_ca_layers.append(Block(config))

        self.p2i_ca_layers = nn.ModuleList()
        for _ in range(config.num_ca_layer_coarse):
            self.p2i_ca_layers.append(Block(config))

        self.pt_sa_layers = nn.ModuleList()
        for _ in range(config.num_ca_layer_coarse):
            self.pt_sa_layers.append(Block(config))

        self.img_sa_layers = nn.ModuleList()
        for _ in range(config.num_ca_layer_coarse):
            self.img_sa_layers.append(Block(config))

    def forward(self, data_batch):
        img = data_batch['img'].cuda()
        pc = data_batch['pc'].cuda()
        data_batch['pc_i'] = pc
        node = data_batch['node'].cuda()
        idx = data_batch['pt2node'].cuda()

        img_proxy, img_feat_2, img_feat_1, img_feat_0 = self.img_transformer(img)
        data_batch['img_feat_2'] = img_feat_2
        data_batch['img_feat_1'] = img_feat_1
        data_batch['img_feat_0'] = img_feat_0

        # <------this process will sample proxies from nodes again------>
        pt_proxy, node_proxy_idx, pt_feat, node_feat = self.pt_transformer(pc, node, idx)
        data_batch['node2proxy'] = node_proxy_idx[:,:,0]
        data_batch['pt_feat'] = pt_feat
        data_batch['node_feat'] = node_feat

        for i in range(self.config.num_ca_layer_coarse):
            # <------cross-attention from point to image------>
            layer = self.p2i_ca_layers[i]
            img_proxy = layer(img_proxy, pt_proxy)
            # <------cross-attention from image to point------>
            layer = self.i2p_ca_layers[i]
            pt_proxy = layer(pt_proxy, img_proxy)
            # <------self-attention------>
            layer = self.img_sa_layers[i]
            img_proxy = layer(img_proxy, img_proxy)
            layer = self.pt_sa_layers[i]
            pt_proxy = layer(pt_proxy, pt_proxy)

        data_batch['img_proxy'] = img_proxy
        data_batch['pt_proxy'] = pt_proxy
        data_batch['pc'] = pc

        return 0

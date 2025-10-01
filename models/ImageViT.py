import torch
import torch.nn as nn
import numpy as np
import math
from .ImageResNet import MiniResNet


class Embeddings(nn.Module):
    """
    Construct the embeddings from patch, positional embeddings.
    """
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.config = config
        self.embedding_layers = nn.ModuleList()

        self.mini_resnet = MiniResNet(inchannel=3, outchannel=config.embed_dim)
        self.embedding_layers.append(self.mini_resnet)
        self.patch_embeddings = nn.Conv2d(in_channels=config.embed_dim,
                                          out_channels=config.embed_dim,
                                          kernel_size=config.patch_size,
                                          stride=config.patch_size)
        self.embedding_layers.append(self.patch_embeddings)

        self.num_patches = (config.image_H // config.patch_size) * (config.image_W // config.patch_size)
        self.position_embeddings = nn.Parameter(data=self.get_sinusoid_embedding(n_position=self.num_patches, \
                                                d_hid=config.embed_dim), requires_grad=False)

        self.dropout = nn.Dropout(config.embed_dropout)

    def _get_position_angle_vec(self, position, d_hid):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    def get_sinusoid_embedding(self, n_position, d_hid):
        sinusoid_table = np.array([self._get_position_angle_vec(pos_i, d_hid) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        """
        Input:
            x: source images, [B, 3, H, W]
        Output:
            embeddings: patch embedding add positional embedding
        """
        # pyramid extractor
        img_feat_2, img_feat_1, img_feat_0 = self.embedding_layers[0](x)

        # patch partition
        x = self.embedding_layers[1](img_feat_2)

        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = x + self.position_embeddings
        embeddings = self.dropout(x)

        return embeddings, img_feat_2, img_feat_1, img_feat_0


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

    def forward(self, hidden_states):
        """
        Input:
            hidden_states: embeddings, [B, C, Ne]
        Output:
            attention_output: embeddings after self-attention, [B, C, Ne]
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

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

    def forward(self, x):
        """
        Input:
            x: image patch embedding, [B, Ne, Ce]
        """
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class ImageTransformer(nn.Module):
    def __init__(self, config):
        super(ImageTransformer, self).__init__()
        self.config = config
        self.embeddings = Embeddings(config)

        self.sa_encoder_layers = nn.ModuleList()
        for _ in range(config.num_sa_layer):
            self.sa_encoder_layers.append(Block(config))

    def forward(self, x):
        """
        Input:
            x: source images, [B, 3, H, W]
        """
        proxy_feat, img_feat_2, img_feat_1, img_feat_0 = self.embeddings(x)

        for layer_block in self.sa_encoder_layers:
            proxy_feat = layer_block(proxy_feat)

        return proxy_feat, img_feat_2, img_feat_1, img_feat_0

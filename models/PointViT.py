import torch
import torch.nn as nn
import math
from .PointNN import MiniGNN, GroupPointTransformer, MiniPointNet, KnnPointTransformer
import torch_scatter


class Embeddings(nn.Module):
    """
    Construct the embeddings from local geometry, positional embeddings.
    """
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.config = config

        if config.use_gnn_embedding:
            self.mini_gnn = MiniGNN(config.point_feat_dim, config.edge_conv_dim, config.embed_dim)
            self.pos_embed = nn.Sequential(
                nn.Conv1d(3, 128, 1),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv1d(128, config.embed_dim, 1)
            )
        else:
            self.raw_point_mlp = MiniPointNet(config.point_feat_dim, config.embed_dim)

            self.group_transformer_0 = GroupPointTransformer(config.embed_dim, config.embed_dim)
            self.point_mlp_0 = MiniPointNet(2 * config.embed_dim, config.embed_dim)

            self.group_transformer_1 = GroupPointTransformer(config.embed_dim, config.embed_dim)
            self.point_mlp_1 = MiniPointNet(2 * config.embed_dim, config.embed_dim)

            self.group_transformer_node = GroupPointTransformer(config.embed_dim, config.embed_dim)

            self.knn_transformers = nn.ModuleList()
            self.knn_transformers.append(KnnPointTransformer(config.embed_dim, config.embed_dim, k=16))
            self.knn_transformers.append(KnnPointTransformer(config.embed_dim, config.embed_dim, k=16))
            self.knn_transformers.append(KnnPointTransformer(config.embed_dim, config.embed_dim, k=16))

            self.group_transformer_proxy = GroupPointTransformer(config.embed_dim, config.embed_dim)

    def forward(self, x, node, idx):
        """
        Input:
            x: source points, [B, 3 or (3+3+1), N]
            node: node points, [B, 3 or (3+3+1), M]
            idx: index from x -> node, [B, N]
        Output:
            embeddings: point embedding add positional embedding, [B, N, f]
        """
        if self.config.use_gnn_embedding:
            coor = x[:, 0:3, :]
            feat = self.mini_gnn(x, idx)
            pos = self.pos_embed(coor)
            embeddings = feat + pos
            # <------pick the gragh feature of some proxies------>
            pass
        else:
            f = self.config.embed_dim
            b, n = x.shape[0], x.shape[2]
            x_feat = self.raw_point_mlp(x)
            node_feat = self.raw_point_mlp(node)

            node_feat = self.group_transformer_0(x, x_feat, node, node_feat, idx)
            # node_feat, _ = torch_scatter.scatter_max(x_feat, idx.unsqueeze(1).expand(b, f, n), dim=2)
            back_pt_feat = torch.gather(node_feat, index=idx.unsqueeze(1).expand(b, f, n), dim=2)
            cat_pt_feat = torch.cat((x_feat, back_pt_feat),dim=1)
            x_feat = self.point_mlp_0(cat_pt_feat)

            node_feat = self.group_transformer_1(x, x_feat, node, node_feat, idx)
            # node_feat, _ = torch_scatter.scatter_max(x_feat, idx.unsqueeze(1).expand(b, f, n), dim=2)
            back_pt_feat = torch.gather(node_feat, index=idx.unsqueeze(1).expand(b, f, n), dim=2)
            cat_pt_feat = torch.cat((x_feat, back_pt_feat), dim=1)
            x_feat = self.point_mlp_1(cat_pt_feat)

            node_feat = self.group_transformer_node(x, x_feat, node, node_feat, idx)
            # node_feat, _ = torch_scatter.scatter_max(x_feat, idx.unsqueeze(1).expand(b, f, n), dim=2)

            for layer in self.knn_transformers:
                node_feat = layer(node, node_feat)

            # <------proxy is sampled by FPS (Farthest point sampling)------>
            proxy = node[:, :, :self.config.num_proxy]
            proxy_feat = node_feat[:, :, :self.config.num_proxy]
            with torch.no_grad():
                dist = torch.norm(node.unsqueeze(3) - proxy.unsqueeze(2), p=2, dim=1, keepdim=False)
                _, node_proxy_idx = torch.topk(dist, k=1, dim=2, largest=False, sorted=True)

            embeddings = self.group_transformer_proxy(node, node_feat, proxy, proxy_feat, node_proxy_idx[:,:,0])
            # embeddings, _ = torch_scatter.scatter_max(node_feat, node_proxy_idx[:,:,0].unsqueeze(1).expand(b, f, 1280), dim=2)

            embeddings = embeddings.permute(0, 2, 1)
        return embeddings, node_proxy_idx, x_feat, node_feat


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
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class PointTransformer(nn.Module):
    def __init__(self, config):
        super(PointTransformer, self).__init__()
        self.config = config
        self.embeddings = Embeddings(config)

        self.sa_encoder_layers = nn.ModuleList()
        for _ in range(config.num_sa_layer):
            self.sa_encoder_layers.append(Block(config))

    def forward(self, pc, node, idx):
        """
        Input:
            x: source points, [B, 3 or (3+3+1), N]
            node: knn idx, [B, N, K]
        """
        proxy_feat, node_proxy_idx, x_feat, node_feat = self.embeddings(pc, node, idx)
        for layer_block in self.sa_encoder_layers:
            proxy_feat = layer_block(proxy_feat)
        return proxy_feat, node_proxy_idx, x_feat, node_feat

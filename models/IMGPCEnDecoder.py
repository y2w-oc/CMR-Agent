import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import sys
import cv2
import scipy.io as scio
import torch_scatter
from .IMGPCEncoder import IMGPCEncoder
from .PointNN import ConvBNReLURes1D
from .ImageResNet import ResidualBlock
sys.path.append("..")
from .LinearAttention import LinearAttention
from utils import PositionEncodingSine2D


class IMGPCEnDecoder(nn.Module):
    def __init__(self, config):
        super(IMGPCEnDecoder, self).__init__()
        self.config = config
        self.encoder = IMGPCEncoder(config)

        self.H_proxy = self.config.image_H // self.config.patch_size
        self.W_proxy = self.config.image_W // self.config.patch_size
        self.img_proxy_num = self.H_proxy * self.W_proxy
        self.pt_sample_num = config.pt_sample_num
        f = config.embed_dim

        # <------ fine features extraction with linear-attention ------>
        self.upsample_img_proxy = torch.nn.Upsample(scale_factor=self.config.patch_size, mode='nearest')

        self.node_fuse_convs = nn.ModuleList()
        self.node_fuse_convs.append(ConvBNReLURes1D(2 * f, f))
        for _ in range(config.node_fuse_res_num - 1):
            self.node_fuse_convs.append(ConvBNReLURes1D(f, f))
        self.node_fuse_convs.append(nn.Dropout(0.1))

        self.node_self_LA = nn.ModuleList()
        self.pixel_to_node_LA = nn.ModuleList()
        self.node_to_pixel_LA = nn.ModuleList()
        self.pixel_self_LA = nn.ModuleList()
        for i in range(config.linear_attention_num):
            self.node_self_LA.append(LinearAttention(d_model=f, nhead=config.LA_head_num))
            self.pixel_to_node_LA.append(LinearAttention(d_model=f, nhead=config.LA_head_num))
            self.node_to_pixel_LA.append(LinearAttention(d_model=f, nhead=config.LA_head_num))
            self.pixel_self_LA.append(LinearAttention(d_model=f, nhead=config.LA_head_num))

        self.img_fuse_convs = nn.ModuleList()
        self.img_fuse_convs.append(ResidualBlock(2 * f, f))
        for _ in range(config.img_fuse_res_num - 1):
            self.img_fuse_convs.append(ResidualBlock(f, f))
        self.img_fuse_convs.append(nn.Dropout(0.1))

        self.pixel_pos_encoding = PositionEncodingSine2D(f, (40, 128))


    def forward(self, data_batch):
        self.encoder(data_batch)

        img_feat_2 = data_batch['img_feat_2']
        # pt_feat = data_batch['pt_feat']
        node_feat = data_batch['node_feat']


        img_proxy = data_batch['img_proxy']
        img_proxy = img_proxy.permute(0, 2, 1)
        pt_proxy = data_batch['pt_proxy']
        pt_proxy = pt_proxy.permute(0, 2, 1)

        node2proxy = data_batch['node2proxy']

        # <------fuse the proxy features with point features via grouping------>
        f = pt_proxy.shape[1]
        b, n = node2proxy.shape[0], node2proxy.shape[1]
        scattered_node_proxy_feat = torch.gather(pt_proxy, index=node2proxy.unsqueeze(1).expand(b, f, n), dim=2)
        fused_node_feat = torch.cat([node_feat, scattered_node_proxy_feat], dim=1)  # [b, 2f, 1280]

        for layer in self.node_fuse_convs:
            fused_node_feat = layer(fused_node_feat)

        # <------fuse the proxy features with pixel features via nearest upsampling------>
        f = img_proxy.shape[1]
        img_proxy_4d = img_proxy.reshape(b, f, self.H_proxy, self.W_proxy)
        scattered_img_proxy_feat = self.upsample_img_proxy(img_proxy_4d)
        # scattered_img_proxy_feat = scattered_img_proxy_feat + self.img_position_embeddings
        fused_img_feat = torch.cat([img_feat_2, scattered_img_proxy_feat], dim=1)
        i = 0
        for layer in self.img_fuse_convs:
            fused_img_feat = layer(fused_img_feat)
            if i == 0:
                fused_img_feat = self.pixel_pos_encoding(fused_img_feat)
                i = i + 1

        data_batch['vis_feat'] = fused_img_feat
        fused_img_feat = fused_img_feat.view(b, f, -1)
        fused_img_feat = fused_img_feat.permute(0, 2, 1)
        fused_node_feat = fused_node_feat.permute(0, 2, 1)

        # <------ linear attention between fused_node_feat and fused_img_feat(pixel-level) ------>
        for i in range(self.config.linear_attention_num):
            layer = self.pixel_to_node_LA[i]
            fused_node_feat = layer(fused_node_feat, fused_img_feat)
            layer = self.node_to_pixel_LA[i]
            fused_img_feat = layer(fused_img_feat, fused_node_feat)
            layer = self.node_self_LA[i]
            fused_node_feat = layer(fused_node_feat, fused_node_feat)
            layer = self.pixel_self_LA[i]
            fused_img_feat = layer(fused_img_feat, fused_img_feat)

        fused_img_feat = fused_img_feat.permute(0, 2, 1)
        fused_img_feat = fused_img_feat.view(b, f, self.config.image_H, self.config.image_W)
        fused_node_feat = fused_node_feat.permute(0, 2, 1)

        data_batch['fused_img_feat'] = fused_img_feat
        data_batch['fused_node_feat'] = fused_node_feat

        return 0

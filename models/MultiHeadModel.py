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
from .IMGPCEnDecoder import IMGPCEnDecoder
from .PointNN import ConvBNReLURes1D
from .ImageResNet import ResidualBlock
from .PointViT import PointTransformer
sys.path.append("..")
from utils import PositionEncodingSine2D, Lovasz_loss
from .LinearAttention import LinearAttention
from .focal_loss import FocalLoss

import time
import scipy.io as scio


class OverlapDetectionHead(nn.Module):
    def __init__(self,config):
        super(OverlapDetectionHead, self).__init__()
        self.config = config
        f = config.embed_dim

        self.point_fuse_convs = nn.ModuleList()
        self.point_fuse_convs.append(ConvBNReLURes1D(2 * f, f))
        for _ in range(config.pt_head_res_num - 1):
            self.point_fuse_convs.append(ConvBNReLURes1D(f, f))
        self.pc_overlap_head = nn.Sequential(
                nn.Conv1d(f, 32, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv1d(32, 2, kernel_size=1, stride=1, padding=0)
            )

        self.img_res_convs = nn.ModuleList()
        for _ in range(config.img_fuse_res_num):
            self.img_res_convs.append(ResidualBlock(f, f))
        self.img_overlap_head = nn.Sequential(
            nn.Conv2d(f, 32, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 2, 1, 1, 0),
        )

        self.pc_overlap_criteria = FocalLoss(alpha=0.75, gamma=2, reduction='mean')
        self.img_overlap_criteria = FocalLoss(alpha=0.5, gamma=2, reduction='mean')

    def forward(self, data_batch):
        # <------ point cloud overlap ------>
        fused_node_feat = data_batch['fused_node_feat']
        device = fused_node_feat.device
        pt2node = data_batch['pt2node'].to(device)
        pt_feat = data_batch['pt_feat']

        f = fused_node_feat.shape[1]
        b, n = pt2node.shape[0], pt2node.shape[1]
        scattered_pt_node_feat = torch.gather(fused_node_feat, index=pt2node.unsqueeze(1).expand(b, f, n), dim=2)
        fused_pt_feat = torch.cat([pt_feat, scattered_pt_node_feat], dim=1)

        for layer in self.point_fuse_convs:
            fused_pt_feat = layer(fused_pt_feat)

        pc_overlap_logits = self.pc_overlap_head(fused_pt_feat)
        pc_overlap_label = data_batch['pc_mask'].cuda()
        # print(pc_overlap_logits.shape, pc_overlap_label.shape)
        pc_overlap_loss = self.pc_overlap_criteria(pc_overlap_logits, pc_overlap_label)

        # <------ image overlap ------>
        fused_img_feat = data_batch['fused_img_feat']
        for layer in self.img_res_convs:
            fused_img_feat = layer(fused_img_feat)
        img_overlap_logits = self.img_overlap_head(fused_img_feat)
        img_overlap_logits = img_overlap_logits.view(b, 2, -1)
        img_overlap_label = data_batch['img_mask'].cuda()
        img_overlap_label = img_overlap_label.view(b, -1)
        img_overlap_loss = self.img_overlap_criteria(img_overlap_logits, img_overlap_label)

        data_batch['pc_overlap_logits'] = pc_overlap_logits
        data_batch['img_overlap_logits'] = img_overlap_logits

        prediction = pc_overlap_logits.argmax(1)
        pc_overlap_precision = (pc_overlap_label[prediction == 1]).sum() / prediction.sum()
        pc_overlap_recall = (prediction[pc_overlap_label == 1]).sum() / pc_overlap_label.sum()
        pc_overlap_accuracy = (prediction == pc_overlap_label).sum() / b / n

        prediction = img_overlap_logits.argmax(1)
        b, n = prediction.shape[0], prediction.shape[1]
        img_overlap_precision = (img_overlap_label[prediction == 1]).sum() / prediction.sum()
        img_overlap_recall = (prediction[img_overlap_label == 1]).sum() / img_overlap_label.sum()
        img_overlap_accuracy = (prediction == img_overlap_label).sum() / b / n

        data_batch['pc_overlap_loss'] = pc_overlap_loss
        data_batch['img_overlap_loss'] = img_overlap_loss
        data_batch['loss'] += pc_overlap_loss
        data_batch['loss'] += img_overlap_loss

        data_batch['pc_overlap_precision'] = pc_overlap_precision
        data_batch['pc_overlap_recall'] = pc_overlap_recall
        data_batch['pc_overlap_accuracy'] = pc_overlap_accuracy
        data_batch['img_overlap_precision'] = img_overlap_precision
        data_batch['img_overlap_recall'] = img_overlap_recall
        data_batch['img_overlap_accuracy'] = img_overlap_accuracy
        # print(pc_overlap_precision, pc_overlap_recall, pc_overlap_accuracy, img_overlap_precision, img_overlap_recall, img_overlap_accuracy)

        return 0


class GeometricDistanceHead(nn.Module):
    def __init__(self, config):
        super(GeometricDistanceHead, self).__init__()
        self.config = config
        self.dist_thres = 1
        self.pos_margin = 0.1
        self.neg_margin = 1.4
        self.lambda_geo = 1
        f = config.embed_dim

        self.point_fuse_convs = nn.ModuleList()
        self.point_fuse_convs.append(ConvBNReLURes1D(2 * f, f))
        for _ in range(config.pt_head_res_num - 1):
            self.point_fuse_convs.append(ConvBNReLURes1D(f, f))
        self.pc_geo_head = nn.Sequential(
            nn.Conv1d(f, f, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(f, f, kernel_size=1, stride=1, padding=0)
        )

        self.img_res_convs = nn.ModuleList()
        for _ in range(config.img_fuse_res_num):
            self.img_res_convs.append(ResidualBlock(f, f))
        self.img_geo_head = nn.Sequential(
            nn.Conv2d(f, f, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(f, f, 1, 1, 0),
        )

    def circle_loss(self, img_features, pc_features, distance_map, log_scale=10):
        mask = (distance_map <= self.dist_thres).float()

        # cv2.imshow('correspondence_mask',mask[0].cpu().numpy())
        # key = cv2.waitKey(0)

        pos_mask = mask
        neg_mask = 1 - mask

        # dists = torch.einsum('bdn,bdm->bnm', pc_features, img_features) / 8.0
        dists = torch.sqrt(torch.sum((pc_features.unsqueeze(-1)-img_features.unsqueeze(-2))**2,dim=1))
        # dists = 1 - torch.sum(pc_features.unsqueeze(-1) * img_features.unsqueeze(-2), dim=1)

        pos = dists - 1e5 * neg_mask
        pos_weight = (pos - self.pos_margin).detach()
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight)

        # pos_weight[pos_weight>0]=1.
        # positive_row=torch.sum((pos[:,:num_kpt,:]-pos_margin)*pos_weight[:,:num_kpt,:],dim=-1)/(torch.sum(pos_weight[:,:num_kpt,:],dim=-1)+1e-8)
        # positive_col=torch.sum((pos[:,:,:num_kpt]-pos_margin)*pos_weight[:,:,:num_kpt],dim=-2)/(torch.sum(pos_weight[:,:,:num_kpt],dim=-2)+1e-8)
        lse_positive_row = torch.logsumexp(log_scale * (pos - self.pos_margin) * pos_weight, dim=-1)
        lse_positive_col = torch.logsumexp(log_scale * (pos - self.pos_margin) * pos_weight, dim=-2)

        neg = dists + 1e5 * pos_mask
        neg_weight = (self.neg_margin - neg).detach()
        neg_weight = torch.max(torch.zeros_like(neg_weight), neg_weight)
        # neg_weight[neg_weight>0]=1.
        # negative_row=torch.sum((neg[:,:num_kpt,:]-neg_margin)*neg_weight[:,:num_kpt,:],dim=-1)/torch.sum(neg_weight[:,:num_kpt,:],dim=-1)
        # negative_col=torch.sum((neg[:,:,:num_kpt]-neg_margin)*neg_weight[:,:,:num_kpt],dim=-2)/torch.sum(neg_weight[:,:,:num_kpt],dim=-2)
        lse_negative_row = torch.logsumexp(log_scale * (self.neg_margin - neg) * neg_weight, dim=-1)
        lse_negative_col = torch.logsumexp(log_scale * (self.neg_margin - neg) * neg_weight, dim=-2)

        loss_col = F.softplus(lse_positive_row + lse_negative_row) / log_scale
        loss_row = F.softplus(lse_positive_col + lse_negative_col) / log_scale

        loss = loss_col + loss_row

        return torch.mean(loss), dists

    def cal_match_accuracy(self, data_batch):
        # <====== only calculate the matching inlier ratio of the first sample in a batch ======>
        with torch.no_grad():
            pc_geo_feat = data_batch['pc_geo_feat'][0]
            img_geo_feat = data_batch['img_geo_feat'][0]
            pc_mask = data_batch['pc_mask'][0].bool()
            # if self.training:
            #     pc_mask = data_batch['pc_mask'][0].bool()
            # else:
            #     pc_mask = data_batch['pc_overlap_logits'][0].argmax(0)
            #     pc_mask = pc_mask.bool()
            point_xy_all = data_batch['point_xy_float_all'][0].cuda()

            pc_geo_feat = pc_geo_feat[:, pc_mask]
            point_xy = point_xy_all[:, pc_mask]
            img_geo_feat = img_geo_feat.view(64, -1)

            dist = []
            for i in range((pc_geo_feat.shape[1] // 2000) + 1):
                temp_pc = pc_geo_feat[:, i * 2000: (i + 1) * 2000]
                # dist_temp = 1 - torch.sum(temp_pc.unsqueeze(-1) * img_geo_feat.unsqueeze(-2), dim=0)
                dist_temp = torch.sqrt(torch.sum((temp_pc.unsqueeze(-1) - img_geo_feat.unsqueeze(-2)) ** 2, dim=0))
                dist.append(dist_temp)
            dist = torch.cat(dist, dim=0)

            min_idx = dist.argmin(1).cpu()
            img_x = torch.linspace(0, 128 - 1, 128).cuda().unsqueeze(0).expand(40, 128).unsqueeze(0)
            img_y = torch.linspace(0, 40 - 1, 40).cuda().unsqueeze(1).expand(40, 128).unsqueeze(0)
            img_xy = torch.cat([img_x, img_y], dim=0)
            img_xy = img_xy.view(2, -1)

            pred_xy = img_xy[:, min_idx]
            right = torch.sqrt(torch.sum((pred_xy - point_xy) ** 2, dim=0)) <= 3.0
            # print(data_batch["inlier_mask_i"])
            # print("inlier mask:", right.sum())
            ir = right.sum() / right.shape[0]
            data_batch['matching_ir'] = ir

    def forward(self, data_batch):
        # <------ point cloud feature ------>
        fused_node_feat = data_batch['fused_node_feat']
        device = fused_node_feat.device
        pt2node = data_batch['pt2node'].to(device)
        pt_feat = data_batch['pt_feat']

        f = fused_node_feat.shape[1]
        b, n = pt2node.shape[0], pt2node.shape[1]
        scattered_pt_node_feat = torch.gather(fused_node_feat, index=pt2node.unsqueeze(1).expand(b, f, n), dim=2)
        fused_pt_feat = torch.cat([pt_feat, scattered_pt_node_feat], dim=1)

        for layer in self.point_fuse_convs:
            fused_pt_feat = layer(fused_pt_feat)

        pc_geo_feat = self.pc_geo_head(fused_pt_feat)
        pc_geo_feat = F.normalize(pc_geo_feat, dim=1, p=2)

        # <------ image feature ------>
        fused_img_feat = data_batch['fused_img_feat']
        for layer in self.img_res_convs:
            fused_img_feat = layer(fused_img_feat)
        img_geo_feat = self.img_geo_head(fused_img_feat)
        img_geo_feat = F.normalize(img_geo_feat, dim=1, p=2)

        # <------ index image feature ------>
        pc_xy_int_for_circle_loss = data_batch['pc_xy_int_for_circle_loss'].cuda()
        temp = []
        for i in range(img_geo_feat.shape[0]):
            temp.append(img_geo_feat[i][:, pc_xy_int_for_circle_loss[i][1, :], pc_xy_int_for_circle_loss[i][0, :]])
        pixel_feat = torch.stack(temp, 0)

        # <------ index point cloud feature ------>
        pc_idx_for_circle_loss = data_batch['pc_idx_for_circle_loss'].cuda()
        temp = []
        for i in range(pc_geo_feat.shape[0]):
            temp.append(pc_geo_feat[i][:, pc_idx_for_circle_loss[i]])
        point_feat = torch.stack(temp, 0)

        pc_xy_float_for_circle_loss = data_batch['pc_xy_float_for_circle_loss'].cuda()
        distance_map = torch.sqrt(torch.sum(torch.square(pc_xy_float_for_circle_loss.unsqueeze(-1) - pc_xy_int_for_circle_loss.unsqueeze(-2)), dim=1))

        # print(pixel_feat.shape, point_feat.shape, distance_map.shape)
        geometric_loss, dist = self.circle_loss(pixel_feat, point_feat, distance_map)

        data_batch['pc_geo_feat'] = pc_geo_feat
        data_batch['img_geo_feat'] = img_geo_feat

        # <====== run it during training the geo_model ======>
        # self.cal_match_accuracy(data_batch)

        data_batch['geometric_loss'] = self.lambda_geo * geometric_loss
        data_batch['loss'] += self.lambda_geo * geometric_loss

        return 0


class MultiHeadModel(nn.Module):
    def __init__(self, config):
        super(MultiHeadModel, self).__init__()
        self.config = config
        self.encoder_decoder = IMGPCEnDecoder(config)

        self.overlap_head = OverlapDetectionHead(config)

        self.geo_head = GeometricDistanceHead(config)

    def cal_matcning_ground_truth(self, data_batch):
        # <====== only calculate the matching inlier ratio of the first sample in a batch ======>
        with torch.no_grad():
            pc_geo_feat = data_batch['pc_geo_feat'][0]
            img_geo_feat = data_batch['img_geo_feat'][0]
            pc_mask = data_batch['pc_overlap_pred'][0]

            point_xy_all = data_batch['point_xy_float_all'][0].cuda()

            pc_geo_feat = pc_geo_feat[:, pc_mask]
            point_xy = point_xy_all[:, pc_mask]
            img_geo_feat = img_geo_feat.view(64, -1)

            dist = []
            for i in range((pc_geo_feat.shape[1] // 2000) + 1):
                temp_pc = pc_geo_feat[:, i * 2000: (i + 1) * 2000]
                # dist_temp = 1 - torch.sum(temp_pc.unsqueeze(-1) * img_geo_feat.unsqueeze(-2), dim=0)
                dist_temp = torch.sqrt(torch.sum((temp_pc.unsqueeze(-1) - img_geo_feat.unsqueeze(-2)) ** 2, dim=0))
                dist.append(dist_temp)
            dist = torch.cat(dist, dim=0)

            min_idx = dist.argmin(1).cpu()
            img_x = torch.linspace(0, 128 - 1, 128).cuda().unsqueeze(0).expand(40, 128).unsqueeze(0)
            img_y = torch.linspace(0, 40 - 1, 40).cuda().unsqueeze(1).expand(40, 128).unsqueeze(0)
            img_xy = torch.cat([img_x, img_y], dim=0)
            img_xy = img_xy.view(2, -1)

            pred_xy = img_xy[:, min_idx]
            data_batch['feat_matching_centers'] = pred_xy
            right = torch.sqrt(torch.sum((pred_xy - point_xy) ** 2, dim=0)) <= 3.0
            data_batch['inlier_matching_ground_truth'] = right

    def forward(self, data_batch):

        self.encoder_decoder(data_batch)

        data_batch['loss'] = 0.

        self.overlap_head(data_batch)

        self.geo_head(data_batch)

        pc_overlap_logits = data_batch['pc_overlap_logits']
        # pc_overlap_pred = pc_overlap_logits.argmax(1).bool()
        # print(pc_overlap_pred.sum())
        pc_overlap_prob = torch.softmax(pc_overlap_logits, dim=1)[:, 1, :]
        pc_overlap_pred = pc_overlap_prob > 0.5
        data_batch['pc_overlap_pred'] = pc_overlap_pred
        pc_overlap_pred = pc_overlap_prob > 0.8
        data_batch['pc_overlap_pred_standby'] = pc_overlap_pred
        data_batch['pc_is_in_cam_scores'] = pc_overlap_prob

        img_overlap_logits = data_batch['img_overlap_logits']
        img_overlap_pred = torch.softmax(img_overlap_logits, dim=1)[:, 1, :]
        # img_overlap_pred = img_overlap_pred > 0.8
        img_overlap_pred = img_overlap_pred.view(img_overlap_pred.shape[0], 40, 128)
        data_batch['img_overlap_pred'] = img_overlap_pred

        inlier_mask_in_cam_i = torch.zeros_like(pc_overlap_pred).cuda()
        inlier_mask_in_cam_i[pc_overlap_pred] = 1.0
        data_batch['inlier_mask_in_cam_i'] = inlier_mask_in_cam_i.bool()
        matrix = torch.eye(4).cuda()
        matrix = matrix.unsqueeze(0)
        data_batch['matrix_accumulated'] = matrix

        # <====== run it after convergence of the geo_model ======>
        # self.cal_matcning_ground_truth(data_batch)

        return 0

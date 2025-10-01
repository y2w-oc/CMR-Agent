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
from .MultiHeadModel import MultiHeadModel
from .PointNN import ConvBNReLURes1D
from .ImageResNet import ResidualBlock
from .PointViT import PointTransformer
sys.path.append("..")
from utils import PositionEncodingSine2D, Lovasz_loss
from .LinearAttention import LinearAttention
from .focal_loss import FocalLoss

import time
import scipy.io as scio


class IterModel(nn.Module):
    def __init__(self, config):
        super(IterModel, self).__init__()
        self.config = config
        self.nlabel = 9
        self.base = torch.from_numpy(np.array(range(int(-(self.nlabel-1) / 2), int((self.nlabel-1) / 2) + 1))).unsqueeze(0).cuda()

        # self.multi_head_model = MultiHeadModel(config)

        # sate_dict = torch.load("./checkpoint/geo_feat.pth")
        # self.multi_head_model.load_state_dict(sate_dict)
        # self.multi_head_model.eval()

        self.ce_loss = nn.CrossEntropyLoss()

        # <====== predict the pose ======>
        self.cost_volume_convs = nn.Sequential(nn.Conv3d(128 + 2, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=1),
                                               nn.BatchNorm3d(64),
                                               nn.LeakyReLU(inplace=True),
                                               nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=1),
                                               nn.LeakyReLU(inplace=True),
                                               nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2)),
                                               nn.Conv3d(64, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=1),
                                               nn.BatchNorm3d(32),
                                               nn.LeakyReLU(inplace=True),
                                               nn.Conv3d(32, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=1),
                                               nn.LeakyReLU(inplace=True),
                                               nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2)),
                                               nn.Conv3d(32, 16, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=1),
                                               nn.BatchNorm3d(16),
                                               nn.LeakyReLU(inplace=True),
                                               nn.Conv3d(16, 16, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=1),
                                               nn.LeakyReLU(inplace=True),
                                               nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2)),
                                               nn.Conv3d(16, 16, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=1),
                                               nn.BatchNorm3d(16),
                                               nn.LeakyReLU(inplace=True),
                                               nn.Conv3d(16, 8, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=1),
                                               nn.LeakyReLU(inplace=True),
                                               nn.AvgPool3d((1, 5, 16), stride=1),
                                               nn.Conv3d(8, 4, kernel_size=(1, 1, 1), padding=(0, 0, 0), stride=1),
                                               nn.LeakyReLU(inplace=True),
                                               nn.Conv3d(4, 1, kernel_size=(1, 1, 1), padding=(0, 0, 0), stride=1),
                                               )

        # <====== predict the projected point feat weight ======>
        # f = config.embed_dim
        #
        # self.fuse_conv_0 = ResidualBlock(2 * f, f)
        # self.fuse_conv_1 = ResidualBlock(f, f)
        # self.fuse_conv_2 = nn.Sequential(
        #     nn.Conv2d(f, f, 3, 1, 1),
        #     nn.BatchNorm2d(f),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.Conv2d(f, 32, 3, 1, 1),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.Conv2d(32, 1, 3, 1, 1),
        #     nn.Sigmoid()
        # )
        #
        # self.depth_conv_0 = nn.Sequential(
        #     nn.Conv2d(1, f, 3, 1, 1),
        #     nn.BatchNorm2d(f),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.Conv2d(f, f, 3, 1, 1),
        #     nn.BatchNorm2d(f),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        # )
        # self.depth_conv_1 = ResidualBlock(f, f, 2)
        # self.depth_conv_2 = ResidualBlock(f, f, 2)

    @torch.no_grad()
    def angle2matrix(self, angle):
        """
        ref: https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/euler.py
        input size:  ... * 3
        output size: ... * 3 * 3
        """
        dims = [dim for dim in angle.shape]
        angle = angle.view(-1, 3)

        i = 0
        j = 1
        k = 2
        ai = angle[:, 0]
        aj = angle[:, 1]
        ak = angle[:, 2]
        si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
        ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
        cc, cs = ci * ck, ci * sk
        sc, ss = si * ck, si * sk

        M = torch.eye(3)
        M = M.view(1, 3, 3)
        M = M.repeat(angle.shape[0], 1, 1).cuda()

        M[:, i, i] = cj * ck
        M[:, i, j] = sj * sc - cs
        M[:, i, k] = sj * cc + ss
        M[:, j, i] = cj * sk
        M[:, j, j] = sj * ss + cc
        M[:, j, k] = sj * cs - sc
        M[:, k, i] = -sj
        M[:, k, j] = cj * si
        M[:, k, k] = cj * ci

        return M.view(dims + [3])

    @torch.no_grad()
    def sample_poses(self, data_batch):
        R_amplitude = data_batch['R_amplitude'].cuda()
        T_amplitude = data_batch['T_amplitude'].cuda()

        delta_R = 2 * R_amplitude / (self.nlabel - 1)
        delta_R = delta_R * self.base
        data_batch["delta_R"] = delta_R
        delta_R_y = delta_R.unsqueeze(-1)
        delta_R_x = torch.zeros_like(delta_R_y)
        delta_R_z = torch.zeros_like(delta_R_y)
        delta_R = torch.cat([delta_R_x, delta_R_y, delta_R_z], dim=-1)
        delta_R = self.angle2matrix(delta_R)

        delta_T = 2 * T_amplitude / (self.nlabel - 1)
        delta_T = delta_T * self.base
        data_batch["delta_T"] = delta_T

        delta_T_x = delta_T.clone().unsqueeze(-1)
        delta_T_x = delta_T_x.repeat(1, 1, self.nlabel)
        delta_T_x = delta_T_x.unsqueeze(-1)
        delta_T_z = delta_T.clone().unsqueeze(-2)
        delta_T_z = delta_T_z.repeat(1, self.nlabel, 1)
        delta_T_z = delta_T_z.unsqueeze(-1)
        delta_T_y = torch.zeros_like(delta_T_x)
        delta_T = torch.cat([delta_T_x, delta_T_y, delta_T_z], dim=-1)

        delta_RT = torch.eye(4).cuda()
        delta_RT = delta_RT.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        delta_RT = delta_RT.repeat(R_amplitude.shape[0], self.nlabel, self.nlabel, self.nlabel, 1, 1)
        delta_T = delta_T.unsqueeze(1).repeat(1, self.nlabel, 1, 1, 1)
        delta_R = delta_R.unsqueeze(2).unsqueeze(2).repeat(1, 1, self.nlabel, self.nlabel, 1, 1)
        delta_RT[:, :, :, :, 0:3, 3] = delta_T
        delta_RT[:, :, :, :, 0:3, 0:3] = delta_R
        # del delta_R
        # del delta_T
        delta_RT_inv = torch.linalg.inv(delta_RT)
        # del delta_RT
        delta_RT_inv = delta_RT_inv[:, :, :, :, 0:3, :]

        return delta_RT_inv

    def cost_volume_ce_loss(self, data_batch):
        label_R = data_batch['label_R'].cuda().float()
        label_T_x = data_batch['label_T_x'].cuda().float()
        label_T_z = data_batch['label_T_z'].cuda().float()

        label_T_x = label_T_x.unsqueeze(-1)
        label_T_z = label_T_z.unsqueeze(-2)
        label = label_T_x @ label_T_z
        label_R = label_R.unsqueeze(-1)
        label = label.view(label.shape[0], -1).unsqueeze(-2)
        label = label_R @ label
        label = label.view(label.shape[0], -1)
        data_batch['cost_volume_label'] = label
        label = label.argmax(dim=1)
        label = label.view(-1)
        fused_out = data_batch["cost_colume_logits"]
        # print(fused_out.shape, label.shape)
        loss = self.ce_loss(fused_out, label)
        data_batch['cost_volume_loss'] = loss

    def cost_volume_ce_loss_separated(self, data_batch, pred):
        label_R = data_batch['label_R'].cuda().float()
        label_T_x = data_batch['label_T_x'].cuda().float()
        label_T_z = data_batch['label_T_z'].cuda().float()

        pred = pred.view(self.nlabel, self.nlabel, self.nlabel)
        pred_ry = pred.sum(-1).sum(-1)
        pred_tx = pred.sum(0).sum(-1)
        pred_tz = pred.sum(0).sum(0)
        pred_ry = pred_ry.unsqueeze(0)
        pred_tx = pred_tx.unsqueeze(0)
        pred_tz = pred_tz.unsqueeze(0)

        label_ry = label_R.argmax(dim=1)
        label_tx = label_T_x.argmax(dim=1)
        label_tz = label_T_z.argmax(dim=1)

        loss_ry = self.ce_loss(pred_ry, label_ry)
        loss_tx = self.ce_loss(pred_tx, label_tx)
        loss_tz = self.ce_loss(pred_tz, label_tz)
        data_batch['cost_volume_loss'] = loss_ry + loss_tx + loss_tz

    def rt_l1_loss(self, data_batch, ry, tx, tz):
        R_amplitude = data_batch['R_amplitude'].cuda()
        T_amplitude = data_batch['T_amplitude'].cuda()
        angles = data_batch['angles'].cuda().float()
        translation = data_batch['translation'].cuda().float()
        angles_error = 180 * torch.abs(ry - angles[:,1]) / R_amplitude
        translation_error = (torch.abs(tx - translation[:, 0]) + torch.abs(tz - translation[:, 2])) / T_amplitude
        data_batch['cost_volume_loss'] = angles_error + translation_error
        print(angles_error, translation_error)

    def get_transform_matrix(self, ry, tx, tz):
        rx = torch.zeros_like(ry)
        rz = torch.zeros_like(ry)
        ty = torch.zeros_like(tx)

        rx = rx.unsqueeze(-1)
        ry = ry.unsqueeze(-1)
        rz = rz.unsqueeze(-1)
        tx = tx.unsqueeze(-1)
        ty = ty.unsqueeze(-1)
        tz = tz.unsqueeze(-1)

        r = torch.cat([rx, ry, rz], dim=1)
        rm = self.angle2matrix(r)
        t = torch.cat([tx, ty, tz], dim=1)

        m = torch.eye(4).cuda()
        m = m.unsqueeze(0)
        m[:, 0:3, 0:3] = rm
        m[:, 0:3, 3] = t

        m_inv = torch.linalg.inv(m)
        return m_inv

    def forward(self, data_batch):
        # with torch.no_grad():
        #     self.multi_head_model(data_batch)

        # <======================================== predict the weight ========================================>
        # img_geo_feat = data_batch['img_geo_feat']
        # depth = data_batch['depth'].cuda().unsqueeze(1)
        # depth = depth / depth.max()
        #
        # depth_feat = self.depth_conv_0(depth)
        # depth_feat = self.depth_conv_1(depth_feat)
        # depth_feat = self.depth_conv_2(depth_feat)
        #
        # depth_img_feat = torch.cat([img_geo_feat, depth_feat], dim=1)
        #
        # img_geo_feat_0 = self.fuse_conv_0(depth_img_feat)
        #
        # img_geo_feat_1 = self.fuse_conv_1(img_geo_feat_0)
        #
        # weight = self.fuse_conv_2(img_geo_feat_1)

        # <======================================== sample pose and observe =========================================>
        pc_mask = data_batch['pc_overlap_pred'][0]
        if pc_mask.sum() == 0:
            pc_mask = data_batch['pc_overlap_pred_standby'][0]
        delta_RT = self.sample_poses(data_batch)
        delta_RT = delta_RT.view(delta_RT.shape[0], -1, 3, 4)
        pc = data_batch['pc_i']
        pc = pc.unsqueeze(1)

        pc_RT = delta_RT[:, :, 0:3, 0:3] @ pc + delta_RT[:, :, 0:3, 3:4]

        K = data_batch['K'].cuda().unsqueeze(1)

        pc_RT_K = K @ pc_RT
        pc_RT_K[:, :, 0:2, :] = pc_RT_K[:, :, 0:2, :] / pc_RT_K[:, :, 2:3, :]

        img = data_batch['img']
        H = img.shape[2] // 4
        W = img.shape[3] // 4
        # print(H,W)

        pc_is_in_cam = (pc_RT_K[:, :, 0, :] >= 0) & \
                       (pc_RT_K[:, :, 0, :] <= (W - 1)) & \
                       (pc_RT_K[:, :, 1, :] >= 0) & \
                       (pc_RT_K[:, :, 1, :] <= (H - 1)) & \
                       (pc_RT_K[:, :, 2, :] > 0)

        # pc_overlap_pred = data_batch['pc_overlap_pred'].unsqueeze(1)

        # pc_is_in_cam = pc_is_in_cam & pc_overlap_pred
        pc_is_in_cam = pc_is_in_cam[:, :, pc_mask]

        pc_RT_K = pc_RT_K[:, :, 0:2, :].round().int()

        pc_geo_feat = data_batch['pc_geo_feat']

        pc_geo_feat = pc_geo_feat[:, :, pc_mask]

        pc_geo_feat = pc_geo_feat.unsqueeze(1)
        pc_geo_feat = pc_geo_feat.repeat(1, self.nlabel*self.nlabel*self.nlabel, 1, 1)

        pc_RT_K = pc_RT_K[:, :, :, pc_mask]
        pc_warped = pc_RT_K.permute(0, 1, 3, 2)

        pc_idx = pc_warped[:, :, :, 1] * W + pc_warped[:, :, :, 0]

        pc_idx[~pc_is_in_cam] = 5120
        # pc_idx[:, :, 0] = 5120

        pc_is_in_cam_scores = data_batch['pc_is_in_cam_scores']
        pc_is_in_cam_scores = pc_is_in_cam_scores[:, pc_mask]
        pc_is_in_cam_scores = pc_is_in_cam_scores.unsqueeze(1).repeat(1,pc_is_in_cam.shape[1],1)
        # print(pc_is_in_cam_scores.shape, pc_is_in_cam.shape)
        pc_is_in_cam_scores[~pc_is_in_cam] = 0.0
        # pc_is_in_cam_scores = pc_is_in_cam.float()

        pc_warped_geo_feat = []
        pc_warped_occupancy = []
        for i in range((pc_idx.shape[1] // 200) + 1):
            pc_geo_feat_temp = pc_geo_feat[:, 200 * i:200 * (i + 1), :, :]
            feat_pad = torch.zeros_like(pc_geo_feat_temp[:, :, :, 0:1]).cuda()
            pc_geo_feat_temp = torch.cat([pc_geo_feat_temp, feat_pad], dim=-1)

            occupancy_temp = pc_is_in_cam_scores[:, 200 * i:200 * (i + 1), :]
            occupancy_pad = torch.zeros_like(occupancy_temp[:, :, 0:1]).cuda()
            occupancy_temp = torch.cat([occupancy_temp, occupancy_pad], dim=-1)

            pc_idx_temp = pc_idx[:, 200 * i:200 * (i + 1), :].long()
            idx_pad = torch.ones_like(pc_idx_temp[:, :, 0:1]).cuda().long() * 5120
            pc_idx_temp = torch.cat([pc_idx_temp, idx_pad], dim=-1)
            feat_temp = torch_scatter.scatter_mean(pc_geo_feat_temp, pc_idx_temp.unsqueeze(2).repeat(1, 1, 64, 1),
                                                   dim=3)
            o_temp = torch_scatter.scatter_sum(occupancy_temp, pc_idx_temp, dim=2)
            pc_warped_geo_feat.append(feat_temp)
            pc_warped_occupancy.append(o_temp)
        pc_warped_geo_feat = torch.cat(pc_warped_geo_feat, dim=1)
        pc_warped_occupancy = torch.cat(pc_warped_occupancy, dim=1)
        # pc_warped_occupancy = (pc_warped_occupancy > 0).float()

        pc_warped_geo_feat = pc_warped_geo_feat[:, :, :, :5120]
        pc_warped_occupancy = pc_warped_occupancy[:, :, :5120]

        img_geo_feat = data_batch['img_geo_feat']
        # img_geo_feat = img_geo_feat * img_weight

        img_geo_feat = img_geo_feat.unsqueeze(1).repeat(1, self.nlabel * self.nlabel * self.nlabel, 1, 1, 1)

        b, n, c, h, w = img_geo_feat.shape
        pc_warped_geo_feat = pc_warped_geo_feat.view(b, n, c, h, w)

        # weight = weight.unsqueeze(1)
        # pc_warped_geo_feat = pc_warped_geo_feat * weight

        # data_batch['weight'] = weight
        # data_batch["weight_regularization"] = weight.mean()

        # fused_geo_feat = torch.cat([img_geo_feat, pc_warped_geo_feat], dim=2)

        # fused_geo_feat = fused_geo_feat * weight

        img_overlap_pred = data_batch['img_overlap_pred'].unsqueeze(1).unsqueeze(1)
        data_batch['weight'] = img_overlap_pred
        pc_warped_occupancy = pc_warped_occupancy.view(1, self.nlabel**3, 40, 128)
        data_batch['3d_weight'] = pc_warped_occupancy
        pc_warped_occupancy = pc_warped_occupancy.unsqueeze(2)
        img_overlap_pred = img_overlap_pred.repeat(1, self.nlabel**3, 1, 1, 1)
        fused_geo_feat = torch.cat([img_geo_feat, pc_warped_geo_feat, pc_warped_occupancy, img_overlap_pred], dim=2)

        fused_geo_feat = fused_geo_feat.permute(0, 2, 1, 3, 4)

        fused_out = self.cost_volume_convs(fused_geo_feat)

        fused_out = fused_out.view(fused_out.shape[0], fused_out.shape[2])

        data_batch["cost_colume_logits"] = fused_out

        # <================================== ce-loss separated ==========================>
        # self.cost_volume_ce_loss_separated(data_batch, fused_out)
        # delta_r = data_batch['delta_R']
        # delta_t = data_batch['delta_T']
        # pred = fused_out.view(self.nlabel, self.nlabel, self.nlabel)
        # pred_ry = pred.sum(-1).sum(-1)
        # pred_tx = pred.sum(0).sum(-1)
        # pred_tz = pred.sum(0).sum(0)
        # pred_ry = pred_ry.unsqueeze(0)
        # pred_tx = pred_tx.unsqueeze(0)
        # pred_tz = pred_tz.unsqueeze(0)
        # pred_ry = torch.softmax(pred_ry, dim=1)
        # pred_tx = torch.softmax(pred_tx, dim=1)
        # pred_tz = torch.softmax(pred_tz, dim=1)
        #
        # pred_ry_i = (pred_ry * delta_r).sum(dim=1)
        # pred_tx_i = (pred_tx * delta_t).sum(dim=1)
        # pred_tz_i = (pred_tz * delta_t).sum(dim=1)

        self.cost_volume_ce_loss(data_batch)
        # <======== predict the transformation and update the matrix and point cloud ======>
        pred = torch.softmax(fused_out, dim=1)
        delta_r = data_batch['delta_R']
        delta_t = data_batch['delta_T']

        # delta_ry = delta_r.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, delta_t.shape[1], delta_t.shape[1]).view(
        #     delta_t.shape[0], -1)
        # delta_tx = delta_t.unsqueeze(1).unsqueeze(-1).repeat(1, delta_r.shape[1], 1, delta_r.shape[1]).view(
        #     delta_t.shape[0], -1)
        # delta_tz = delta_t.unsqueeze(1).unsqueeze(1).repeat(1, delta_r.shape[1], delta_r.shape[1], 1).view(
        #     delta_t.shape[0], -1)
        #
        # pred_ry_i = (pred * delta_ry).sum(dim=1)
        # pred_tx_i = (pred * delta_tx).sum(dim=1)
        # pred_tz_i = (pred * delta_tz).sum(dim=1)

        # temp = torch.abs(delta_tx[0,:] - pred_tx_i)
        # temp_x = (temp == temp.min())
        # temp = torch.abs(delta_tz[0, :] - pred_tz_i)
        # temp_z = (temp == temp.min())
        # mask_r = temp_x & temp_z
        # pred[:, ~mask_r] = 0
        # # idx = pred.argmax(dim=1)
        # # pred_ry_i = delta_ry[0, idx]
        # pred = pred / pred.sum()
        # pred_ry_i = (pred * delta_ry).sum(dim=1)

        # idx = pred.argmax(dim=1)
        # pred_ry_i = delta_ry[0, idx]
        # pred_tx_i = delta_tx[0, idx]
        # pred_tz_i = delta_tz[0, idx]

        # <====================================================================>
        # self.rt_l1_loss(data_batch, pred_ry_i, pred_tx_i, pred_tz_i)
        # <====================================================================>

        delta_r = delta_r[0]
        delta_t = delta_t[0]
        pred = pred[0]
        pred = pred.view(self.nlabel, self.nlabel, self.nlabel)
        pred_ry = pred.sum(-1).sum(-1)
        pred_tx = pred.sum(0).sum(-1)
        pred_tz  = pred.sum(0).sum(0)

        idx = pred_ry.argmax()
        pred_ry_i = delta_r[idx].unsqueeze(0)
        idx_x = pred_tx.argmax()
        pred_tx_i = delta_t[idx_x].unsqueeze(0)
        idx_z = pred_tz.argmax()
        pred_tz_i = delta_t[idx_z].unsqueeze(0)

        # pred_ry = pred[:, idx_x, idx_z]
        # idx = pred_ry.argmax()
        # pred_ry_i = delta_r[idx].unsqueeze(0)

        pred = pred.view(-1)
        id_weight = pred.argmax()
        data_batch['3d_weight_id'] = id_weight

        matrix_i = self.get_transform_matrix(pred_ry_i, pred_tx_i, pred_tz_i)

        data_batch['matrix_i'] = matrix_i

        data_batch['matrix_accumulated'] = matrix_i @ data_batch['matrix_accumulated']

        pc_i = data_batch['pc_i']
        data_batch['pc_i'] = matrix_i[:, 0:3, 0:3] @ pc_i + matrix_i[:, 0:3, 3:4]

        return 0

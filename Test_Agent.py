import math

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
import time
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import random
import argparse
from config import KittiConfiguration, NuScenesConfiguration
from dataset import KittiDataset, NuScenesDataset
from models import CMRAgent, MultiHeadModel
from environment import environment as env
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import scipy.io as scio
import cv2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def angle2matrix(angle):
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


def get_transform_matrix(ry, tx, tz):
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
    rm = angle2matrix(r)
    t = torch.cat([tx, ty, tz], dim=1)

    m = torch.eye(4).cuda()
    m = m.unsqueeze(0)
    m[:, 0:3, 0:3] = rm
    m[:, 0:3, 3] = t

    m_inv = torch.linalg.inv(m)
    return m_inv


def get_P_diff(P_pred_np, P_gt_np):
    r_diff = np.dot(P_pred_np[0:3, 0:3], P_gt_np[0:3, 0:3].T)
    R_diff = Rotation.from_matrix(r_diff)
    error_r = R_diff.as_euler('XYZ', degrees=True)
    angles_diff = np.sum(np.abs(error_r))
    t_diff = np.linalg.norm(P_pred_np[0:3, 3] - P_gt_np[0:3, 3])
    return t_diff, angles_diff


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image to point Registration')
    parser.add_argument('--dataset', type=str, default='kitti', help=" 'kitti' or 'nuscenes' ")
    args = parser.parse_args()

    # <------Configuration parameters------>
    if args.dataset == "kitti":
        config = KittiConfiguration()
        val_dataset = KittiDataset(config, mode='test')
    elif args.dataset == "nuscenes":
        config = NuScenesConfiguration()
        val_dataset = NuScenesDataset(config, mode='test')
    else:
        assert False, "No this dataset choice. Please configure your custom dataset first!"

    set_seed(config.seed)

    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                              drop_last=True, num_workers=config.num_workers)

    geo_model = MultiHeadModel(config)
    sate_dict = torch.load("./checkpoint/KITTI/geo_feat.pth")
    geo_model.load_state_dict(sate_dict)
    geo_model = geo_model.cuda()
    geo_model.eval()

    agent = CMRAgent(config)
    sate_dict = torch.load("./checkpoint/KITTI/agent.pth")
    agent.load_state_dict(sate_dict)
    agent = agent.cuda()
    agent.eval()

    time_list = []
    Angle_Error_list = []
    Translation_Error_list = []

    # for i in range(config.action_num):
    #     Angle_Error_list.append([])
    #     Translation_Error_list.append([])
    #     time_list.append([])

    with torch.no_grad():
        for data in tqdm(test_loader):
            start_time = time.time()
            geo_model(data)

            pose_source, pose_target = env.init(data)
            pose_target = env.to_disentangled(pose_target, data['pc'])

            # inference iteration
            for step in range(config.action_num):
                # observation
                current_state_2d, current_state_3d = env.observation_from_a_pose(data, pose_source)

                # student prediction -- stochastic policy
                action_r_logits, action_t_logits, state_value = agent(current_state_2d, current_state_3d)

                # sample discrete action from the above prediction
                action_r, action_t = agent.action_from_logits(action_r_logits, action_t_logits, deterministic=True)
                # print("Step {} Action: {}, {}".format(step, action_r, action_t))

                # update the current pose according to the action (environment step)
                pose_source = env.step(action_r, action_t, pose_source, config)

                # end_time = time.time()
                # time_list[step].append(end_time - start_time)

                # pred_mat = pose_source[0].cpu().numpy()
                # matrix_gt = pose_target[0].cpu().numpy()
                # t_diff, r_diff = get_P_diff(pred_mat, matrix_gt)
                # Angle_Error_list[step].append(r_diff)
                # Translation_Error_list[step].append(t_diff)
                # print("Step {} Error: {}, {}".format(step, t_diff, r_diff))
            # print("---------------------------------------------------------------------------------------")
            # end_time = time.time()
            # time_list.append(end_time-start_time)

            pred_mat = pose_source[0].cpu().numpy()
            matrix_gt = pose_target[0].cpu().numpy()
            t_diff, r_diff = get_P_diff(pred_mat, matrix_gt)
            print(t_diff, r_diff)
            print("---------------------------------------------------------------------------------------")
            Angle_Error_list.append(r_diff)
            Translation_Error_list.append(t_diff)

        time_list = np.array(time_list)
        t_diff_set = np.array(Translation_Error_list)
        angles_diff_set = np.array(Angle_Error_list)
        scio.savemat("diff_cost_volume.mat", { "Time": time_list})

        mask = (t_diff_set < 5) & (angles_diff_set < 10)
        t_diff_set = t_diff_set[mask]
        angles_diff_set = angles_diff_set[mask]
        print("Average time:", time_list.mean())
        print("Registration Recall:", mask.sum() / mask.shape[0])
        print('RTE Mean:', t_diff_set.mean())
        print('RTE Std:', t_diff_set.std())
        print('RRE Mean:', angles_diff_set.mean())
        print('RRE Std:', angles_diff_set.std())

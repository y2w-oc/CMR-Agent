import torch
import torch_scatter
import time
from scipy.spatial.transform import Rotation
import numpy as np
import math
import sys
import functools
import open3d as o3d

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def to_disentangled(poses, pcd):
    """
    Add rotation-induced translation to translation vector - see eq. 11 in paper.
    """
    poses[:, :3, 3] = poses[:, :3, 3] - pcd[:, 0:3, :].mean(dim=2) \
                      + (poses[:, :3, :3] @ (pcd[:, 0:3, :].mean(dim=2).unsqueeze(-1))).squeeze(-1)
    return poses


@torch.no_grad()
def observation_from_a_pose(data, RT):
    K = data['K'].to(DEVICE)
    pc = data['pc']
    overlap_pred = data['pc_overlap_pred']
    pc_geo_feat = data['pc_geo_feat']
    img_geo_feat = data['img_geo_feat']
    B = pc.shape[0]

    img = data['img']
    H = img.shape[2] // 4
    W = img.shape[3] // 4

    # 2D observation = project the 3D feat onto the image
    observations_2d = []
    for i in range(B):
        pc_i = pc[i:i+1, ...]
        RT_i = RT[i:i + 1, ...]
        K_i = K[i:i + 1, ...]
        overlap_pred_i = overlap_pred[i]

        # used for disentangled transformations
        ret_mean = pc_i.mean(dim=2, keepdim=True)

        in_pc_i = pc_i[:, :, overlap_pred_i]
        pc_geo_feat_i = pc_geo_feat[i:i+1, :, overlap_pred_i]
        img_geo_feat_i = img_geo_feat[i:i+1, ...]

        # disentangled transformations
        # pc_RT_i = RT_i[:, 0:3, 0:3] @ in_pc_i + RT_i[:, 0:3, 3:4]
        pc_RT_i = in_pc_i - ret_mean  # to origin
        pc_RT_i = RT_i[:, 0:3, 0:3] @ pc_RT_i  # rotate
        pc_RT_i = pc_RT_i + ret_mean + RT_i[:, 0:3, 3:4]  # translate

        pc_RT_K_i = K_i @ pc_RT_i
        pc_RT_K_i[:, 0:2, :] = pc_RT_K_i[:, 0:2, :] / pc_RT_K_i[:, 2:3, :]

        pc_is_in_cam_i = (pc_RT_K_i[:, 0, :] >= 0) & \
                       (pc_RT_K_i[:, 0, :] <= (W - 1)) & \
                       (pc_RT_K_i[:, 1, :] >= 0) & \
                       (pc_RT_K_i[:, 1, :] <= (H - 1)) & \
                       (pc_RT_K_i[:, 2, :] > 0)

        pc_RT_K_i = pc_RT_K_i[:, 0:2, :].round().int()

        pc_proj_idx = pc_RT_K_i[:, 1, :] * W + pc_RT_K_i[:, 0, :]

        total_length = H*W
        pc_proj_idx[~pc_is_in_cam_i] = total_length

        feat_pad = torch.zeros_like(pc_geo_feat_i[:, :, 0:1]).to(DEVICE)
        pc_geo_feat_i = torch.cat([pc_geo_feat_i, feat_pad], dim=-1)
        idx_pad = torch.ones_like(pc_proj_idx[:, 0:1]).to(DEVICE).long() * total_length
        pc_proj_idx = torch.cat([pc_proj_idx, idx_pad], dim=-1)

        proj_feat_temp = torch_scatter.scatter_mean(pc_geo_feat_i, pc_proj_idx.unsqueeze(1).repeat(1, 64, 1), dim=2)
        pc_proj_geo_feat_i = proj_feat_temp[:, :, :total_length]
        b, c, h, w = img_geo_feat_i.shape
        pc_proj_geo_feat_i = pc_proj_geo_feat_i.view(b, c, h, w)
        observation_2d_i = torch.cat([img_geo_feat_i, pc_proj_geo_feat_i], dim=1)
        observations_2d.append(observation_2d_i)

    observation_2d = torch.cat(observations_2d, dim=0)

    # 3D observation
    # disentangled transformations
    # pc_RT = RT[:, 0:3, 0:3] @ pc + RT[:, 0:3, 3:4]
    ret_mean = pc.mean(dim=2, keepdim=True)
    pc_RT = pc - ret_mean
    pc_RT = RT[:, 0:3, 0:3] @ pc_RT + ret_mean + RT[:, 0:3, 3:4]

    pc_RT_K = K @ pc_RT
    pc_RT_K[:, 0:2, :] = pc_RT_K[:, 0:2, :] / pc_RT_K[:, 2:3, :]
    pc_is_in_cam = (pc_RT_K[:, 0, :] >= 0) & \
                       (pc_RT_K[:, 0, :] <= (W - 1)) & \
                       (pc_RT_K[:, 1, :] >= 0) & \
                       (pc_RT_K[:, 1, :] <= (H - 1)) & \
                       (pc_RT_K[:, 2, :] > 0)

    # visualization
    # raw_pc = pc[0]
    # pc_1 = o3d.geometry.PointCloud()
    # pc_1.points = o3d.utility.Vector3dVector(raw_pc.cpu().numpy().T)
    # pc_1.paint_uniform_color([100 / 255.0, 100 / 255.0, 100 / 255.0])
    #
    # in_pc = pc[0, :, overlap_pred[0]]
    # pc_2 = o3d.geometry.PointCloud()
    # pc_2.points = o3d.utility.Vector3dVector(in_pc.cpu().numpy().T + [0, -5, 0])
    # pc_2.paint_uniform_color([31 / 255.0, 119 / 255.0, 180 / 255.0])
    #
    # in_pc = pc[0, :, pc_is_in_cam[0]]
    # pc_3 = o3d.geometry.PointCloud()
    # pc_3.points = o3d.utility.Vector3dVector(in_pc.cpu().numpy().T + [0, -10, 0])
    # pc_3.paint_uniform_color([255 / 255.0, 127 / 255.0, 14 / 255.0])
    #
    # o3d.visualization.draw_geometries([pc_1, pc_2, pc_3])

    overlap_pred = overlap_pred.unsqueeze(1).float()
    pc_is_in_cam = pc_is_in_cam.unsqueeze(1).float()

    observation_3d = torch.cat([pc, overlap_pred, pc_is_in_cam], dim=1)

    return observation_2d, observation_3d


def init(data):
    """
    Get the initial observation, the ground-truth pose for the expert and initialize the agent's accumulator (identity).
    """
    # GT (for expert)
    B = data['pc'].shape[0]
    pose_target = data['P'].to(DEVICE)

    # initial estimates (identity, for student)
    pose_source = torch.eye(4, device=DEVICE).repeat(B, 1, 1)

    return pose_source, pose_target


def expert(pose_source, targets, config, data):
    """
    Get the expert action in the current state.
    """
    # compute delta, eq. 10 in paper
    delta_t = targets[:, :3, 3] - pose_source[:, :3, 3]

    delta_R = targets[:, :3, :3] @ pose_source[:, :3, :3].transpose(2, 1)  # global accumulator
    R_diff = Rotation.from_matrix(delta_R.cpu().numpy())
    delta_r = R_diff.as_euler('xyz')
    mask = delta_r[:, 0] > 3
    delta_r[mask, 0] = 0
    delta_r[mask, 2] = 0
    mask_p = delta_r[:, 1] > 0
    delta_r[mask & mask_p, 1] = math.pi - delta_r[mask & mask_p, 1]
    mask_n = delta_r[:, 1] < 0
    delta_r[mask & mask_n, 1] = -1 * math.pi - delta_r[mask & mask_n, 1]
    delta_r = torch.from_numpy(delta_r).to(DEVICE)

    # print(delta_r)
    # print(data['angles'])
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    error_r = torch.abs(delta_r.unsqueeze(-1) - config.r_steps.unsqueeze(0).unsqueeze(0))
    action_r = error_r.argmin(dim=2)

    error_t = torch.abs(delta_t.unsqueeze(-1) - config.t_steps.unsqueeze(0).unsqueeze(0))
    action_t = error_t.argmin(dim=2)

    if not config.is_6_DoF:
        action_r = action_r[:, 1:2]
        action_t = torch.cat([action_t[:, 0:1], action_t[:, 2:3]], dim=1)

    return action_r, action_t


def step(action_r, action_t, pose_source, config):
    """
    Update the current pose using the given actions.
    """
    r_steps = config.r_steps
    t_steps = config.t_steps

    move_r = torch.zeros((action_r.shape[0], 3), device=DEVICE)
    move_t = torch.zeros((action_t.shape[0], 3), device=DEVICE)

    if config.is_6_DoF:
        for i in range(3):
            r = action_r[:, i]
            t = action_t[:, i]
            move_r[:, i] = r_steps[r]
            move_t[:, i] = t_steps[t]
    else:
        r = action_r[:, 0]
        move_r[:, 1] = r_steps[r]
        t = action_t[:, 0]
        move_t[:, 0] = t_steps[t]
        t = action_t[:, 1]
        move_t[:, 2] = t_steps[t]

    # disentangled transformations
    pose_source[:, :3, :3] = euler_angles_to_matrix(move_r, 'XYZ') @ pose_source[:, :3, :3]
    pose_source[:, :3, 3] += move_t

    return pose_source


def euler_angles_to_matrix(euler_angles, convention: str):
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = map(_axis_angle_rotation, convention, torch.unbind(euler_angles, -1))
    return functools.reduce(torch.matmul, matrices)


def _axis_angle_rotation(axis: str, angle):
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def reward(RT, data, prev_distance=None):
    """
    Compute the dense step reward for the updated state.
    """
    pc_in_cam_space = data['pc_in_cam_space'].to(DEVICE)
    pc_mask = data['pc_mask'].to(DEVICE).bool()
    pc = data['pc']
    B = pc.shape[0]

    # disentangled transformations
    # pc_RT = RT[:, 0:3, 0:3] @ pc + RT[:, 0:3, 3:4]
    ret_mean = pc.mean(dim=2, keepdim=True)
    pc = pc - ret_mean

    p2p_distance = torch.zeros(B).to(DEVICE)

    for i in range(B):
        pc_in_cam_space_i = pc_in_cam_space[i]
        pc_i = pc[i]
        pc_mask_i = pc_mask[i]

        pc_in_cam_space_i = pc_in_cam_space_i[:, pc_mask_i]
        pc_i = pc_i[:, pc_mask_i]

        dist = (pc_in_cam_space_i - pc_i) * (pc_in_cam_space_i - pc_i)
        dist = dist.sum(dim=0)
        dist = dist.mean()
        p2p_distance[i] = dist

    p2p_distance = p2p_distance.unsqueeze(-1).unsqueeze(-1)

    if prev_distance is not None:
        better = (p2p_distance < prev_distance).float() * 0.5
        same = (p2p_distance == prev_distance).float() * 0
        worse = (p2p_distance > prev_distance).float() * 0.5

        reward = better - worse - same
        return reward, p2p_distance
    else:
        return torch.zeros_like(p2p_distance), p2p_distance


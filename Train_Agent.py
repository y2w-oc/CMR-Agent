import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import random
import argparse
from config import KittiConfiguration, NuScenesConfiguration
from dataset import KittiDataset, NuScenesDataset
from models import CMRAgent, MultiHeadModel
from environment import environment as env
from environment.buffer import Buffer
from scipy.spatial.transform import Rotation

import scipy.io as scio


cross_entropy = nn.CrossEntropyLoss()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def get_P_diff(P_pred_np, P_gt_np):
    r_diff = np.dot(P_pred_np[0:3, 0:3], P_gt_np[0:3, 0:3].T)
    R_diff = Rotation.from_matrix(r_diff)
    error_r = R_diff.as_euler('XYZ', degrees=True)
    angles_diff = np.sum(np.abs(error_r))
    t_diff = np.linalg.norm(P_pred_np[0:3, 3] - P_gt_np[0:3, 3])
    return t_diff, angles_diff


# @torch.no_grad()
# def select_pc_overlap(data):
#     pc = data['pc']
#     device = pc.device
#     overlap_pred = data['pc_overlap_pred']
#     batch_overlap_sum = overlap_pred.sum(dim=1)
#     B = pc.shape[0]
#     in_pc = []
#     for i in range(B):
#         pc_i = pc[i,...]
#         in_sum_i = batch_overlap_sum[i]
#         overlap_pred_i = overlap_pred[i]
#         in_pc_i = pc_i[:, overlap_pred_i]
#         sample_idx = torch.arange(4096).to(device)
#         sample_idx = sample_idx % in_sum_i
#         in_pc_i = in_pc_i[:, sample_idx]
#         in_pc.append(in_pc_i.unsqueeze(0))
#     in_pc = torch.cat(in_pc, dim=0)
#     data["in_pc_pred"] = in_pc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image to point Registration')
    parser.add_argument('--dataset', type=str, default='kitti', help=" 'kitti' or 'nuscenes' ")
    args = parser.parse_args()

    # <------Configuration parameters------>
    if args.dataset == "kitti":
        config = KittiConfiguration()
        train_dataset = KittiDataset(config, mode='train')
        val_dataset = KittiDataset(config, mode='val')
    elif args.dataset == "nuscenes":
        config = NuScenesConfiguration()
        train_dataset = NuScenesDataset(config, mode='train')
        val_dataset = NuScenesDataset(config, mode='val')
    else:
        assert False, "No this dataset choice. Please configure your custom dataset first!"

    set_seed(config.seed)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               drop_last=True, num_workers=config.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False,
                                              drop_last=True, num_workers=config.num_workers)

    geo_model = MultiHeadModel(config)
    sate_dict = torch.load("./checkpoint/KITTI/geo_feat.pth")
    geo_model.load_state_dict(sate_dict)
    geo_model = geo_model.cuda()
    geo_model.eval()

    agent = CMRAgent(config)
    # sate_dict = torch.load("./checkpoint/agent.pth")
    # model.load_state_dict(sate_dict)
    agent = agent.cuda()

    if config.resume:
        assert config.checkpoint is not None, "Resume checkpoint error, please set a checkpoint in configuration file!"
        sate_dict = torch.load(config.checkpoint)
        agent.load_state_dict(sate_dict)
    else:
        print("New Training!")

    if config.optimizer == 'SGD':
        optimizer = optim.SGD(
            agent.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == 'ADAM':
        optimizer = optim.Adam(
            agent.parameters(),
            lr=config.lr,
            betas=(0.9, 0.99),
            weight_decay=config.weight_decay,
        )

    if config.lr_scheduler == "ExponentialLR":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.scheduler_gamma,
        )
    elif config.lr_scheduler == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.scheduler_gamma,
        )
    elif config.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10,
            eta_min=0.0001,
        )

    now_time = time.strftime('%m-%d-%H-%M', time.localtime())
    log_dir = os.path.join(config.logdir, args.dataset + "_IL_" + now_time)
    ckpt_dir = os.path.join(config.ckpt_dir, args.dataset + "_IL_" + now_time)
    if os.path.exists(ckpt_dir):
        pass
    else:
        os.makedirs(ckpt_dir)
    writer = SummaryWriter(log_dir=log_dir)

    global_step = 0
    pre_fine_loss = 1e7
    current_lr = 0.001
    agent.eval()

    buffer = Buffer(config)
    buffer.start_trajectory()

    current_error_r = np.Inf
    current_error_t = np.Inf

    for epoch in range(config.epoch):
        print("Learning rate: ", optimizer.param_groups[0]['lr'])
        for data in tqdm(train_loader):

            with torch.no_grad():
                # validating
                if global_step % config.val_interval == 0:

                    error_r = []
                    error_t = []
                    for v_data in tqdm(val_loader):
                        geo_model(v_data)

                        pose_source, pose_target = env.init(v_data)
                        pose_target = env.to_disentangled(pose_target, v_data['pc'])

                        for step in range(config.action_num):
                            # observation
                            current_state_2d, current_state_3d = env.observation_from_a_pose(v_data, pose_source)

                            # student prediction -- stochastic policy
                            action_r_logits, action_t_logits, state_value = agent(current_state_2d, current_state_3d)

                            # sample discrete action from the above prediction
                            action_r, action_t = agent.action_from_logits(action_r_logits, action_t_logits,
                                                                          deterministic=True)

                            # update the current pose according to the action (environment step)
                            pose_source = env.step(action_r, action_t, pose_source, config)

                        pred_mat = pose_source[0].cpu().numpy()
                        matrix_gt = pose_target[0].cpu().numpy()
                        t_diff, r_diff = get_P_diff(pred_mat, matrix_gt)
                        error_r.append(r_diff)
                        error_t.append(t_diff)

                    new_error_r = np.mean(error_r)
                    new_error_t = np.mean(error_t)
                    writer.add_scalar('val_error/error_r', new_error_r, global_step=global_step)
                    writer.add_scalar('val_error/error_t', new_error_t, global_step=global_step)

                    if new_error_r < current_error_r or new_error_t < current_error_t:
                        current_error_r = new_error_r if new_error_r < current_error_r else current_error_r
                        current_error_t = new_error_t if new_error_t < current_error_t else current_error_t
                        filename = "epoch-%d-step-%d-R-%f-T-%f.pth" % (epoch, global_step, current_error_r, current_error_t)
                        save_path = os.path.join(ckpt_dir, filename)
                        torch.save(agent.state_dict(), save_path)

                    print("Step:{}, lowest rotation error:{}, lowest translation error:{}".format(global_step, current_error_r, current_error_t))

                # training process
                geo_model(data)
                pose_source, pose_target = env.init(data)
                pose_target = env.to_disentangled(pose_target, data['pc'])
                _, prev_p2p_distance = env.reward(pose_source, data)
                train_rewards = []
                # print(prev_p2p_distance)

                # build trajectory
                for _ in range(config.action_num):
                    # expert prediction
                    expert_action_r, expert_action_t = env.expert(pose_source, pose_target, config, data)

                    # observation
                    current_state_2d, current_state_3d = env.observation_from_a_pose(data, pose_source)

                    # student prediction -- stochastic policy
                    action_r_logits, action_t_logits, state_value = agent(current_state_2d, current_state_3d)

                    # sample discrete action from the above prediction
                    action_r, action_t = agent.action_from_logits(action_r_logits, action_t_logits, deterministic=False)
                    action_logprob, action_entropy = agent.action_logprob_and_entropy(action_r_logits, action_t_logits, action_r, action_t)

                    # update the current pose according to the action (;environment step)
                    pose_source = env.step(action_r, action_t, pose_source, config)

                    # reward computation
                    # reward = torch.zeros((pose_source.shape[0], 1, 1)).to(DEVICE)
                    reward, prev_p2p_distance = env.reward(pose_source, data, prev_distance=prev_p2p_distance)
                    # print(reward.shape, prev_p2p_distance.shape)
                    # print(reward, prev_p2p_distance)

                    # log trajectory in the replay buffer
                    # print(reward.shape, state_value.shape, action_logprob.shape)
                    buffer.log_step(current_state_2d, current_state_3d, state_value, reward,
                                    expert_action_r, expert_action_t, action_r, action_t, action_logprob)
                    train_rewards.append(reward.view(-1))
                # print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
            loss_bc = []
            loss_ppo = []

            if len(buffer) == config.num_trajectory:
                agent.train()
                # convert trajectory to tensor for training (also computes return and advantage over trajectories)
                samples = buffer.get_samples()
                ppo_dataset = torch.utils.data.TensorDataset(*samples)
                ppo_loader = torch.utils.data.DataLoader(ppo_dataset, batch_size=10, shuffle=True,
                                                         drop_last=False)

                for batch in ppo_loader:
                    states_2d, states_3d, state_values, expert_actions_r, expert_actions_t,\
                    action_r, action_t, action_logprob, state_value_ref, advantages = batch

                    # predict using current policy
                    new_action_r_logits, new_action_t_logits, new_state_value = agent(states_2d, states_3d)
                    new_action_logprob, new_action_entropy = agent.action_logprob_and_entropy(new_action_r_logits, new_action_t_logits, action_r, action_t)

                    # behavior cloning loss
                    new_action_r_logits = new_action_r_logits.view(-1, new_action_r_logits.shape[2])
                    new_action_t_logits = new_action_t_logits.view(-1, new_action_t_logits.shape[2])
                    expert_actions_r = expert_actions_r.view(-1)
                    expert_actions_t = expert_actions_t.view(-1)
                    loss_r = cross_entropy(new_action_r_logits, expert_actions_r)
                    loss_t = cross_entropy(new_action_t_logits, expert_actions_t)
                    clone_loss = loss_r + loss_t

                    # reinforcement loss
                    if config.alpha > 0:
                        # -- policy term
                        # ratio: lp > prev_lp --> probability of selecting that action increased
                        ratio = torch.exp(new_action_logprob - action_logprob)
                        # print(new_action_logprob.shape, action_logprob.shape, ratio.shape, advantages.shape)
                        policy_loss = -torch.min(ratio * advantages, ratio.clamp(1 - config.CLIP_EPS, 1 + config.CLIP_EPS) * advantages).mean()

                        # -- value term MSE
                        value_loss = (new_state_value.view(-1, 1) - state_value_ref).pow(2)
                        value_loss = value_loss.mean()

                        # -- entropy term
                        entropy_loss = new_action_entropy.mean()

                    # update agent
                    optimizer.zero_grad()
                    loss = clone_loss
                    # loss = 0
                    loss_bc.append(clone_loss.item())
                    if config.alpha > 0:
                        ppo_loss = policy_loss + value_loss * config.W_VALUE - entropy_loss * config.W_ENTROPY
                        loss += ppo_loss * config.alpha
                        loss_ppo.append(ppo_loss.item())
                    loss.backward()
                    optimizer.step()

                writer.add_scalar('train_loss/BC_Loss', np.mean(loss_bc), global_step=global_step)
                writer.add_scalar('train_loss/PPO_Loss', np.mean(loss_ppo), global_step=global_step)
                writer.add_scalar("train_loss/reward", float(torch.cat(train_rewards, dim=0).mean()), global_step=global_step)
                buffer.clear()
                agent.eval()

            buffer.start_trajectory()
            global_step += 1
        print("%d-th epoch end." % (epoch))
        time.sleep(5)
        lr_scheduler.step()


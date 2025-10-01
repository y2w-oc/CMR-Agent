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
from models import MultiHeadModel


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

    model = MultiHeadModel(config)
    # sate_dict = torch.load("./checkpoint/best.pth")
    # model.load_state_dict(sate_dict)
    model = model.cuda()

    if config.resume:
        assert config.checkpoint is not None, "Resume checkpoint error, please set a checkpoint in configuration file!"
        sate_dict = torch.load(config.checkpoint)
        model.load_state_dict(sate_dict)
    else:
        print("New Training!")

    if config.optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == 'ADAM':
        optimizer = optim.Adam(
            model.parameters(),
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
    log_dir = os.path.join(config.logdir, args.dataset + "_"  + str(config.num_pt) + "_" + now_time)
    ckpt_dir = os.path.join(config.ckpt_dir, args.dataset + "_"  + str(config.num_pt) + "_" + now_time)
    if os.path.exists(ckpt_dir):
        pass
    else:
        os.makedirs(ckpt_dir)
    writer = SummaryWriter(log_dir=log_dir)

    global_step = 0
    pre_fine_loss = 1e7

    model.train()
    for epoch in range(config.epoch):
        print("Learning rate: ", optimizer.param_groups[0]['lr'])
        for data in tqdm(train_loader):
            if global_step % config.val_interval == 0:
                with torch.no_grad():
                    model.eval()
                    loss_list = []
                    loss_geo_list = []
                    loss_pc_overlap_list = []
                    loss_img_overlap_list = []

                    matching_ir_list = []
                    pc_overlap_precision_list = []
                    pc_overlap_recall_list = []
                    pc_overlap_accuracy_list = []
                    img_overlap_precision_list = []
                    img_overlap_recall_list = []
                    img_overlap_accuracy_list = []

                    for v_data in tqdm(val_loader):
                        model(v_data)
                        loss_list.append(v_data['loss'].cpu().numpy())
                        loss_geo_list.append(v_data['geometric_loss'].cpu().numpy())
                        loss_pc_overlap_list.append(v_data['pc_overlap_loss'].cpu().numpy())
                        loss_img_overlap_list.append(v_data['img_overlap_loss'].cpu().numpy())
                        pc_overlap_precision_list.append(v_data['pc_overlap_precision'].cpu().numpy())
                        pc_overlap_recall_list.append(v_data['pc_overlap_recall'].cpu().numpy())
                        pc_overlap_accuracy_list.append(v_data['pc_overlap_accuracy'].cpu().numpy())
                        img_overlap_precision_list.append(v_data['img_overlap_precision'].cpu().numpy())
                        img_overlap_recall_list.append(v_data['img_overlap_recall'].cpu().numpy())
                        img_overlap_accuracy_list.append(v_data['img_overlap_accuracy'].cpu().numpy())

                    x = np.array(loss_list).mean()
                    writer.add_scalar('val_loss/loss', x, global_step=global_step)
                    writer.add_scalar('val_loss/geometric_loss', np.array(loss_geo_list).mean(), global_step=global_step)
                    writer.add_scalar('val_loss/pc_overlap_loss', np.array(loss_pc_overlap_list).mean(), global_step=global_step)
                    writer.add_scalar('val_loss/img_overlap_loss', np.array(loss_img_overlap_list).mean(), global_step=global_step)
                    writer.add_scalar('val_metrics/pc_overlap_precision', np.array(pc_overlap_precision_list).mean(), global_step=global_step)
                    writer.add_scalar('val_metrics/pc_overlap_recall', np.array(pc_overlap_recall_list).mean(), global_step=global_step)
                    writer.add_scalar('val_metrics/pc_overlap_accuracy', np.array(pc_overlap_accuracy_list).mean(), global_step=global_step)
                    writer.add_scalar('val_metrics/img_overlap_precision', np.array(img_overlap_precision_list).mean(), global_step=global_step)
                    writer.add_scalar('val_metrics/img_overlap_recall', np.array(img_overlap_recall_list).mean(), global_step=global_step)
                    writer.add_scalar('val_metrics/img_overlap_accuracy', np.array(img_overlap_accuracy_list).mean(), global_step=global_step)

                    print("Current loss:", x, "Lowest loss:", pre_fine_loss)
                    if x < pre_fine_loss:
                        pre_fine_loss = x if ~np.isnan(x) else pre_fine_loss
                        # filename = "step-%d-loss-%f.pth" % (global_step, pre_fine_loss)
                        # save_path = os.path.join(ckpt_dir, filename)
                        # torch.save(model.state_dict(), save_path)
                    filename = "epoch-%d-loss-%f.pth" % (epoch, x)
                    save_path = os.path.join(ckpt_dir, filename)
                    torch.save(model.state_dict(), save_path)
                    model.train()

            optimizer.zero_grad()

            model(data)

            loss = data['loss']

            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 1)
            optimizer.step()
            writer.add_scalar('train_loss/loss', loss, global_step=global_step)
            writer.add_scalar('train_loss/geometric_loss', data['geometric_loss'], global_step=global_step)
            writer.add_scalar('train_loss/pc_overlap_loss', data['pc_overlap_loss'], global_step=global_step)
            writer.add_scalar('train_loss/img_overlap_loss', data['img_overlap_loss'], global_step=global_step)
            writer.add_scalar('train_metrics/pc_overlap_precision', data['pc_overlap_precision'], global_step=global_step)
            writer.add_scalar('train_metrics/pc_overlap_recall', data['pc_overlap_recall'], global_step=global_step)
            writer.add_scalar('train_metrics/pc_overlap_accuracy', data['pc_overlap_accuracy'], global_step=global_step)
            writer.add_scalar('train_metrics/img_overlap_precision', data['img_overlap_precision'], global_step=global_step)
            writer.add_scalar('train_metrics/img_overlap_recall', data['img_overlap_recall'], global_step=global_step)
            writer.add_scalar('train_metrics/img_overlap_accuracy', data['img_overlap_accuracy'], global_step=global_step)

            global_step += 1
            # torch.cuda.empty_cache()
        print("%d-th epoch end." % (epoch))
        time.sleep(5)
        lr_scheduler.step()


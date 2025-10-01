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
from models import IterModel, MultiHeadModel


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
    sate_dict = torch.load("./checkpoint/geo_feat.pth")
    geo_model.load_state_dict(sate_dict)
    geo_model = geo_model.cuda()
    geo_model.eval()

    model = IterModel(config)
    model = model.cuda()

    model.eval()

    img_x = torch.linspace(0, 128 - 1, 128).unsqueeze(0).expand(40, 128).unsqueeze(0)
    img_y = torch.linspace(0, 40 - 1, 40).unsqueeze(1).expand(40, 128).unsqueeze(0)
    img_xy = torch.cat([img_x, img_y], dim=0)
    img_xy = img_xy.view(2, -1).cuda()

    IR_list = []
    pc_overlap_precision_list = []
    pc_overlap_recall_list = []

    IR_list_1 = []
    IR_list_2 = []
    IR_list_3 = []

    threshold = 3.0

    with torch.no_grad():
        for v_data in tqdm(test_loader):
            geo_model(v_data)
            model(v_data)

            IR_list.append(v_data['matching_ir'].cpu().numpy())
            pc_overlap_precision_list.append(v_data['pc_overlap_precision'].cpu().numpy())
            pc_overlap_recall_list.append(v_data['pc_overlap_recall'].cpu().numpy())

            pc_overlap_logits = v_data['pc_overlap_logits'][0]
            img_overlap_logits = v_data['img_overlap_logits'][0]
            pc_overlap = pc_overlap_logits.argmax(0)
            pc_mask = pc_overlap.bool()
            img_overlap = img_overlap_logits.argmax(0)
            # print(pc_overlap_logits.shape, img_overlap_logits.shape, pc_overlap.shape, img_overlap.shape)

            pc_geo_feat = v_data['pc_geo_feat'][0]
            img_geo_feat = v_data['img_geo_feat'][0]
            point_xy_all = v_data['point_xy_float_all'][0]
            pc_geo_feat = pc_geo_feat[:, pc_mask]
            point_xy = point_xy_all[:, pc_mask].cuda()
            img_geo_feat = img_geo_feat.view(64, -1)

            dist = []
            for i in range((pc_geo_feat.shape[1] // 2000) + 1):
                temp_pc = pc_geo_feat[:, i * 2000: (i + 1) * 2000]
                # dist_temp = 1 - torch.sum(temp_pc.unsqueeze(-1) * img_geo_feat.unsqueeze(-2), dim=0)
                dist_temp = torch.sqrt(torch.sum((temp_pc.unsqueeze(-1) - img_geo_feat.unsqueeze(-2)) ** 2, dim=0))
                dist.append(dist_temp)
            dist = torch.cat(dist, dim=0)
            min_idx = dist.argmin(1)

            pred_xy = img_xy[:, min_idx]
            right = torch.sqrt(torch.sum((pred_xy - point_xy) ** 2, dim=0)) <= threshold
            ir_1 = right.sum() / right.shape[0]
            IR_list_1.append(ir_1.cpu().numpy())

            pc_overlap_in_img = img_overlap[min_idx]
            pc_overlap_in_img = pc_overlap_in_img.bool()
            pred_xy = pred_xy[:, pc_overlap_in_img]
            point_xy = point_xy[:, pc_overlap_in_img]
            right = torch.sqrt(torch.sum((pred_xy - point_xy) ** 2, dim=0)) <= threshold
            ir_2 = right.sum() / right.shape[0]
            IR_list_2.append(ir_2.cpu().numpy())

            print(ir_1, ir_2)


    pc_overlap_precision = np.array(pc_overlap_precision_list).mean()
    pc_overlap_recall = np.array(pc_overlap_recall_list).mean()


    IR = np.array(IR_list).mean()
    IR1 = np.array(IR_list_1).mean()
    IR2 = np.array(IR_list_2).mean()

    print(pc_overlap_precision, pc_overlap_recall, IR, IR1, IR2)



import os
import torch
import torch.utils.data as data
from torchvision import transforms
import numpy as np
from PIL import Image
from multiprocessing import Process
import open3d
import random
import math
import open3d as o3d
import cv2
import struct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.sparse import coo_matrix
import torch_scatter
import time
import sys
from scipy.spatial import cKDTree
sys.path.append("..")
from config import KittiConfiguration
import yaml


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


class KittiCalibHelper:
    def __init__(self, root_path):
        self.root_path = root_path
        self.calib_matrix_dict = self.read_calib_files()

    def read_calib_files(self):
        seq_folders = [name for name in os.listdir(
            os.path.join(self.root_path, 'calib'))]
        calib_matrix_dict = {}
        for seq in seq_folders:
            calib_file_path = os.path.join(
                self.root_path, 'calib', seq, 'calib.txt')
            with open(calib_file_path, 'r') as f:
                for line in f.readlines():
                    seq_int = int(seq)
                    if calib_matrix_dict.get(seq_int) is None:
                        calib_matrix_dict[seq_int] = {}

                    key = line[0:2]
                    mat = np.fromstring(line[4:], sep=' ').reshape(
                        (3, 4)).astype(np.float32)
                    if 'Tr' == key:
                        P = np.identity(4)
                        P[0:3, :] = mat
                        calib_matrix_dict[seq_int][key] = P
                    else:
                        K = mat[0:3, 0:3]
                        calib_matrix_dict[seq_int][key + '_K'] = K
                        fx = K[0, 0]
                        fy = K[1, 1]
                        cx = K[0, 2]
                        cy = K[1, 2]

                        tz = mat[2, 3]
                        tx = (mat[0, 3] - cx * tz) / fx
                        ty = (mat[1, 3] - cy * tz) / fy
                        P = np.identity(4)
                        P[0:3, 3] = np.asarray([tx, ty, tz])
                        calib_matrix_dict[seq_int][key] = P
        return calib_matrix_dict

    def get_matrix(self, seq: int, matrix_key: str):
        return self.calib_matrix_dict[seq][matrix_key]


class FarthestSampler:
    def __init__(self, dim=3):
        self.dim = dim

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=0)

    def sample(self, pts, k):
        farthest_pts = np.zeros((self.dim, k))
        farthest_pts_idx = np.zeros(k, dtype=np.int64)
        init_idx = np.random.randint(len(pts))
        farthest_pts[:, 0] = pts[:, init_idx]
        farthest_pts_idx[0] = init_idx
        distances = self.calc_distances(farthest_pts[:, 0:1], pts)
        for i in range(1, k):
            idx = np.argmax(distances)
            farthest_pts[:, i] = pts[:, idx]
            farthest_pts_idx[i] = idx
            distances = np.minimum(distances, self.calc_distances(farthest_pts[:, i:i + 1], pts))
        return farthest_pts, farthest_pts_idx


class KittiDataset(data.Dataset):
    def __init__(self, config, mode):
        super(KittiDataset, self).__init__()
        self.dataset_root = config.dataset_root
        self.data_color = config.data_color
        self.data_velodyne = config.data_velodyne
        self.mode = mode
        self.num_pt = config.num_pt
        self.img_H = config.cropped_img_H
        self.img_W = config.cropped_img_W
        self.patch_size = config.patch_size

        self.T_amplitude = 10.0
        self.R_amplitude = math.pi
        self.P_Tx_amplitude = self.T_amplitude
        self.P_Ty_amplitude = 0.0
        self.P_Tz_amplitude = self.T_amplitude
        self.P_Rx_amplitude = 0.0
        self.P_Ry_amplitude = self.R_amplitude
        self.P_Rz_amplitude = 0.0

        self.farthest_sampler = FarthestSampler(dim=3)
        self.dataset = self.make_kitti_dataset()
        self.calib_helper = KittiCalibHelper(config.dataset_root)

        self.num_node = config.num_node
        self.config = config

        print("%d samples in %s set..." % (len(self.dataset), mode))

    def make_kitti_dataset(self):
        dataset = []
        if self.mode == 'train':
            seq_list = list([0, 1, 2, 3, 4, 5, 6, 7, 8])
        elif self.mode == 'val' or self.mode == 'test':
            seq_list = [9, 10]
        else:
            raise Exception('Invalid mode...')

        for seq in seq_list:
            img2_folder = os.path.join(self.dataset_root, self.data_color, 'sequences/', '%02d' % seq, 'image_2')
            img3_folder = os.path.join(self.dataset_root, self.data_color, 'sequences/', '%02d' % seq, 'image_3')
            pc_folder = os.path.join(self.dataset_root, self.data_velodyne, 'sequences/', '%02d' % seq, 'voxel0.1-SNr0.6')

            num = round(len(os.listdir(img2_folder)))
            if self.mode == 'val':
                num = 100
            # start_i = 575 #- 1591
            for i in range(num): #start_i,start_i+1): #
                dataset.append((img2_folder, pc_folder, seq, i, 'P2'))
                dataset.append((img3_folder, pc_folder, seq, i, 'P3'))
        return dataset

    def downsample_pc(self, pc_np, labels=None):
        if pc_np.shape[1] >= self.num_pt:
            choice_idx = np.random.choice(pc_np.shape[1], self.num_pt, replace=False)
        else:
            fix_idx = np.asarray(range(pc_np.shape[1]))
            while pc_np.shape[1] + fix_idx.shape[0] < self.num_pt:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(pc_np.shape[1]))), axis=0)
            random_idx = np.random.choice(pc_np.shape[1], self.num_pt - fix_idx.shape[0], replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
        pc_np = pc_np[:, choice_idx]
        if labels is None:
            return pc_np, None
        else:
            labels = labels[choice_idx, 0]
            return pc_np, labels

    def camera_matrix_cropping(self, K: np.ndarray, dx: float, dy: float):
        K_crop = np.copy(K)
        K_crop[0, 2] -= dx
        K_crop[1, 2] -= dy
        return K_crop

    def camera_matrix_scaling(self, K: np.ndarray, s: float):
        K_scale = s * K
        K_scale[2, 2] = 1
        return K_scale

    def augment_img(self, img_np):
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        color_aug = transforms.ColorJitter(
            brightness, contrast, saturation, hue)
        img_color_aug_np = np.array(color_aug(Image.fromarray(img_np)))

        return img_color_aug_np

    def angles2rotation_matrix(self, angles):
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        return R

    def random_RT_amplitude(self):
        T_amplitude = np.random.choice(self.T_list)
        R_amplitude = np.random.choice(self.R_list) / 180 * math.pi
        return R_amplitude, T_amplitude

    def generate_random_transform(self):
        """
        Generate a random transform matrix according to the configuration
        """
        t = [random.uniform(-self.P_Tx_amplitude, self.P_Tx_amplitude),
             random.uniform(-self.P_Ty_amplitude, self.P_Ty_amplitude),
             random.uniform(-self.P_Tz_amplitude, self.P_Tz_amplitude)]
        angles = [random.uniform(-self.P_Rx_amplitude, self.P_Rx_amplitude),
                  random.uniform(-self.P_Ry_amplitude, self.P_Ry_amplitude),
                  random.uniform(-self.P_Rz_amplitude, self.P_Rz_amplitude)]
        rotation_mat = self.angles2rotation_matrix(angles)
        P_random = np.identity(4, dtype=np.float32)
        P_random[0:3, 0:3] = rotation_mat
        P_random[0:3, 3] = t

        return P_random, angles, t

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        cv2.ocl.setUseOpenCL(False)
        cv2.setNumThreads(0)

        img_folder, pc_folder, seq, seq_i, key = self.dataset[index]
        img = np.load(os.path.join(img_folder, '%06d.npy' % seq_i))
        data = np.load(os.path.join(pc_folder, '%06d.npy' % seq_i))
        pc = data[0:3, :]

        # <------ Loading point-wise semantic labels ------>
        # v_f = pc_folder.split("/")[-1]
        # label_file = pc_folder.replace(v_f, "labels") + "/%06d.npy" % (seq_i)
        # labels = np.load(label_file)

        # <------ convert velodyne coordinates to camera coordinates ------>
        P_Tr = np.dot(self.calib_helper.get_matrix(seq, key),
                      self.calib_helper.get_matrix(seq, 'Tr'))
        pc = np.dot(P_Tr[0:3, 0:3], pc) + P_Tr[0:3, 3:]

        # <------ matrix: camera intrinsics ------>
        K = self.calib_helper.get_matrix(seq, key + '_K')

        # <------ sampling a specified number of points ------>
        # pc, labels = self.downsample_pc(pc, labels)
        pc, _ = self.downsample_pc(pc)

        # # <------ crop the useless pixels (the sky) ------>
        # print(img.shape)
        # img_crop_dy = 40
        # img = img[img_crop_dy:, :, :]
        # K = self.camera_matrix_cropping(K, dx=0, dy=img_crop_dy)

        img = cv2.resize(img,
                         (int(round(img.shape[1] * 0.5)),
                          int(round((img.shape[0] * 0.5)))),
                         interpolation=cv2.INTER_LINEAR)
        K = self.camera_matrix_scaling(K, 0.5)

        # print(img.shape)
        # <------ cropping a random patch from image ------>
        if self.mode == 'train':
            img_crop_dx = random.randint(0, img.shape[1] - self.img_W)
            img_crop_dy = random.randint(0, img.shape[0] - self.img_H)
        else:
            img_crop_dx = int((img.shape[1] - self.img_W) / 2)
            img_crop_dy = int((img.shape[0] - self.img_H) / 2)
        img = img[img_crop_dy:img_crop_dy + self.img_H, img_crop_dx:img_crop_dx + self.img_W, :]

        K = self.camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)

        # <------ solve the PnP problem at 1/4 scale of the input image ------>
        K = self.camera_matrix_scaling(K, 0.25)

        if self.mode == 'train':
            img = self.augment_img(img)

        pc_in_cam_space = pc
        pc_ = np.dot(K, pc)
        pc_mask = np.zeros((1, pc.shape[1]), dtype=np.float32)
        pc_[0:2, :] = pc_[0:2, :] / pc_[2:, :]
        xy = np.round(pc_[0:2, :])
        is_in_picture = (xy[0, :] >= 0) & (xy[0, :] <= (self.img_W*0.25 - 1)) & (xy[1, :] >= 0) & \
                        (xy[1, :] <= (self.img_H*0.25-1)) & (pc_[2, :] > 0)

        # in_pc = pc[:, is_in_picture]
        # src_pc = o3d.geometry.PointCloud()
        # src_pc.points = o3d.utility.Vector3dVector(in_pc.T)
        # src_pc.paint_uniform_color([255/255.0, 127/255.0, 14/255.0])
        #
        # out_pc = pc[:, ~is_in_picture]
        # tgt_pc = o3d.geometry.PointCloud()
        # tgt_pc.points = o3d.utility.Vector3dVector(out_pc.T)
        # tgt_pc.paint_uniform_color([31/255.0, 119/255.0, 180/255.0])
        #
        # o3d.visualization.draw_geometries([src_pc, tgt_pc])

        pc_mask[:, is_in_picture] = 1.

        xy2 = xy[:, is_in_picture]
        img_mask = coo_matrix((np.ones_like(xy2[0, :]), (xy2[1, :], xy2[0, :])),
                              shape=(int(self.img_H * 0.25), int(self.img_W * 0.25))).toarray()
        img_mask = np.array(img_mask)

        img_mask[img_mask > 0] = 1

        point_num_circle_loss = 512
        pc_idx_for_circle_loss = np.where(is_in_picture)[0]
        idx = np.random.permutation(len(pc_idx_for_circle_loss))[0: point_num_circle_loss]
        pc_idx_for_circle_loss = pc_idx_for_circle_loss[idx]
        pc_xy_float_for_circle_loss = pc_[0:2, pc_idx_for_circle_loss]
        pc_xy_int_for_circle_loss = np.round(pc_xy_float_for_circle_loss).astype(np.int64)
        # print(pc_xy_projected_float[0,:].max(), pc_xy_projected_float[0,:].min(), pc_xy_projected_float[1,:].max(), pc_xy_projected_float[1,:].min())

        # <------transform the point cloud------>
        # pc = pc - pc.mean(axis=1, keepdim=True)
        P, angles, t = self.generate_random_transform()
        angles = np.array(angles)
        t = np.array(t)
        pc = np.dot(P[0:3, 0:3], pc) + P[0:3, 3:]

        # <------sample some node to extract features------>
        node_np, _ = self.farthest_sampler.sample(pc[:, np.random.choice(pc.shape[1],\
                                                  self.num_node * 8, replace=False)], k=self.num_node)

        if self.config.use_gnn_embedding:
            kdtree = cKDTree(pc.T)
            _, I = kdtree.query(pc.T, k=16)
        else:
            kdtree = cKDTree(node_np.T)
            _, I = kdtree.query(pc.T, k=1)

        # <======================================================>
        # if self.mode == 'train':
        #     pass
        # else:
        #     pc_mask_raw = pc_mask_raw.astype(np.bool_)
        #     img_label = labels[pc_mask_raw[0]]
        #     img_xy = xy_raw[:, pc_mask_raw[0]]
        #     labels_inv = np.vectorize(self.learning_map_inv.get)(img_label)
        #     label_colors = []
        #     print(labels_inv)
        #     for i in labels_inv[:]:
        #         label_colors.append([self.color_map[i]])
        #
        #     label_colors = np.concatenate(label_colors, axis=0)
        #
        #     xy = img_xy
        #     in_pc = xy
        #     print(in_pc.shape, img.shape, label_colors.shape)
        #     in_colors = label_colors
        #     # in_pc = in_pc[mm,:]
        #     for i in range(in_pc.shape[1]):
        #         img[int(in_pc[1, i]), int(in_pc[0, i]), :] = in_colors[i, :]
        #     cv2.namedWindow('semantic_img', cv2.WINDOW_NORMAL)
        #     cv2.imshow('semantic_img', img)
        #     key = cv2.waitKey(0)
        # <======================================================>
        # src_pc = o3d.geometry.PointCloud()
        # src_pc.points = o3d.utility.Vector3dVector(pc.T)
        #
        # o3d.visualization.draw_geometries([src_pc])

        return {
                'img': torch.from_numpy(img.astype(np.float32) / 255.).permute(2, 0, 1).contiguous(),
                'pc': torch.from_numpy(pc.astype(np.float32)),
                'K': torch.from_numpy(K.astype(np.float32)),
                # 'P': torch.from_numpy(P.astype(np.float32)),
                'P': torch.from_numpy(np.linalg.inv(P).astype(np.float32)),

                'img_mask': torch.from_numpy(img_mask).long(),
                'pc_mask': torch.from_numpy(is_in_picture).long(),
                'pc_idx_for_circle_loss': torch.from_numpy(pc_idx_for_circle_loss).long(),
                'pc_xy_float_for_circle_loss': torch.from_numpy(pc_xy_float_for_circle_loss).float(),
                'pc_xy_int_for_circle_loss': torch.from_numpy(pc_xy_int_for_circle_loss).long(),
                # 'point2pixel_xy_int': torch.from_numpy(xy).long(),
                # 'point2pixel_z_': torch.from_numpy(pc_[0:2, :]).float(),
                'pc_in_cam_space': torch.from_numpy(pc_in_cam_space).float(),

                # 'labels': torch.from_numpy(labels).long(),

                'pt2node': torch.from_numpy(I).long(),
                'node': torch.from_numpy(node_np).float(),

                'angles': torch.from_numpy(angles),
                'translation': torch.from_numpy(t)
               }


def func_test(dataset, idx_list):
    for i in idx_list:
        dataset[i]


def main_test(dataset):
    thread_num = 12
    idx_list_list = []
    for i in range(thread_num):
        idx_list_list.append([])
    kitti_threads = []
    for i in range(len(dataset)):
        thread_seq_list = [i]
        idx_list_list[int(i % thread_num)].append(i)

    for i in range(thread_num):
        kitti_threads.append(Process(target=func_test,
                                     args=(dataset,
                                           idx_list_list[i])))

    for thread in kitti_threads:
        thread.start()

    for thread in kitti_threads:
        thread.join()


# <------ debug ------>
if __name__ == '__main__':
    config = KittiConfiguration("/home/yao/workspace/I2P/kitti")
    dataset = KittiDataset(config, 'test')
    dataset[2156]
    # for i in range(len(dataset)):
    #     print(i)
    #     dataset[i]

    # main_test(dataset)
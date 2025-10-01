import numpy as np
import math
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NuScenesConfiguration:
    """
    The configuration to train on NuScenes dataset
    """
    def __init__(self, data_root=None):
        print("NuScenes Dataset Configuration...")
        # <----------- dataset configuration ---------->
        self.dataset_root = '/home/yao/workspace/I2P/nuscenes2/' if data_root is None else data_root
        self.num_pt = 40960
        self.P_Tx_amplitude = 10.0
        self.P_Ty_amplitude = 0.0
        self.P_Tz_amplitude = 10.0
        self.P_Rx_amplitude = 0.0
        self.P_Ry_amplitude = math.pi
        self.P_Rz_amplitude = 0.0
        self.cropped_img_H = 160
        self.cropped_img_W = 320
        # <-------------------------------------------->

        # <--------- training and testing configuration ---------->
        self.seed = 2023

        # <--------- coarse matching ------->
        self.train_batch_size = 8
        self.val_batch_size = 8
        self.val_interval = 1000
        self.epoch = 30
        self.lr = 0.001
        self.resume = False
        self.checkpoint = None

        self.num_workers = 16

        # <-------- optimizer -------->
        self.optimizer = "ADAM"  # "SGD" or "ADAM"
        self.momentum = 0.98
        self.weight_decay = 1e-06

        # <-------- lr_scheduler -------->
        self.lr_scheduler = "StepLR"
        self.scheduler_gamma = 0.6
        self.step_size = 2

        self.logdir = "log/"
        self.ckpt_dir = "checkpoint/"

        # <-----------model configuration---------->
        # <---------image ViT-------->
        self.image_H = int(self.cropped_img_H * 0.25)
        self.image_W = int(self.cropped_img_W * 0.25)
        self.patch_size = 8
        self.use_resnet_embedding = True

        self.embed_dim = 64
        self.mlp_dim = 1024

        self.embed_dropout = 0.1
        self.mlp_dropout = 0.1
        self.attention_dropout = 0.1

        self.num_sa_layer = 3
        self.num_head = 8
        # <---------point ViT-------->
        self.use_gnn_embedding = False
        self.point_feat_dim = 3  # coordinate + intensity + normal

        if self.use_gnn_embedding:
            self.num_node = 256
            self.edge_conv_dim = 64
        else:
            self.num_node = 1280
            self.num_proxy = 256
        # <---------Coarse I2P-------->
        self.num_ca_layer_coarse = 6
        self.sinkhorn_iters = 100
        self.coarse_matching_thres = 0.01

        # <------ Fine I2P ------>
        self.pt_sample_num = 65
        self.fine_dist_theshold = 1
        self.topk_proxy = 3
        self.pixel_positional_embedding = True
        self.fine_loss_weight = 0.5

        self.img_fuse_res_num = 2
        self.node_fuse_res_num = 2
        self.pt_head_res_num = 3
        self.linear_attention_num = 4
        self.LA_head_num = 8

        # <------- Agnet -------->
        self.is_6_DoF = False
        self.EXPERT_MODE = 'steady'
        self.action_num = 10

        self.r_steps = np.array([-62.5, -12.5, -2.5, -0.5, -0.1, 0.0, 0.1, 0.5, 2.5, 12.5, 62.5]) * math.pi / 180
        self.t_steps = np.array([-8.1, -2.7, -0.9, -0.3, -0.1, 0.0, 0.1, 0.3, 0.9, 2.7, 8.1])
        self.r_steps = torch.from_numpy(self.r_steps).to(DEVICE)
        self.t_steps = torch.from_numpy(self.t_steps).to(DEVICE)
        self.num_steps = self.r_steps.shape[0]

        self.num_trajectory = 4

        self.GAMMA = 0.99
        self.GAE_LAMBDA = 0.95
        self.alpha = 1.0
        self.CLIP_EPS = 0.2
        self.W_VALUE = 0.3
        self.W_ENTROPY = 1e-3



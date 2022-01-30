import os
import torch
from random import seed

import numpy as np
import matplotlib.pyplot as plt

from utils.utils import multi_acc, oht_to_scalar, colorize_mask
from utils.data_util import tma_12_palette
from torch.utils.data import DataLoader
from pixel_features_dataset import PixelFeaturesDataset
from networks.pixel_classifier import PixelClassifier
from load_networks import load_stylegan2_ada, get_upsamplers
from utils.utils import latent_to_image


chosen_seed = 0
seed(chosen_seed)
np.random.seed(chosen_seed)
torch.manual_seed(chosen_seed)
torch.cuda.manual_seed_all(chosen_seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def generate_one_image_mask_pair(idx, g_all, upsamplers):
    latent_np = np.random.RandomState(idx).randn(1, g_all.z_dim)  # shape [1, 512]
    img, upsampled_featmap = latent_to_image(g_all, upsamplers, latent_np, is_w_latent=False,
                                             dim=args['featuremaps_dim'][1], truncation_psi=args['truncation'],
                                             return_upsampled_featuremaps=True)

    # Get mask


def main():
    g_all, avg_latent = load_stylegan2_ada(args)
    upsamplers = get_upsamplers(args)

    for i in range(100):
        generate_one_image_mask_pair(g_all, upsamplers)



if __name__ == "__main__":
    training_run_dir = "/home/cougarnet.uh.edu/srizvi7/Desktop/Histopathology_Dataset_GAN/training-runs/"
    training_run = "0009-TMA_Arteriole_stylegan2_ada"
    SAVE_PATH = "/data/syed/hdgan/TMA_1024_Arteriole2/TMA_1024_Arteriole2_generating_dataset"
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    args = {
        "average_latent": "",
        "batch_size": 128,
        "category": "TMA",
        "data_dir": "/home/cougarnet.uh.edu/srizvi7/Desktop/Histopathology_Dataset_GAN/generated_datasets/TMA_1024_Arteriole2",
        "deeplab_res": 1024,
        "early_stopping_patience": 8,
        "epochs": 50,
        "experiment_dir": "training-runs/00010-TMA_Arteriole_stylegan2_ada",
        "featuremaps_dim": [1024, 1024, 6080],
        "G_kwargs": {
        "class_name": "networks.stylegan2_ada.Generator",
        "z_dim": 512,
        "w_dim": 512,
        "mapping_kwargs": {
          "num_layers": 2
        },
        "synthesis_kwargs": {
          "channel_base": 32768,
          "channel_max": 512,
          "num_fp16_res": 4,
          "conv_clamp": 256
        }
        },
        "log_frequency": 30000,
        "max_training": 24,
        "model_num": 10,
        "num_classes": 7,
        "pixel_classifier_lr": 0.001,
        "pixel_feat_save_dir": "/data/syed/hdgan/TMA_1024_Arteriole2",
        "resolution": 1024,
        "stylegan_checkpoint": "/data/syed/stylegan2-ada-training/00000--mirror-auto4-Arteriole/network-snapshot-006000.pkl",
        "stylegan_ver": "2_ADA",
        "testing_data_number_class": 1,
        "testing_path": "",
        "truncation": 0.7,
        "upsample_mode":"bilinear",
        "val_fold_idx": 0,
        "val_frequency": 60000
    }

    main()
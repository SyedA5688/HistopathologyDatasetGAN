import os
import json
import argparse


import torch
import numpy as np

from load_networks import load_stylegan2_ada, get_upsamplers
from utils.utils import latent_to_image

np.random.seed(0)
torch.manual_seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_pixel_classifier_compressed_dataset(data_path, g_all, upsamplers, save_path):
    avg_latent_np = np.load(os.path.join(data_path, "avg_latent_stylegan2.npy"))  # shape (18, 512)
    latent_np = np.load(os.path.join(data_path, "latent_stylegan2.npy"))

    # Create numpy save files for each image in whole dataset
    for i in range(len(latent_np) + 1):  # Plus one because average w latent is saved separately
        print("Processing image", i)
        if i == 0:
            dataset_split_helper(avg_latent_np, i, g_all, upsamplers, save_path,
                                 save_name="pixel_level_feat_img_" + str(i) + ".npy",
                                 is_w_latent=True)
        else:
            dataset_split_helper(latent_np, i, g_all, upsamplers, save_path,
                                 save_name="pixel_level_feat_img_" + str(i) + ".npy")


def dataset_split_helper(latent_np, img_idx, g_all, upsamplers, save_path, save_name, is_w_latent=False):
    if is_w_latent:
        latent = torch.from_numpy(latent_np).to(device).unsqueeze(0)  # shape [1, 18, 512]
        img, upsampled_featmap = latent_to_image(g_all, upsamplers, latent, is_w_latent=True,
                                                 dim=args['featuremaps_dim'][1], truncation_psi=args['truncation'],
                                                 return_upsampled_featuremaps=True)
    else:
        latent = torch.from_numpy(latent_np[img_idx - 1]).to(device).unsqueeze(0)  # shape [1, 512]
        img, upsampled_featmap = latent_to_image(g_all, upsamplers, latent, is_w_latent=False,
                                                 dim=args['featuremaps_dim'][1], truncation_psi=args['truncation'],
                                                 return_upsampled_featuremaps=True)
    b, ch, h, w = upsampled_featmap.shape  # [1, 6080, 1024, 1024]

    pixel_features_list = []
    for row in range(h):
        for col in range(w):
            pixel_features_list.append(upsampled_featmap[:, :, row, col])  # Append [1, 6080]

    pixel_features_list = torch.cat(pixel_features_list, dim=0)  # [1024*1024, 6080]

    print(pixel_features_list.shape)
    np.save(os.path.join(save_path, save_name), pixel_features_list)


""" Debugging if dataset is loading pixel features and ground truth labels in correct order
- Debug to line after img, upsampled_featmap = latent_to_img()

import os
import numpy as np
reloaded_pixel_data = np.load(os.path.join("/data/syed/hdgan/TMA_1024_Arteriole2/pixel_level_feat_img_" + str(0) + ".npy"), mmap_mode='r')
reloaded_pixel_data.shape
"""


def main():
    # Load StyleGAN checkpoint
    g_all, avg_latent = load_stylegan2_ada(args)
    upsamplers = get_upsamplers(args)

    # Create dataset
    create_pixel_classifier_compressed_dataset(args["dataset_save_dir"], g_all, upsamplers, save_path=args["dataset_save_dir"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default="/home/cougarnet.uh.edu/srizvi7/Desktop/Histopathology_Dataset_GAN/experiments/TMA_Arteriole_stylegan2_ada.json")
    opts = parser.parse_args()
    args = json.load(open(opts.experiment, 'r'))

    main()

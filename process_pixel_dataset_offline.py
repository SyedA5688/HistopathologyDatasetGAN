import os
import json
import argparse
import functools

import torch
import numpy as np

from load_networks import load_stylegan2_ada, get_upsamplers
from utils.utils import latent_to_image

np.random.seed(0)
torch.manual_seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def check_img_idx_in_filename_tma4096(all_files, img_idx):
    for file in all_files:
        if file == "image_{}.png".format(img_idx):
            return True
    return False


def create_pixel_classifier_compressed_dataset(data_path, g_all, upsamplers, save_path):
    #--- First image is averaged latent, pass as w latent ---#
    avg_latent_np = np.load(os.path.join(data_path, "avg_latent_stylegan2.npy"))  # shape (18, 512)
    dataset_split_helper(avg_latent_np, 0, g_all, upsamplers, save_path,
                         save_name="pixel_level_feat_img_{}.npy".format(0), is_w_latent=True)

    #--- Get all image filenames in dataset save directory ---#
    all_files = os.listdir(args["dataset_save_dir"])
    all_files = [file for file in all_files if ".png" in file]
    latent_np = np.load(os.path.join(data_path, "latent_stylegan2.npy"))

    #--- For each index, check if there is a corresponding image. If so, call helper function ---#
    for img_name_idx in range(0, len(latent_np) + 1):
        if check_img_idx_in_filename_tma4096(all_files, img_name_idx):
            print("Processing image", img_name_idx)
            dataset_split_helper(latent_np, img_name_idx, g_all, upsamplers, save_path,
                                 save_name="pixel_level_feat_img_{}.npy".format(img_name_idx))


def dataset_split_helper(latent_np, img_name_idx, g_all, upsamplers, save_path, save_name, is_w_latent=False):
    latent = torch.from_numpy(latent_np).to(device).unsqueeze(0) if is_w_latent \
        else torch.from_numpy(latent_np[img_name_idx - 1]).to(device).unsqueeze(0)
    # latent is shape [1, 18, 512] if w_latent is True else [1, 512]

    #--- Get image and upsampled feature maps ---#
    img, upsampled_featmap = latent_to_image(g_all, upsamplers, latent, is_w_latent=is_w_latent,
                                             dim=args['featuremaps_dim'][1], truncation_psi=args['truncation'],
                                             return_upsampled_featuremaps=True, device=device)

    #--- Make list of pixel features for image, save as np array in dataset save path ---#
    b, ch, h, w = upsampled_featmap.shape  # [1, ch, 4096, 4096]
    pixel_features_list = []
    for row in range(h):
        for col in range(w):
            pixel_features_list.append(upsampled_featmap[:, :, row, col])  # Append [1, ch]

    pixel_features_list = torch.cat(pixel_features_list, dim=0).cpu().numpy()  # [4096*4096, ch]
    print("Image {} shape:".format(img_name_idx), pixel_features_list.shape, "\n")
    np.save(os.path.join(save_path, save_name), pixel_features_list)


def main():
    #-- Load StyleGAN checkpoint --#
    g_all, avg_latent = load_stylegan2_ada(args)
    g_all.to(device)

    #-- Line for inference on CPU --#
    g_all = g_all.float()
    g_all.forward = functools.partial(g_all.forward, force_fp32=True)
    upsamplers = get_upsamplers(args)

    #-- Create dataset --#
    save_path = os.path.join(args["dataset_save_dir"], "pixel_3_block_features_dataset")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    create_pixel_classifier_compressed_dataset(args["dataset_save_dir"], g_all, upsamplers, save_path=save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default="/path/to/Histopathology_Dataset_GAN/experiments/TMA_4096_tile.json")
    opts = parser.parse_args()
    args = json.load(open(opts.experiment, 'r'))

    main()

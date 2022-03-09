import os
import json
import argparse
import functools

import torch
import numpy as np
from sklearn.decomposition import IncrementalPCA

from load_networks import load_stylegan2_ada, get_upsamplers
from utils.utils import latent_to_image

np.random.seed(0)
torch.manual_seed(0)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


def check_img_idx_in_filename_tma4096(all_files, img_idx):
    for file in all_files:
        if file == "image_{}.png".format(img_idx):
            return True
    return False


def center_and_scale(pixel_feature_list):
    pixel_feat_mean = np.mean(pixel_feature_list, axis=0, dtype=np.float64)
    pixel_feat_std = np.std(pixel_feature_list, axis=0, dtype=np.float64)
    return (pixel_feature_list - pixel_feat_mean) / pixel_feat_std


def create_pixel_classifier_compressed_dataset(data_path, g_all, upsamplers, save_path):
    # First image is averaged latent, pass as w latent
    avg_latent_np = np.load(os.path.join(data_path, "avg_latent_stylegan2.npy"))  # shape (18, 512)
    latent_np = np.load(os.path.join(data_path, "latent_stylegan2.npy"))

    # For TMA_4096, chose 65 out of 500 latents. Only process indices that are in an image name
    all_files = os.listdir(args["dataset_save_dir"])
    all_files = [file for file in all_files if ".png" in file]
    ipca = IncrementalPCA(n_components=100)

    for img_name_idx in range(0, 38):  # len(latent_np) + 1
        if check_img_idx_in_filename_tma4096(all_files, img_name_idx):
            print("Fitting on image", img_name_idx)
            is_w_latent = img_name_idx == 0
            latent = torch.from_numpy(avg_latent_np).to(device).unsqueeze(0) if is_w_latent \
                else torch.from_numpy(latent_np[img_name_idx - 1]).to(device).unsqueeze(0)
            # latent is shape [1, 18, 512] if w_latent is True else [1, 512]

            img, upsampled_featmap = latent_to_image(g_all, upsamplers, latent, is_w_latent=is_w_latent,
                                                     dim=args['featuremaps_dim'][1], truncation_psi=args['truncation'],
                                                     return_upsampled_featuremaps=True, device=device)

            b, ch, h, w = upsampled_featmap.shape  # [1, 6128, 4096, 4096]
            pixel_features_list = []
            for row in range(h):
                for col in range(w):
                    pixel_features_list.append(upsampled_featmap[:, :, row, col])  # Append [1, 6128]

            pixel_features_list = torch.cat(pixel_features_list, dim=0).numpy()  # [4096*4096, 6128]
            pixel_features_list = center_and_scale(pixel_features_list)

            # Fit this chunk of the data. Calculating PCA incrementally
            for feat_idx in range(0, len(pixel_features_list), 262144):
                ipca.partial_fit(pixel_features_list[feat_idx:feat_idx + 262144])

    print("\n\nFinished fitting PCA incrementally. Transforming and saving data now")
    for img_name_idx in range(0, 92):  # len(latent_np) + 1
        if check_img_idx_in_filename_tma4096(all_files, img_name_idx):
            print("Processing image", img_name_idx)
            is_w_latent = img_name_idx == 0
            latent = torch.from_numpy(avg_latent_np).to(device).unsqueeze(0) if is_w_latent \
                else torch.from_numpy(latent_np[img_name_idx - 1]).to(device).unsqueeze(0)
            # latent is shape [1, 18, 512] if w_latent is True else [1, 512]

            img, upsampled_featmap = latent_to_image(g_all, upsamplers, latent, is_w_latent=is_w_latent,
                                                     dim=args['featuremaps_dim'][1], truncation_psi=args['truncation'],
                                                     return_upsampled_featuremaps=True, device=device)

            b, ch, h, w = upsampled_featmap.shape  # [1, 6128, 4096, 4096]
            pixel_features_list = []
            for row in range(h):
                for col in range(w):
                    pixel_features_list.append(upsampled_featmap[:, :, row, col])  # Append [1, 6128]

            pixel_features_list = torch.cat(pixel_features_list, dim=0).numpy()  # [4096*4096, 6128]
            transformed_pca_features_list = []
            for feat_idx in range(0, len(pixel_features_list), 262144):
                pixel_pca_features = ipca.transform(pixel_features_list[feat_idx:feat_idx + 262144])
                transformed_pca_features_list.append(pixel_pca_features)
            transformed_pca_features = np.concatenate(transformed_pca_features_list, axis=0)
            np.save(os.path.join(save_path, "pixel_level_feat_PCA_img_{}.npy".format(img_name_idx)), transformed_pca_features)


def main():
    # Load StyleGAN checkpoint
    g_all, avg_latent = load_stylegan2_ada(args)
    g_all.to(device)

    # Line for inference on CPU
    g_all = g_all.float()
    g_all.forward = functools.partial(g_all.forward, force_fp32=True)
    upsamplers = get_upsamplers(args)

    # Create dataset
    save_path = os.path.join(args["dataset_save_dir"], "6block_features_PCA_incremental")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    create_pixel_classifier_compressed_dataset(args["dataset_save_dir"], g_all, upsamplers, save_path=save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default="/home/srizvi7/Desktop/Histopathology_Dataset_GAN/experiments/TMA_4096_tile.json")
    # /home/cougarnet.uh.edu/srizvi7/Desktop/Histopathology_Dataset_GAN/experiments/TMA_4096_tile.json
    opts = parser.parse_args()
    args = json.load(open(opts.experiment, 'r'))
    args["dataset_save_dir"] = "/project/hnguyen2/syed/hdgan/TMA_4096_snapshot2600"
    args["stylegan_checkpoint"] = "/home/srizvi7/Desktop/Histopathology_Dataset_GAN/network-snapshot-002600.pkl"

    main()

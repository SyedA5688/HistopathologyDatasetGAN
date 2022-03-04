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
# device = 'cpu'


def check_img_idx_in_filename_tma4096(all_files, img_idx):
    for file in all_files:
        if file == "image_{}.png".format(img_idx):
            return True
    return False


def create_pixel_classifier_compressed_dataset(data_path, g_all, upsamplers, save_path):
    # First image is averaged latent, pass as w latent
    avg_latent_np = np.load(os.path.join(data_path, "avg_latent_stylegan2.npy"))  # shape (18, 512)
    dataset_split_helper(avg_latent_np, 0, g_all, upsamplers, save_path,
                         save_name="pixel_level_feat_img_{}.npy".format(0), is_w_latent=True)

    # For TMA_4096, chose 65 out of 500 latents. Only process indices that are in an image name
    all_files = os.listdir(args["dataset_save_dir"])
    all_files = [file for file in all_files if ".png" in file]

    latent_np = np.load(os.path.join(data_path, "latent_stylegan2.npy"))
    # for img_name_idx in range(1, len(latent_np) + 1):
    for img_name_idx in range(200, len(latent_np) + 1):  # Already did up to idx 199
        if check_img_idx_in_filename_tma4096(all_files, img_name_idx):
            print("Processing image", img_name_idx)
            dataset_split_helper(latent_np, img_name_idx, g_all, upsamplers, save_path,
                                 save_name="pixel_level_feat_img_{}.npy".format(img_name_idx))


def dataset_split_helper(latent_np, img_name_idx, g_all, upsamplers, save_path, save_name, is_w_latent=False):
    latent = torch.from_numpy(latent_np).to(device).unsqueeze(0) if is_w_latent \
        else torch.from_numpy(latent_np[img_name_idx - 1]).to(device).unsqueeze(0)
    # latent is shape [1, 18, 512] if w_latent is True else [1, 512]

    img, upsampled_featmap = latent_to_image(g_all, upsamplers, latent, is_w_latent=is_w_latent,
                                             dim=args['featuremaps_dim'][1], truncation_psi=args['truncation'],
                                             return_upsampled_featuremaps=True, device=device)

    # print("Upsampled featuremap shape:", upsampled_featmap.shape)
    b, ch, h, w = upsampled_featmap.shape  # [1, 1008, 4096, 4096]

    pixel_features_list = []
    for row in range(h):
        for col in range(w):
            pixel_features_list.append(upsampled_featmap[:, :, row, col])  # Append [1, 1008]

    pixel_features_list = torch.cat(pixel_features_list, dim=0).cpu().numpy()  # [4096*4096, 1008]
    print("Image {} shape:".format(img_name_idx), pixel_features_list.shape, "\n")
    np.save(os.path.join(save_path, save_name), pixel_features_list)

    # pixel_features_list_PCA = PCA(n_components=100).fit_transform(pixel_features_list)  ToDo: Check PCA feasibility
    # np.save(os.path.join(save_path, "pixel_level_feat_PCA_img_{}.npy".format(img_idx)), pixel_features_list_PCA)


""" 
import numpy as np
from sklearn.decomposition import PCA
import os
pixel_level_feat_img0 = np.load("/data/syed/hdgan/TMA_4096_snapshot2600/pixel_features_dataset/pixel_level_feat_img_0.npy")

pixel_feat_mean = np.mean(pixel_level_feat_img0, axis=0)
pixel_feat_std = np.std(pixel_level_feat_img0, axis=0)
pixel_feat_shifted_scaled = (pixel_level_feat_img0 - pixel_feat_mean) / pixel_feat_std
pixel_feat_shifted_scaled_subset = pixel_feat_shifted_scaled[4096*2000+2000:4096*2000+4000:20]
pixel_features_list_PCA = PCA(n_components=100).fit_transform(pixel_feat_shifted_scaled_subset)
 np.save("/data/syed/hdgan/TMA_4096_snapshot2600/pixel_features_dataset/pixel_level_feat_PCA_img_0.npy", pixel_features_list_PCA)


# Loading corresponding mask
mask = np.load(os.path.join(args["dataset_save_dir"], "image_{}_mask.npy".format(img_idx)))
pixel_ground_truths = []
for row in range(h):
    for col in range(w):
        pixel_ground_truths.append(mask[row, col])




import os
import matplotlib.pyplot as plt
import numpy as np
pixel_level_feat_img0_PCA = np.load("/data/syed/hdgan/TMA_4096_snapshot2600/pixel_features_dataset/pixel_level_feat_PCA_img_0.npy")
mask = np.load("/data/syed/hdgan/TMA_4096_snapshot2600/image_{}_mask.npy".format(0))

pixel_ground_truths = []
for row in range(4096):
    for col in range(4096):
        pixel_ground_truths.append(mask[row, col])
        
# Plotting
plt.figure(figsize=(8,6))
plt.scatter(pixel_level_feat_img0_PCA[:,0], pixel_level_feat_img0_PCA[:,1], c=pixel_ground_truths[4096*2000+2000:4096*2000+4000:20])
plt.title("TMA 4096 PCA Reduced 100 Pixels (First Two Components Visualized)")
plt.xlabel("First Component")
plt.ylabel("Second Component")
plt.savefig("/home/cougarnet.uh.edu/srizvi7/Desktop/Histopathology_Dataset_GAN/pca_tma4096_100pixels_test.png", bbox_inches="tight", facecolor="white")
plt.show()





plt.figure(figsize=(8,6))
plt.scatter(pixel_level_feat_img0_PCA[:,0], pixel_level_feat_img0_PCA[:,1], c=pixel_ground_truths)
plt.show()
    

Debugging if dataset is loading pixel features and ground truth labels in correct order
- Debug to line after img, upsampled_featmap = latent_to_img()

import os
import numpy as np
reloaded_pixel_data = np.load(os.path.join("/data/syed/hdgan/TMA_1024_Arteriole2/pixel_level_feat_img_" + str(0) + ".npy"), mmap_mode='r')
reloaded_pixel_data.shape
"""


def main():
    # Load StyleGAN checkpoint
    g_all, avg_latent = load_stylegan2_ada(args)
    g_all.to(device)

    # Line for inference on CPU
    g_all = g_all.float()
    g_all.forward = functools.partial(g_all.forward, force_fp32=True)
    upsamplers = get_upsamplers(args)

    # Create dataset
    save_path = os.path.join(args["dataset_save_dir"], "pixel_features_dataset")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    create_pixel_classifier_compressed_dataset(args["dataset_save_dir"], g_all, upsamplers, save_path=save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default="/home/cougarnet.uh.edu/srizvi7/Desktop/Histopathology_Dataset_GAN/experiments/TMA_4096_tile.json")
    opts = parser.parse_args()
    args = json.load(open(opts.experiment, 'r'))

    main()

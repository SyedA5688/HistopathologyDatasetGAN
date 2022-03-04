import os
import numpy as np
from sklearn.decomposition import PCA


def check_img_idx_in_filename_tma4096(all_files, img_idx):
    for file in all_files:
        if file == "image_{}.png".format(img_idx):
            return True
    return False


all_files = os.listdir("/data/syed/hdgan/TMA_4096_snapshot2600/")
all_files = [file for file in all_files if ".png" in file]


for idx in range(500):
    if check_img_idx_in_filename_tma4096(all_files, idx):
        print("Processing image", idx)
        pixel_level_feat_img = np.load("/data/syed/hdgan/TMA_4096_snapshot2600/pixel_features_dataset/pixel_level_feat_img_{}.npy".format(idx)).astype(np.float64)

        pixel_feat_mean = np.mean(pixel_level_feat_img, axis=0, dtype=np.float64)
        pixel_feat_std = np.std(pixel_level_feat_img, axis=0, dtype=np.float64)
        pixel_feat_shifted_scaled = (pixel_level_feat_img - pixel_feat_mean) / pixel_feat_std

        # By setting n_components to a percent, this is percent of variance we want explained by our principal components.
        pixel_features_list_PCA = PCA(n_components=0.95).fit_transform(pixel_feat_shifted_scaled)
        np.save("/data/syed/hdgan/TMA_4096_snapshot2600/pixel_features_PCA_dataset/pixel_level_feat_PCA_img_{}.npy".format(idx), pixel_features_list_PCA)

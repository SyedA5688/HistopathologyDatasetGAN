import os
import math
from welford import Welford
import numpy as np

data_path = "/data/syed/hdgan/TMA_4096_snapshot2600/pixel_4_block_features_dataset"
all_files = os.listdir(data_path)
all_files = [file for file in all_files if ".npy" in file]

dataset_means = np.zeros((240,), dtype=np.float64)
dataset_stds = np.zeros((240,), dtype=np.float64)
total_size = 0.
first_moment_total = np.zeros((240,), dtype=np.float64)
# second_moment_total = np.zeros((496,), dtype=np.float64)

w = Welford()


for idx, filename in enumerate(all_files[0:16]):  # Calculate on training set only
    print("Processing: ", filename)
    pixel_data = np.load(os.path.join(data_path, filename)).astype(np.float64)
    # Shape (nsamples, nfeatures), dtype=np.float16

    total_size += len(pixel_data)
    first_moment_total += np.sum(pixel_data, axis=0, dtype=np.float64)
    # second_moment_total += np.sum(np.square(pixel_data), axis=0)

    for row in range(len(pixel_data)):
        w.add(pixel_data[row])


dataset_means = first_moment_total / total_size
dataset_stds = np.sqrt(w.var_p)
# dataset_stds = np.sqrt(((total_size * second_moment_total) - (first_moment_total * first_moment_total)) / total_size)

# Save
print(dataset_means.shape)
print(dataset_means.dtype)
print(dataset_stds.shape)
print(dataset_stds.dtype)
np.save("./4_block_dataset_means.npy", dataset_means)
np.save("./4_block_dataset_stds.npy", dataset_stds)


"""
import numpy as np
stds = np.load("/home/cougarnet.uh.edu/srizvi7/Desktop/Histopathology_Dataset_GAN/dataset_stds.npy")
means = np.load("/home/cougarnet.uh.edu/srizvi7/Desktop/Histopathology_Dataset_GAN/dataset_means.npy")

pixel_feat_img0 = np.load("/data/syed/hdgan/TMA_4096_snapshot2600/pixel_5block_features_dataset/pixel_level_feat_img_0.npy")
pixel_feat_img1 = np.load("/data/syed/hdgan/TMA_4096_snapshot2600/pixel_5block_features_dataset/pixel_level_feat_img_1.npy")

combined = np.concatenate([pixel_feat_img0, pixel_feat_img1], axis=0).astype(np.float64)


s0 = sum(1 for x in samples)
s1 = sum(x for x in samples)
s2 = sum(x*x for x in samples)
std_dev = math.sqrt((s0 * s2 - s1 * s1)/(s0 * (s0 - 1)))
"""

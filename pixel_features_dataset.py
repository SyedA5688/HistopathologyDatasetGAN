import os
import random

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from utils.data_util import tma_4096_image_idxs


class PixelFeaturesDataset(Dataset):
    def __init__(self, data_path, split="train", val_fold=0):
        """
        Args:
            data_path: absolute path to directory containing training image masks as well as a file called
                        latent_stylegan2.npy containing the latent vectors of each of the images that were
                        annotated in order to create the training mask set.

            split: dataset split to create for this dataset. Possible values are train, val, and test

        Function:
            Basic details:
                This dataloader loads pixel-level features for an image (1 pixel -> np array of (1, 6080)
                and ground truth of (1,). Each image in dataset gives (1048576, 6080) features, taking up
                25.5 GB of disk space per image as a saved npy file and 18.2 GB of space per image as a
                compressed npz file.

            Strategy for reducing memory consumption:
                Use Numpy memory-mapped array, only load chunks of arrays that are needed.
                For ground truth masks, the entire (1024x1024) mask is loaded and then turned into a single
                array of 1048576 * num_dataset_images, and this gives the true length of the dataset.

                The net effect of this memory-consumption-reduction scheme is that training is slower due to
                constantly loading things from disk, which is the tradeoff accepted in order to not require
                25.5 * 36 GB of RAM memory in order to run training.

            Notes:
                For this dataset, specifying num_workers >= 4 and pin_memory=True is a good idea, since many
                batches can fit into memory.
        """
        assert split in ["train", "val", "test"]
        assert val_fold in [0, 1, 2, 3, 4], "Unknown validation fold specified, must be 0-4"
        print("Creating dataset...")

        train_val_split = 16
        if split == "train":
            img_name_idxs = tma_4096_image_idxs[0:train_val_split]
        elif split == "val":
            start = train_val_split + 4 * val_fold
            idxs = list(range(start, start + 4))
            img_name_idxs = [tma_4096_image_idxs[i] for i in idxs]
        else:
            start1 = train_val_split + 4 * val_fold
            start2 = train_val_split + 4 * val_fold + 4
            idxs = list(range(train_val_split, start1)) + list(range(start2, len(tma_4096_image_idxs)))
            img_name_idxs = [tma_4096_image_idxs[i] for i in idxs]

        self.img_pixel_feat_len = 1024*1024  # 4096*4096  ToDo: Reduced pixels
        self.split = split
        self.total_size = 0
        self.pixel_feat_mean = np.load(os.path.join(data_path, "dataset_means.npy"))
        self.pixel_feat_stds = np.load(os.path.join(data_path, "dataset_stds.npy"))
        self.class_samp_weights = {0: 0.05, 1: 0., 2: 0.019, 3: 0.08, 4: 0.97, 5: 0.28}

        self.features = {}
        for idx, image_idx in enumerate(img_name_idxs):
            pixel_data = np.load(os.path.join(data_path, "pixel_5block_features_dataset", "pixel_level_feat_img_{}.npy".format(image_idx)), mmap_mode='r')  # Shape (nsamples, nfeatures)
            self.features[idx] = pixel_data

        self.ground_truth = {}
        self.temp_idx_mapping = {}
        idx_counter = 0
        for idx, image_idx in enumerate(img_name_idxs):
            print("Processing image", image_idx)
            mask = np.load(os.path.join(data_path, "image_{}_mask.npy".format(image_idx)))
            # h, w = mask.shape
            image_pixel_label_list = []
            for row in range(1024, 2048):  # ToDo: Reduced pixels
                for col in range(1024, 2048):
                    image_pixel_label_list.append(mask[row, col])  # Append 1 by 1, ensure correct order
                    self.temp_idx_mapping[idx_counter] = 4096 * row + col
                    idx_counter += 1

            image_pixel_label_list = np.array(image_pixel_label_list)
            self.total_size += len(image_pixel_label_list)
            self.ground_truth[idx] = image_pixel_label_list  # match pixel features
        print("Dataset creation completed.")

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        image_idx = index // self.img_pixel_feat_len
        pixel_feat_idx = index % self.img_pixel_feat_len  # 1023

        ground_truth_class = self.ground_truth[image_idx][pixel_feat_idx]

        # pixel feature index is different because of
        pixel_feat = self.features[image_idx][self.temp_idx_mapping[pixel_feat_idx]].astype(np.float32)
        pixel_feat = (pixel_feat - self.pixel_feat_mean) / self.pixel_feat_stds

        if self.split == "train" and random.random() < 0.5:
            # std min for pixel dataset is around 0.01
            pixel_feat += np.random.normal(loc=0., scale=0.01, size=(496,)).astype(np.float32)  # torch.zeros(496).data.normal_(0, 0.01)

        return pixel_feat, ground_truth_class


if __name__ == "__main__":
    training_set = PixelFeaturesDataset(data_path="/data/syed/hdgan/TMA_4096_snapshot2600/", split="train")
    sample_weights = [training_set.class_samp_weights[training_set.ground_truth[idx // training_set.img_pixel_feat_len][
        idx % training_set.img_pixel_feat_len]] for idx in range(len(training_set))]
    sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(training_set), replacement=True)

    train_loader = DataLoader(training_set, batch_size=4096, sampler=sampler)
    # train_loader = DataLoader(training_set, batch_size=65536, shuffle=True)
    train_iter = iter(train_loader)
    print("Loading batch...")
    batch1 = train_iter.__next__()
    print(batch1)

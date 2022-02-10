import os
import random
import numpy as np
from torch.utils.data import Dataset


class PixelFeaturesDataset(Dataset):
    def __init__(self, data_path, split="train"):
        """
        Args:
            data_path: absolute path to directory containing training image masks as well as a file called
                        latent_stylegan2.npy containing the latent vectors of each of the images that were
                        annotated in order to create the training mask set.

            split: dataset split to create for this dataset. Possible values are train, val, and test

            val_fold: integer in range [0,4] indicating which group of four images in evaluation set will be
                        the validation set. For example, if images 1-16 are in training set and val_fold is
                        given as 1, then images 17-20 will be validation set, and 21-36 will be test set.

        Function:
            Basic details:
                This dataloader loads pixel-level features for an image (1 pixel -> np array of (1, 6080)
                and ground truth of (1,). Each image in dataset gives (1048576, 6080) features, taking up
                25.5 GB of disk space per image as a saved npy file and 18.2 GB of space per image as a
                compressed npz file.

            Strategy for reducing memory consumption:
                In order to reduce memory constraints regarding loading the entire pixel-level dataset,
                this datasert assumes that pixel-level features have been saved as separate .npy files
                for each image (i.e. dataset of 36 images -> 36 .npy files). Npy files have the advantage
                of being memory-mappable, meaning numpy can load chunks of the array when they are needed
                rather than loading the entire 25.5 GB array into memory.

                To avoid loading entire 25.5 * 36 images into memory, this dataloader loads the (1048576, 6080)
                pixel-level features of each image as a memory-mapped array into a dictionary; when an index
                is retrieved by Pytorch's dataloader, it is divided by 1048576 to find which image file needs
                to be indexed, and then modulus operator gives us which pixel to load from (1048576, 6080)
                array. Because of memory mapping, only chunks of arrays that are needed are loaded, keeping
                memory consumption down.

                For ground truth masks, the entire (1024x1024) mask is loaded and then turned into a single
                array of 1048576 * num_dataset_images, and this gives the true length of the dataset.

                The net effect of this memory-consumption-reduction scheme is that training is slower due to
                constantly loading things from disk, which is the tradeoff accepted in order to not require
                25.5 * 36 GB of RAM meory in order to run training.

            Regarding validation folds, this dataset assumes around 16-30 images for training, and then
            another 20 images for validation and test (combined together called evaluation set). The
            evaluation set will be split into five folds of 4 images, giving 5-fold cross validation.

            Around 627 GB of cache/buffer is being used up because of OS loading chunks from disk
            and caching them while dataloader is calling for shuffled indices -
        """

        # Assertion statements regarding dataset split for this dataset as well as validation fold picked
        assert split in ["train", "val", "test"]
        # assert val_fold in [0, 1, 2, 3, 4], "Unknown validation fold specified, must be 0-4"

        if split == "train":
            # idxs = list(range(16))
            idxs = list(range(24))
        elif split == "val":
            # idxs = list(range(16, 20))
            idxs = list(range(24, 30))
        elif split == "test":
            # idxs = list(range(20, 36))
            idxs = list(range(30, 36))
        else:
            idxs = None
            assert "Unknown split for pixel feature dataloader."

        self.img_pixel_feat_len = 1024*1024  # 1048576
        self.split = split
        self.total_size = 0

        # self.class_augm_probabilities = {0: 0.119, 1: 0.102, 2: 0.95, 3: 0.047, 4: 0.117, 5: 0.086, 6: 0.023}  # artery_pixel_count / other_class_pixel_count
        # 100 / percentage of class in training dataset, changed class 0 from 12.34, it wasn't showing up as much in batches
        self.class_samp_weights = {0: 12.34, 1: 10.62, 2: 104.0, 3: 4.92, 4: 12.11, 5: 8.89, 6: 2.4}

        # Numpy memmap works on .npy files, will load chunks of array when it is used in computations
        self.features = {}
        for idx, image_idx in enumerate(idxs):
            pixel_data = np.load(os.path.join(data_path, "pixel_level_feat_img_" + str(image_idx) + ".npy"), mmap_mode='r')  # Shape (1048576, 6080)
            self.features[idx] = pixel_data

        # Load image masks corresponding to images produced by StyleGAN2 for saved latent vectors
        self.ground_truth = {}
        for idx, image_idx in enumerate(idxs):
            mask = np.load(os.path.join(data_path, "image_" + str(image_idx) + "_mask.npy"))
            h, w = mask.shape
            image_pixel_label_list = []
            for row in range(h):
                for col in range(w):
                    image_pixel_label_list.append(mask[row, col])  # Append 1 by 1, match order that pixel features were saved in

            image_pixel_label_list = np.array(image_pixel_label_list)  # shape (1048576,)
            self.total_size += len(image_pixel_label_list)
            self.ground_truth[idx] = image_pixel_label_list  # match pixel features

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        image_idx = index // self.img_pixel_feat_len
        pixel_feat_idx = index % self.img_pixel_feat_len

        pixel_feat = self.features[image_idx][pixel_feat_idx]
        ground_truth_class = self.ground_truth[image_idx][pixel_feat_idx]

        # if self.split == "train" and random.random() < self.class_augm_probabilities[ground_truth_class]:
        #     aug_pixel_feat = pixel_feat + np.random.normal(loc=0., scale=0.05, size=(6080)).astype(np.float32)
        #     # torch.zeros(1, 6080).data.normal_(0, 0.05)
        #     return aug_pixel_feat, ground_truth_class

        # Manual form of label smoothing
        if self.split == "train" and random.random() < 0.05:
            ground_truth_class = random.choice(list(range(7)))  # Choose another random class as label

        # returns shapes (6080,) and (1,)
        return pixel_feat, ground_truth_class


# class ClassBalancedPixelFeaturesDataset(Dataset):
#     def __init__(self, data_path, split="train"):
#         # Assertion statements regarding dataset split for this dataset as well as validation fold picked
#         assert split in ["train", "val", "test"]
#         # assert val_fold in [0, 1, 2, 3, 4], "Unknown validation fold specified, must be 0-4"
#
#         if split == "train":
#             # idxs = list(range(16))
#             idxs = list(range(24))
#         elif split == "val":
#             # idxs = list(range(16, 20))
#             idxs = list(range(24, 30))
#         elif split == "test":
#             # idxs = list(range(20, 36))
#             idxs = list(range(30, 36))
#         else:
#             idxs = None
#             assert "Unknown split for pixel feature dataloader."
#
#         self.split = split
#         self.img_pixel_feat_len = 1024*1024  # 1048576
#         self.class_sampling_probabilities = { 0: 0.119, 1: 0.102, 2: 1., 3: 0.047, 4: 0.117, 5: 0.086, 6: 0.023 }
#         self.class_counts = [0, 0, 0, 0, 0, 0, 0]
#         self.total_size = 0
#
#         self.features = []
#         self.ground_truth = []
#         for idx, image_idx in enumerate(idxs):
#             # Load pixel feature data
#             pixel_data = np.load(os.path.join(data_path, "pixel_level_feat_img_" + str(image_idx) + ".npy"), mmap_mode='r')  # Shape (1048576, 6080)
#
#             # Load mask
#             mask = np.load(os.path.join(data_path, "image_" + str(image_idx) + "_mask.npy"))
#             h, w = mask.shape
#             image_pixel_label_list = []
#             for row in range(h):
#                 for col in range(w):
#                     image_pixel_label_list.append(mask[row, col])  # Append 1 by 1, match order that pixel features were saved in
#
#             image_pixel_label_list = np.array(image_pixel_label_list)  # shape (1048576,)
#
#             # Balance classes; filter out with random probability
#             for i in range(len(image_pixel_label_list)):
#                 if random.random() < self.class_sampling_probabilities[image_pixel_label_list[i]]:
#                     self.features.append(pixel_data[i:i+1, :])
#                     self.ground_truth.append(image_pixel_label_list[i])
#                     self.class_counts[image_pixel_label_list[i]] += 1
#                     self.total_size += 1
#
#         self.features = np.concatenate(self.features, axis=0)
#         self.ground_truth = np.array(self.ground_truth)
#
#     def __len__(self):
#         return self.total_size
#
#     def __getitem__(self, index):
#         pixel_feat = self.features[index]
#         ground_truth_class = self.ground_truth[index]
#
#         if self.split == "train" and random.random() < 0.5:  # Classes are balanced
#             aug_pixel_feat = pixel_feat + np.random.normal(loc=0., scale=0.05, size=(6080)).astype(np.float32)  # torch.zeros(1, 6080).data.normal_(0, 0.05)
#             return aug_pixel_feat, ground_truth_class
#
#         return pixel_feat, ground_truth_class


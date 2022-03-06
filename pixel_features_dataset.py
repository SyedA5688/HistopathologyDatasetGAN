import os
import numpy as np

from torch.utils.data import Dataset
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

        if split == "train":
            img_name_idxs = tma_4096_image_idxs[0:30]
        elif split == "val":
            start = 30 + 7 * val_fold
            idxs = list(range(start, start + 7))
            img_name_idxs = [tma_4096_image_idxs[i] for i in idxs]
        else:
            start1 = 30 + 7 * val_fold
            start2 = 30 + 7 * val_fold + 7
            idxs = list(range(30, start1)) + list(range(start2, 65))
            img_name_idxs = [tma_4096_image_idxs[i] for i in idxs]

        self.img_pixel_feat_len = 4096*4096
        self.split = split
        self.total_size = 0
        self.class_samp_weights = {0: 0.0256, 1: 0., 2: 0.0192, 3: 0.1805, 4: 2.7027, 5: 0.3311}

        self.features = {}
        for idx, image_idx in enumerate(img_name_idxs):
            pixel_data = np.load(os.path.join(data_path, "pixel_features_dataset", "pixel_level_feat_img_{}.npy".format(image_idx)), mmap_mode='r')  # Shape (1048576, 6080)
            self.features[idx] = pixel_data

        self.ground_truth = {}
        for idx, image_idx in enumerate(img_name_idxs):
            mask = np.load(os.path.join(data_path, "image_{}_mask.npy".format(image_idx)))
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

        # returns shapes (6128,) and (1,)
        return pixel_feat, ground_truth_class

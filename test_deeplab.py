import os
import json
import argparse

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms

from utils.utils import oht_to_scalar, colorize_mask, dice_coefficient
from utils.data_util import tma_12_palette, tma_12_class

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def log_string(logger, str1):
    print(str1)
    logger.write(str1 + "\n")
    logger.flush()


class GeneratedImageLabelDataset(Dataset):
    def __init__(self, data_path, split="train"):
        assert split in ["train", "val", "test"]

        self.images = []
        self.masks = []

        if split == "train":
            # all_files = os.listdir(os.path.join(data_path, "artificial_dataset_5000"))
            # img_files = [file for file in all_files if "generated_image" in file]

            artificial_dataset_path = os.path.join(data_path, "artificial_dataset_5000")
            for idx in range(1400):  # len(img_files)
                img_filename = "generated_image_" + str(idx) + ".jpg"
                self.images.append(os.path.join(artificial_dataset_path, img_filename))

                mask = np.load(os.path.join(artificial_dataset_path, "multiclass_mask_" + str(idx) + ".npy"))
                self.masks.append(mask)

        elif split == "val":
            # Use validation set that was used to train pixel classifier, 6 images
            image_idxs = list(range(24, 30))
            for idx in image_idxs:
                img_filename = "image_" + str(idx) + ".jpg"
                self.images.append(os.path.join(data_path, img_filename))

                mask = np.load(os.path.join(data_path, "image_" + str(idx) + "_mask.npy"))
                self.masks.append(mask)

        elif split == "test":
            # Use test set that was used to train pixel classifier, 6 images
            image_idxs = list(range(30, 36))
            for idx in image_idxs:
                img_filename = "image_" + str(idx) + ".jpg"
                self.images.append(os.path.join(data_path, img_filename))

                mask = np.load(os.path.join(data_path, "image_" + str(idx) + "_mask.npy"))
                self.masks.append(mask)

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        img = transforms.ToTensor()(img)
        mask = self.masks[index]

        return img, mask


def plot_img_and_colored_mask(mask, ground_truth_img, image_num, split):
    colorized_mask = colorize_mask(mask, tma_12_palette)
    colorize_ground_truth = colorize_mask(ground_truth_img, tma_12_palette)

    plt.imsave(os.path.join(SAVE_PATH, split + "_image" + str(image_num) + "_pred_mask.png"), colorized_mask)
    plt.clf()

    plt.imsave(os.path.join(SAVE_PATH, split + "_image" + str(image_num) + "_ground_truth_mask.png"), colorize_ground_truth)
    plt.clf()
    plt.close()


def test_one_classifier(model_path, data_loader):
    assert len(data_loader) == 1, "If more than 1 test batch, need to edit mask list code"
    model = deeplabv3_resnet50(pretrained=False, num_classes=args["num_classes"])
    checkpoint = torch.load(model_path)['model_state_dict']  # , map_location='cpu'
    prefix = 'module.'
    n_clip = len(prefix)
    adapted_dict = {k[n_clip:]: v for k, v in checkpoint.items() if k.startswith(prefix)}

    model.load_state_dict(adapted_dict)
    model.to(device)
    model.eval()

    logger = open(os.path.join(SAVE_PATH, "test_accuracy_log.txt"), "w")
    ground_truth_mask_list = []
    predicted_mask_list = []

    with torch.no_grad():
        class_correct_count = [0 for _ in range(len(tma_12_class))]
        class_total_count = [0 for _ in range(len(tma_12_class))]

        for idx, (data, ground_truth) in enumerate(data_loader):
            data, ground_truth = data.to(device), ground_truth.to(device)
            pred_logits = model(data)  # out shape: [b, num_classes=7, height, width]
            mask_pred = oht_to_scalar(pred_logits['out'])

            # Accumulating class-wise counts to compute class-wise accuracy later on
            for image_idx in range(len(mask_pred)):
                print("Computing counts for image", image_idx)
                for h in range(args["resolution"]):
                    for w in range(args["resolution"]):
                        class_total_count[ground_truth[image_idx, h, w]] += 1
                        if ground_truth[image_idx, h, w] == mask_pred[image_idx, h, w]:
                            class_correct_count[ground_truth[image_idx, h, w]] += 1

            # Only have 1 batch for now
            ground_truth_mask_list = ground_truth.cpu().numpy()
            predicted_mask_list = mask_pred.cpu().numpy()

            for image_idx in range(len(mask_pred)):
                # Compute and print dice coefficients
                ground_truth_mask_one_hot = F.one_hot(torch.tensor(ground_truth_mask_list[image_idx]).long(),
                                                      num_classes=args["num_classes"])
                predicted_mask_one_hot = F.one_hot(torch.tensor(predicted_mask_list[image_idx]).long(), num_classes=args["num_classes"])
                dice_coeff = dice_coefficient(ground_truth_mask_one_hot, predicted_mask_one_hot)
                log_string(logger, "Image {} dice score: {}".format(image_idx, round(dice_coeff.item(), 5)))

    # Print out class-wise and total pixel-level accuracy
    log_string(logger, "Class-wise Accuracies for DeepLabV3 Model:")
    class_accuracies = []
    for i in range(len(tma_12_class)):
        if class_total_count[i] != 0:
            class_accuracies.append(class_correct_count[i] / class_total_count[i])
        else:
            class_accuracies.append(-1)  # Not applicable, 0 pixels of this class in test set

    for i in range(len(tma_12_class)):
        log_string(logger, tma_12_class[i] + " Accuracy: " + str(round(class_accuracies[i], 5)))
    log_string(logger, "")

    log_string(logger, "Total Accuracy: " + str(round(sum(class_correct_count) / sum(class_total_count), 5)))
    logger.close()

    return ground_truth_mask_list, predicted_mask_list


def main():
    # val_dataset = GeneratedImageLabelDataset(data_path=args["dataset_dir"], split="val")
    # val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_dataset = GeneratedImageLabelDataset(data_path=args["dataset_dir"], split="test")
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    split_name = "test"
    ground_truth_mask_list, predicted_mask_list = test_one_classifier(
        os.path.join(training_run_dir, training_run, "best_deeplab_model.pth"), test_dataloader)
    for idx in range(len(predicted_mask_list)):
        print("Saving predicted mask and ground truth mask for image", idx)
        plot_img_and_colored_mask(predicted_mask_list[idx], ground_truth_mask_list[idx], idx, split_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default="/home/cougarnet.uh.edu/srizvi7/Desktop/Histopathology_Dataset_GAN/experiments/TMA_Arteriole_Segmentation.json")
    opts = parser.parse_args()
    args = json.load(open(opts.experiment, 'r'))

    training_run_dir = "/home/cougarnet.uh.edu/srizvi7/Desktop/Histopathology_Dataset_GAN/segm-training-runs/"
    training_run = "0001-TMA_Arteriole_Segmentation"

    SAVE_PATH = os.path.join(training_run_dir, training_run, "test_set_mask_pred")
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    main()

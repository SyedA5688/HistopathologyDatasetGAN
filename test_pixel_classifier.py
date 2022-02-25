import os
import json
import argparse
from random import seed

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils.utils import oht_to_scalar, colorize_mask, dice_coefficient
from utils.data_util import tma_12_palette, tma_12_class
from pixel_features_dataset import PixelFeaturesDataset
from networks.pixel_classifier import PixelClassifier


chosen_seed = 0
seed(chosen_seed)
np.random.seed(chosen_seed)
torch.manual_seed(chosen_seed)
torch.cuda.manual_seed_all(chosen_seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def log_string(logger, str1):
    print(str1)
    logger.write(str1 + "\n")
    logger.flush()


def plot_img_and_colored_mask(mask, ground_truth_img, image_num, split):
    """
    Args:
        mask: Numpy array of shape (height, width) to save
        ground_truth_img: Numpy array of shape (height, width) to save
        image_num:
        split: dataset split images belong to

    """
    colorized_mask = colorize_mask(mask, tma_12_palette)
    colorize_ground_truth = colorize_mask(ground_truth_img, tma_12_palette)

    plt.imshow(colorized_mask)
    plt.savefig(os.path.join(SAVE_PATH, split + "_image" + str(image_num) + "_pred_mask.png"), bbox_inches='tight')
    plt.clf()

    plt.imshow(colorize_ground_truth)
    plt.savefig(os.path.join(SAVE_PATH, split + "_image" + str(image_num) + "_ground_truth_mask.png"), bbox_inches='tight')
    plt.clf()
    plt.close()


def test_one_classifier(model_path, data_loader):
    model = PixelClassifier(num_classes=args["num_classes"], dim=args['featuremaps_dim'][-1])
    checkpoint = torch.load(model_path)['model_state_dict']
    prefix = 'module.'
    n_clip = len(prefix)
    adapted_dict = {k[n_clip:]: v for k, v in checkpoint.items() if k.startswith(prefix)}

    model.load_state_dict(adapted_dict)
    model.to(device)
    model.eval()

    image_end_points = [1024 * i for i in range(1, 37)]
    ground_truth_mask_list = []
    predicted_mask_list = []
    dice_scores = []

    logger = open(os.path.join(SAVE_PATH, "test_accuracy_log.txt"), "w")
    image_counter = 0

    with torch.no_grad():
        ground_truth_mask = torch.zeros((1024, 1024))
        predicted_mask = torch.zeros((1024, 1024))

        class_correct_count = [0 for _ in range(len(tma_12_class))]
        class_total_count = [0 for _ in range(len(tma_12_class))]

        for idx, (data, ground_truth) in enumerate(data_loader):
            if idx % 100 == 0:
                print("On batch idx", idx)
            data, ground_truth = data.to(device), ground_truth.long().to(device)

            pred_logits = model(data)  # pred shape [b, 7]
            pixel_preds = oht_to_scalar(pred_logits)

            # Accumulating class-wise counts to compute class-wise accuracy later on
            for i in range(len(pred_logits)):
                class_total_count[ground_truth[i]] += 1
                if ground_truth[i] == pixel_preds[i]:  # Correct prediction
                    class_correct_count[ground_truth[i]] += 1

            ground_truth_mask[idx % 1024] = ground_truth
            predicted_mask[idx % 1024] = pixel_preds

            if (idx + 1) in image_end_points:
                ground_truth_mask = ground_truth_mask.numpy()
                predicted_mask = predicted_mask.numpy()

                ground_truth_mask_list.append(np.expand_dims(np.copy(ground_truth_mask), axis=0))
                predicted_mask_list.append(np.expand_dims(np.copy(predicted_mask), axis=0))

                ground_truth_mask_one_hot = F.one_hot(ground_truth_mask, num_classes=args["num_classes"])
                predicted_mask_one_hot = F.one_hot(predicted_mask, num_classes=args["num_classes"])
                dice_coeff = dice_coefficient(ground_truth_mask_one_hot, predicted_mask_one_hot)

                log_string(logger, "Image {} dice score: {}".format(image_counter, dice_coeff))
                image_counter += 1
                dice_scores.append(dice_coeff)

                ground_truth_mask = torch.zeros((1024, 1024))
                predicted_mask = torch.zeros((1024, 1024))

        ground_truth_mask_list = np.concatenate(ground_truth_mask_list, axis=0)  # Shape [num_images, 1024, 1024]
        predicted_mask_list = np.concatenate(predicted_mask_list, axis=0)

    # Print out class-wise and total pixel-level accuracy
    log_string(logger, "\n\nClass-wise Accuracies for Single Model:")
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
    # test_set = PixelFeaturesDataset(args["dataset_save_dir"], split="test")
    # test_dataloader = DataLoader(test_set, batch_size=1024, shuffle=False)

    val_set = PixelFeaturesDataset(args["dataset_save_dir"], split="val")
    val_dataloader = DataLoader(val_set, batch_size=1024, shuffle=False)

    split_name = "test"
    ground_truth_mask_list, predicted_mask_list = test_one_classifier(
        os.path.join(training_run_dir, training_run, "best_model_" + str(0) + "_ep0.pth"), val_dataloader)
    for idx in range(len(predicted_mask_list)):
        print("Saving predicted mask and ground truth mask for image", idx)
        # all_classifiers_pred_mask_list[idx] is (1, 1, 1024, 1024)
        plot_img_and_colored_mask(predicted_mask_list[idx], ground_truth_mask_list[idx], idx, split_name)

    # Code for ensemble mask prediction
    # all_classifiers_ground_truth_mask_list = []
    # all_classifiers_pred_mask_list = []
    #
    # for i in range(args["model_num"]):
    #     ground_truth_mask_list, predicted_mask_list = test_one_classifier(os.path.join(training_run_dir, training_run, "best_model_" + str(i) + ".pth"), args, test_dataloader)
    #     all_classifiers_ground_truth_mask_list.append(np.expand_dims(ground_truth_mask_list, axis=0))
    #     all_classifiers_pred_mask_list.append(np.expand_dims(predicted_mask_list, axis=0))
    #
    # all_classifiers_ground_truth_mask_list = np.concatenate(all_classifiers_ground_truth_mask_list, axis=0)  # [num_classifiers, num_images, 1024, 1024]
    # all_classifiers_pred_mask_list = np.concatenate(all_classifiers_pred_mask_list, axis=0)
    #
    # # ModeResult is {mode, counts), and mode array is [1,nimgs,h,q] so do stats.mode(array, axis=0)[0][0]
    # all_classifiers_ground_truth_mask_list = stats.mode(all_classifiers_ground_truth_mask_list, axis=0)[0][0]  # [num_images, 1024, 1024] majority voting mask
    # all_classifiers_pred_mask_list = stats.mode(all_classifiers_pred_mask_list, axis=0)[0][0]
    #
    # for idx in range(len(all_classifiers_pred_mask_list)):
    #     print("Saving predicted mask and ground truth mask for image", idx)
    #     # all_classifiers_pred_mask_list[idx] is (1, 1, 1024, 1024)
    #     plot_img_and_colored_mask(all_classifiers_pred_mask_list[idx], all_classifiers_ground_truth_mask_list[idx], idx, SAVE_PATH, split_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str,
                        default="/home/cougarnet.uh.edu/srizvi7/Desktop/Histopathology_Dataset_GAN/experiments/TMA_Arteriole_stylegan2_ada.json")
    parser.add_argument('--num_sample', type=int, default=100)

    opts = parser.parse_args()
    args = json.load(open(opts.experiment, 'r'))

    training_run_dir = "/home/cougarnet.uh.edu/srizvi7/Desktop/Histopathology_Dataset_GAN/training-runs/"
    training_run = "0031-TMA_Arteriole_stylegan2_ada"
    # SAVE_PATH = os.path.join(training_run_dir, training_run, "ensemble_mask_pred")
    SAVE_PATH = os.path.join(training_run_dir, training_run, "classifier0_ep0_val_mask_pred")
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    main()

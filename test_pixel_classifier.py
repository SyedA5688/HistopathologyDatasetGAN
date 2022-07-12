import os
import json
import argparse
from random import seed
from statistics import mean

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils.utils import oht_to_scalar, colorize_mask, dice_coefficient
from utils.data_util import tma_4096_crop_class_printname, tma_4096_palette, tma_4096_image_idxs
from pixel_features_dataset import PixelFeaturesDataset
from networks.pixel_classifier import PixelClassifier


chosen_seed = 0
seed(chosen_seed)
np.random.seed(chosen_seed)
torch.manual_seed(chosen_seed)
torch.cuda.manual_seed_all(chosen_seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
    colorized_mask = colorize_mask(mask, tma_4096_palette)
    colorize_ground_truth = colorize_mask(ground_truth_img, tma_4096_palette)

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

    image_end_points = [4096 * i for i in range(1, 66)]
    ground_truth_mask_list = []
    predicted_mask_list = []
    confusion_matrix = torch.zeros(args["num_classes"], args["num_classes"])
    dice_scores = []

    logger = open(os.path.join(SAVE_PATH, "test_accuracy_log.txt"), "w")
    image_counter = 0

    with torch.no_grad():
        ground_truth_mask = torch.zeros((args["featuremaps_dim"][0], args["featuremaps_dim"][1]))
        predicted_mask = torch.zeros((args["featuremaps_dim"][0], args["featuremaps_dim"][1]))

        for idx, (data, ground_truth) in enumerate(data_loader):
            if idx % 100 == 0:
                print("On batch idx", idx)
            data, ground_truth = data.float().to(device), ground_truth.long().to(device)

            pred_logits = model(data)  # pred shape [b, 7]
            pixel_preds = oht_to_scalar(pred_logits)

            # Accumulating predictions for confusion matrix
            for t, p in zip(ground_truth.view(-1), pixel_preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            ground_truth_mask[idx % args["featuremaps_dim"][1]] = ground_truth
            predicted_mask[idx % args["featuremaps_dim"][1]] = pixel_preds

            if (idx + 1) in image_end_points:
                ground_truth_mask_one_hot = F.one_hot(ground_truth_mask.long(), num_classes=args["num_classes"])
                predicted_mask_one_hot = F.one_hot(predicted_mask.long(), num_classes=args["num_classes"])
                dice_coeff = dice_coefficient(ground_truth_mask_one_hot, predicted_mask_one_hot)

                log_string(logger, "Image {} dice score: {}".format(image_counter, round(dice_coeff.item(), 5)))
                image_counter += 1
                dice_scores.append(dice_coeff.item())

                ground_truth_mask = ground_truth_mask.numpy()
                predicted_mask = predicted_mask.numpy()

                ground_truth_mask_list.append(np.expand_dims(np.copy(ground_truth_mask), axis=0))
                predicted_mask_list.append(np.expand_dims(np.copy(predicted_mask), axis=0))

                ground_truth_mask = torch.zeros((args["featuremaps_dim"][0], args["featuremaps_dim"][1]))
                predicted_mask = torch.zeros((args["featuremaps_dim"][0], args["featuremaps_dim"][1]))

        ground_truth_mask_list = np.concatenate(ground_truth_mask_list, axis=0)  # Shape [num_images, 4096, 4096]
        predicted_mask_list = np.concatenate(predicted_mask_list, axis=0)

    # Print out class-wise and total pixel-level accuracy
    log_string(logger, "\n\nClass-wise Accuracies:")
    class_accuracies = []
    for i in range(len(tma_4096_crop_class_printname)):
        if confusion_matrix[i, :].sum().item() == 0:
            class_accuracies.append(-1)  # 0 pixels of this class in validation set
        else:
            class_accuracies.append(confusion_matrix[i, i].item() / confusion_matrix[i, :].sum().item())

    for i in range(len(tma_4096_crop_class_printname)):
        log_string(logger, tma_4096_crop_class_printname[i] + " Accuracy: " + str(round(class_accuracies[i], 5)))
    log_string(logger, "")

    log_string(logger, ",\t".join(tma_4096_crop_class_printname))
    # log_string(str(confusion_matrix))
    log_string(logger, '\n'.join(['\t'.join(['{:12.1f}'.format(num.item()) for num in row]) for row in confusion_matrix]))
    log_string(logger, "\n")

    log_string(logger, "Average Dice Score: " + str(mean(dice_scores)))
    logger.close()

    return ground_truth_mask_list, predicted_mask_list


def main():
    test_set = PixelFeaturesDataset(args["dataset_save_dir"], split="test", val_fold=args["val_fold"])
    test_dataloader = DataLoader(test_set, batch_size=4096, shuffle=False, num_workers=8, pin_memory=True)

    # val_set = PixelFeaturesDataset(args["dataset_save_dir"], split="val")
    # val_dataloader = DataLoader(val_set, batch_size=4096, shuffle=False)

    split_name = "test"
    ground_truth_mask_list, predicted_mask_list = test_one_classifier(
        os.path.join(training_run_dir, training_run, "best_model_{}_ep{}.pth".format(0, 1)), test_dataloader)
    for idx in range(len(predicted_mask_list)):
        print("Saving predicted mask and ground truth mask for image", test_set.img_name_idxs[idx])
        # all_classifiers_pred_mask_list[idx] is (1, 1, 4096, 4096)
        plot_img_and_colored_mask(predicted_mask_list[idx], ground_truth_mask_list[idx], test_set.img_name_idxs[idx], split_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str,
                        default="/home/cougarnet.uh.edu/srizvi7/Desktop/Histopathology_Dataset_GAN/experiments/TMA_4096_tile.json")

    opts = parser.parse_args()
    args = json.load(open(opts.experiment, 'r'))

    training_run_dir = "/home/cougarnet.uh.edu/srizvi7/Desktop/Histopathology_Dataset_GAN/training-runs/"
    training_run = "TMA_4096-0019*"
    SAVE_PATH = os.path.join(training_run_dir, training_run, "classifier0_ep1_test_mask_pred")
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    main()

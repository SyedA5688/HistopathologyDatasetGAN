import os
import torch
# import torch.nn as nn

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from random import seed
from utils.utils import multi_acc, oht_to_scalar, colorize_mask
from utils.data_util import tma_12_palette, tma_12_class
from torch.utils.data import DataLoader
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


def plot_img_and_colored_mask(mask, ground_truth_img, image_num, SAVE_PATH, split):
    """
    Args:
        mask: Numpy array of shape (height, width) to save
        ground_truth_img: Numpy array of shape (height, width) to save
        image_num:
        SAVE_PATH:
        split: dataset split images belong to

    """
    colorized_mask = colorize_mask(mask, tma_12_palette)
    colorize_ground_truth = colorize_mask(ground_truth_img, tma_12_palette)

    plt.imshow(colorized_mask)
    plt.savefig(os.path.join(SAVE_PATH, split + "_image" + str(image_num) + "_pred_mask.png"))
    plt.clf()

    plt.imshow(colorize_ground_truth)
    plt.savefig(os.path.join(SAVE_PATH, split + "_image" + str(image_num) + "_ground_truth_mask.png"), bbox_inches='tight')
    plt.clf()
    plt.close()


def test_one_classifier(model_path, args, data_loader):
    model = PixelClassifier(num_classes=args["num_classes"], dim=args['featuremaps_dim'][-1])
    checkpoint = torch.load(model_path)['model_state_dict']
    prefix = 'module.'
    n_clip = len(prefix)
    adapted_dict = {k[n_clip:]: v for k, v in checkpoint.items() if k.startswith(prefix)}

    model.load_state_dict(adapted_dict)
    model.to(device)
    model.eval()

    # criterion = nn.CrossEntropyLoss()
    image_end_points = [1024 * i for i in range(1, 37)]

    with torch.no_grad():
        ground_truth_mask_list = []
        predicted_mask_list = []

        ground_truth_mask = torch.zeros((1024, 1024))
        predicted_mask = torch.zeros((1024, 1024))

        class_correct_count = [0 for _ in range(len(tma_12_class))]
        class_total_count = [0 for _ in range(len(tma_12_class))]

        for idx, (data, ground_truth) in enumerate(data_loader):
            if idx % 100 == 0:
                print("On batch idx", idx)
            data, ground_truth = data.to(device), ground_truth.long().to(device)
            # data is [b, 6080], ground_truth is [64,]

            pred_logits = model(data)  # pred shape [b, 7]  # 7 class output probabilities
            pixel_preds = oht_to_scalar(pred_logits)

            # Accumulating class-wise counts to compute class-wise accuracy later on
            for i in range(len(pred_logits)):
                class_total_count[ground_truth[i]] += 1
                if ground_truth[i] == pixel_preds[i]:  # Correct prediction
                    class_correct_count[ground_truth[i]] += 1

            ground_truth_mask[idx % 1024] = ground_truth
            predicted_mask[idx % 1024] = pixel_preds

            # loss = criterion(pred_logits, ground_truth)
            # acc = multi_acc(pred_logits, ground_truth)

            # total_loss += loss.item() 1024 2048
            # summed_acc += acc.item()

            if (idx + 1) in image_end_points:
                ground_truth_mask = ground_truth_mask.numpy()
                predicted_mask = predicted_mask.numpy()

                ground_truth_mask_list.append(np.expand_dims(np.copy(ground_truth_mask), axis=0))
                predicted_mask_list.append(np.expand_dims(np.copy(predicted_mask), axis=0))

                ground_truth_mask = torch.zeros((1024, 1024))
                predicted_mask = torch.zeros((1024, 1024))

        ground_truth_mask_list = np.concatenate(ground_truth_mask_list, axis=0)  # Shape [num_images, 1024, 1024]
        predicted_mask_list = np.concatenate(predicted_mask_list, axis=0)

    # Print out class-wise and total pixel-level accuracy
    logger = open(os.path.join(SAVE_PATH, "test_accuracy_log.txt"), "w")
    log_string(logger, "Class-wise Accuracies for Single Model:")
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
    test_set = PixelFeaturesDataset(args["pixel_feat_save_dir"], split="test")
    test_dataloader = DataLoader(test_set, batch_size=1024, shuffle=False)

    split_name = "test"
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

    ground_truth_mask_list, predicted_mask_list = test_one_classifier(
        os.path.join(training_run_dir, training_run, "best_model_" + str(0) + ".pth"),
        args,
        test_dataloader)
    for idx in range(len(predicted_mask_list)):
        print("Saving predicted mask and ground truth mask for image", idx)
        # all_classifiers_pred_mask_list[idx] is (1, 1, 1024, 1024)
        plot_img_and_colored_mask(predicted_mask_list[idx], ground_truth_mask_list[idx], idx, SAVE_PATH, split_name)


if __name__ == "__main__":
    training_run_dir = "/home/cougarnet.uh.edu/srizvi7/Desktop/Histopathology_Dataset_GAN/training-runs/"
    training_run = "0015-TMA_Arteriole_stylegan2_ada"
    # training_run = "0013-pixel_classifier_saves"
    # SAVE_PATH = os.path.join(training_run_dir, training_run, "ensemble_mask_pred")
    SAVE_PATH = os.path.join(training_run_dir, training_run, "classifier0_mask_pred")
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    args = {
        "average_latent": "",
        "batch_size": 128,
        "category": "TMA",
        "data_dir": "/home/cougarnet.uh.edu/srizvi7/Desktop/Histopathology_Dataset_GAN/generated_datasets/TMA_1024_Arteriole2",
        "deeplab_res": 1024,
        "early_stopping_patience": 8,
        "epochs": 50,
        "experiment_dir": "training-runs/00010-TMA_Arteriole_stylegan2_ada",
        "featuremaps_dim": [1024, 1024, 6080],
        "G_kwargs": {
        "class_name": "networks.stylegan2_ada.Generator",
        "z_dim": 512,
        "w_dim": 512,
        "mapping_kwargs": {
          "num_layers": 2
        },
        "synthesis_kwargs": {
          "channel_base": 32768,
          "channel_max": 512,
          "num_fp16_res": 4,
          "conv_clamp": 256
        }
        },
        "log_frequency": 30000,
        "max_training": 24,
        "model_num": 10,
        "num_classes": 7,
        "pixel_classifier_lr": 0.001,
        "pixel_feat_save_dir": "/data/syed/hdgan/TMA_1024_Arteriole2",
        "resolution": 1024,
        "stylegan_checkpoint": "/data/syed/stylegan2-ada-training/00000--mirror-auto4-Arteriole/network-snapshot-006000.pkl",
        "stylegan_ver": "2_ADA",
        "testing_data_number_class": 1,
        "testing_path": "",
        "truncation": 0.7,
        "upsample_mode":"bilinear",
        "val_fold_idx": 0
    }

    main()

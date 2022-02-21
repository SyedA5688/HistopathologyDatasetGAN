import os
import json
import argparse

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_deeplab import GeneratedImageLabelDataset
from utils.utils import oht_to_scalar, colorize_mask
from utils.data_util import tma_12_palette, tma_12_class
from torchvision.models.segmentation import deeplabv3_resnet50


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def log_string(logger, str1):
    print(str1)
    logger.write(str1 + "\n")
    logger.flush()


def plot_img_and_colored_mask(mask, ground_truth_img, image_num, split):
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
    assert len(data_loader) == 1, "If more than 1 test batch, need to edit mask list code"
    model = deeplabv3_resnet50(pretrained=False, num_classes=args["num_classes"])
    checkpoint = torch.load(model_path)['model_state_dict']
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    ground_truth_mask_list = []
    predicted_mask_list = []

    with torch.no_grad():
        class_correct_count = [0 for _ in range(len(tma_12_class))]
        class_total_count = [0 for _ in range(len(tma_12_class))]

        for idx, (data, ground_truth) in enumerate(data_loader):
            data, ground_truth = data.to(device), ground_truth.to(device)
            pred_logits = model(data)  # out shape: [b, num_classes=7, height, width]
            mask_pred = oht_to_scalar(pred_logits)

            # Accumulating class-wise counts to compute class-wise accuracy later on
            for image_idx in range(len(mask_pred)):
                for h in range(args["resolution"]):
                    for w in range(args["resolution"]):
                        class_total_count[ground_truth[image_idx, h, w]] += 1
                        if ground_truth[image_idx, h, w] == mask_pred[image_idx, h, w]:
                            class_correct_count[ground_truth[image_idx, h, w]] += 1

            # Only have 1 batch for now
            ground_truth_mask_list = ground_truth.numpy()
            predicted_mask_list = mask_pred.numpy()

    # Print out class-wise and total pixel-level accuracy
    logger = open(os.path.join(SAVE_PATH, "test_accuracy_log.txt"), "w")
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
    training_run = "0000-TMA_Arteriole_Segmentation"
    # SAVE_PATH = os.path.join(training_run_dir, training_run, "ensemble_mask_pred")
    SAVE_PATH = os.path.join(training_run_dir, training_run, "classifier0_ep0_val_mask_pred")
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    main()

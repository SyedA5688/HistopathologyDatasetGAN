import os
import json
import time
import argparse
from random import seed

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.visualization_utils import plot_loss_curves
from torchvision.models.segmentation import deeplabv3_resnet50

chosen_seed = 0
seed(chosen_seed)
np.random.seed(chosen_seed)
torch.manual_seed(chosen_seed)
torch.cuda.manual_seed_all(chosen_seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def log_string(str1):
    print(str1)
    logger.write(str1 + "\n")
    logger.flush()


def validation(model, val_loader, criterion, lowest_val_loss):
    with torch.no_grad():
        model.eval()
        val_total_loss = 0.

        for batch_idx, (data, ground_truth) in enumerate(val_loader):
            data, ground_truth = data.to(device), ground_truth.to(device)
            pred_logits = model(data)
            loss = criterion(pred_logits['out'], ground_truth)
            val_total_loss += loss.item()

        val_avg_loss = val_total_loss / (batch_idx + 1)
        if val_avg_loss < lowest_val_loss:
            improved, improved_str = True, "(improved accuracy or val loss)"
        else:
            improved, improved_str = False, ""

        return_loss = val_avg_loss if val_avg_loss < lowest_val_loss else lowest_val_loss
        log_string('Validation Avg Batch Loss: {:.8f} {}'.format(float(val_avg_loss), improved_str) + "\n")

        if improved:
            save_name = "best_deeplab_model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'validation_loss': val_avg_loss
            }, os.path.join(SAVE_PATH, save_name))

        return val_avg_loss, return_loss, improved


def train(model, train_dataloader, val_dataloader, criterion, optimizer):
    log_string("Model architecture:\n" + str(model) + "\n")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_string("Number of trainable parameters in DeepLabV3: " + str(trainable_params))

    log_string("Length of train dataset: " + str(len(train_dataloader.dataset)))
    log_string("Length of validation dataset: " + str(len(val_dataloader.dataset)) + "\n")

    lowest_validation_loss = 10000000
    lowest_train_loss = 10000000
    train_losses = []
    val_losses = []

    for epoch in range(args["epochs"]):
        log_string("Epoch " + str(epoch) + " starting...")
        model.train()
        total_train_loss = 0.

        for batch_idx, (data, ground_truth) in enumerate(train_dataloader):
            data, ground_truth = data.to(device), ground_truth.to(device)
            optimizer.zero_grad()

            pred_logits = model(data)  # out shape: [b, num_classes=7, height, width]
            loss = criterion(pred_logits['out'], ground_truth)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_avg_loss = total_train_loss / len(train_dataloader)
        train_losses.append(float(train_avg_loss))

        if train_avg_loss < lowest_train_loss:
            train_improved_str = "(improved)"
            lowest_train_loss = train_avg_loss
        else:
            train_improved_str = ""

        log_string('Epoch {:03d} Completed - Train Avg Batch Loss: {:.8f} {}'.format(epoch, train_avg_loss, train_improved_str))

        val_avg_loss, lowest_validation_loss, val_improved = validation(
            model, val_dataloader, criterion, lowest_validation_loss)
        val_losses.append(float(val_avg_loss))

    log_string("Done training model.\n\n")
    plot_loss_curves(train_losses, val_losses, "DeepLab", SAVE_PATH)


def main():
    log_string("Training configuration:")
    for key in args:
        log_string(key + ": " + str(args[key]))
    log_string("\n")

    model = deeplabv3_resnet50(pretrained=False, num_classes=args["num_classes"])
    model = nn.DataParallel(model).to(device)

    train_dataset = GeneratedImageLabelDataset(data_path=args["dataset_dir"], split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)

    val_dataset = GeneratedImageLabelDataset(data_path=args["dataset_dir"], split="val")
    val_dataloader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args["learning_rate"])

    start_time = time.time()
    try:
        train(model, train_dataloader, val_dataloader, criterion, optimizer)
    except KeyboardInterrupt:
        log_string("Training stopped early by keyboard interrupt.")
        pass

    seconds_elapsed = time.time() - start_time
    log_string("Training took %s minutes" % str(seconds_elapsed / 60.))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default="/home/cougarnet.uh.edu/srizvi7/Desktop/Histopathology_Dataset_GAN/experiments/TMA_Arteriole_Segmentation.json")
    opts = parser.parse_args()
    args = json.load(open(opts.experiment, 'r'))

    # Create experiment training directory
    SAVE_PATH = args['experiment_dir']
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
        os.mkdir(os.path.join(SAVE_PATH, "python_file_saves"))
        print('Experiment folder created at: %s' % SAVE_PATH)

    os.system('cp {} {}'.format(opts.experiment, SAVE_PATH))
    os.system("cp train_deeplab.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))

    logger = open(os.path.join(SAVE_PATH, "training_log.txt"), "w")
    main()
    logger.close()

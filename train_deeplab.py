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
from torchvision.models.segmentation import deeplabv3_resnet50

chosen_seed = 0
seed(chosen_seed)
np.random.seed(chosen_seed)
torch.manual_seed(chosen_seed)
torch.cuda.manual_seed_all(chosen_seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GeneratedImageLabelDataset(Dataset):
    def __init__(self, data_path, split="train"):
        assert split in ["train", "val", "test"]

        self.images = []
        self.masks = []

        if split == "train":
            all_files = os.listdir(os.path.join(data_path, "artificial_dataset_5000"))
            img_files = [file for file in all_files if "generated_image" in file]

            for idx in range(len(img_files)):
                img_filename = "generated_image_" + str(idx) + ".jpg"
                self.images.append(os.path.join(data_path, img_filename))

                mask = np.load(os.path.join(data_path, "multiclass_mask_" + str(idx) + ".npy"))
                self.masks.append(mask)

        elif split == "val":
            pass
            # idxs = list(range(24, 30))
        elif split == "test":
            pass
            # idxs = list(range(30, 36))

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


def train(model, dataloader, criterion, optimizer):
    pass


def main():
    log_string("Training configuration:")
    for key in args:
        log_string(key + ": " + str(args[key]))
    log_string("\n")

    model = deeplabv3_resnet50(pretrained=False, num_classes=args["num_classes"])
    model = nn.DataParallel(model).to(device)

    train_dataset = GeneratedImageLabelDataset(data_path=args["dataset_dir"], split="train")
    dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args["learning_rate"])

    start_time = time.time()
    try:
        train(model, dataloader, criterion, optimizer)
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

    os.system('cp %s %s' % (opts.experiment, SAVE_PATH))
    os.system("cp train_deeplab.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))

    logger = open(os.path.join(SAVE_PATH, "training_log.txt"), "w")
    main()
    logger.close()

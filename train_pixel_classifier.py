import os
import gc
import json
import time
import argparse
from statistics import mean
from random import seed

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

from utils.utils import multi_acc, oht_to_scalar, dice_coefficient  # , EarlyStopping
from utils.data_util import tma_4096_crop_class_printname
from networks.pixel_classifier import PixelClassifier
from pixel_features_dataset import PixelFeaturesDataset

import warnings
warnings.filterwarnings("ignore")


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()
    seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)


def log_string(str1):
    print(str1)
    logger.write(str1 + "\n")
    logger.flush()


class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())
# ToDo: Try training on other 4 folds


def validation(model, model_num, val_loader, criterion, lowest_val_loss, highest_val_acc, highest_val_dice, epoch_num):
    with torch.no_grad():
        model.eval()
        ################################
        # Evaluate on validation dataset
        ################################
        val_total_loss = 0.
        summed_acc = 0.

        image_end_points = [args["featuremaps_dim"][1] * i for i in range(1, 66)]
        ground_truth_mask = torch.zeros((args["featuremaps_dim"][0], args["featuremaps_dim"][1]))
        predicted_mask = torch.zeros((args["featuremaps_dim"][0], args["featuremaps_dim"][1]))
        confusion_matrix = torch.zeros(args["num_classes"], args["num_classes"])
        dice_scores = []

        for batch_idx, (data, ground_truth) in enumerate(val_loader):
            data, ground_truth = data.float().to(device), ground_truth.long().to(device)

            pred_logits = model(data)
            loss = criterion(pred_logits, ground_truth)
            acc = multi_acc(pred_logits, ground_truth)
            pixel_preds = oht_to_scalar(pred_logits)

            # Accumulating predictions for confusion matrix
            for t, p in zip(ground_truth.view(-1), pixel_preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            val_total_loss += loss.item()
            summed_acc += acc.item()

            ground_truth_mask[batch_idx % args["featuremaps_dim"][1]] = ground_truth
            predicted_mask[batch_idx % args["featuremaps_dim"][1]] = pixel_preds

            if (batch_idx + 1) in image_end_points:
                ground_truth_mask_one_hot = F.one_hot(ground_truth_mask.long(), num_classes=args["num_classes"])
                predicted_mask_one_hot = F.one_hot(predicted_mask.long(), num_classes=args["num_classes"])

                dice_coeff = dice_coefficient(ground_truth_mask_one_hot, predicted_mask_one_hot)
                dice_scores.append(dice_coeff.item())

                ground_truth_mask = torch.zeros((args["featuremaps_dim"][0], args["featuremaps_dim"][1]))
                predicted_mask = torch.zeros((args["featuremaps_dim"][0], args["featuremaps_dim"][1]))

        ############################
        # Display validation results
        ############################
        val_avg_loss = val_total_loss / (batch_idx + 1)
        val_avg_acc = summed_acc / (batch_idx + 1)
        val_avg_dice_coeff = mean(dice_scores)

        tf_writer.add_scalar("Loss/val", val_avg_loss, epoch_num)
        tf_writer.add_scalar("Accuracy/val", val_avg_acc, epoch_num)
        tf_writer.add_scalar("Dice/val", val_avg_dice_coeff, epoch_num)

        if val_avg_acc > highest_val_acc or val_avg_loss < lowest_val_loss or val_avg_dice_coeff > highest_val_dice:
            improved, improved_str = True, "(improved)"
        else:
            improved, improved_str = False, ""

        return_loss = val_avg_loss if val_avg_loss < lowest_val_loss else lowest_val_loss
        return_acc = val_avg_acc if val_avg_acc > highest_val_acc else highest_val_acc
        return_dice = val_avg_dice_coeff if val_avg_dice_coeff > highest_val_dice else highest_val_dice

        log_string('Validation: Avg Dice Coefficient: {:.4f}, Avg Batch Acc: {:.4f}, Validation Avg Batch Loss: {:.8f} {}'.format(float(val_avg_dice_coeff), float(val_avg_acc), float(val_avg_loss), improved_str) + "\n")

        # Print out class-wise and total pixel-level accuracy
        log_string("Validation Class-wise Accuracies for Single Model:")
        class_accuracies = []
        for i in range(len(tma_4096_crop_class_printname)):
            if confusion_matrix[i, :].sum().item() == 0:
                class_accuracies.append(-1)  # 0 pixels of this class in validation set
            else:
                class_accuracies.append(confusion_matrix[i, i].item() / confusion_matrix[i, :].sum().item())

        for i in range(len(tma_4096_crop_class_printname)):
            log_string(tma_4096_crop_class_printname[i] + " Accuracy: " + str(round(class_accuracies[i], 5)))
        log_string("")

        log_string(",\t".join(tma_4096_crop_class_printname))
        # log_string(str(confusion_matrix))
        log_string('\n'.join(['\t'.join(['{:12.1f}'.format(num.item()) for num in row]) for row in confusion_matrix]))
        log_string("\n")

        ####################################
        # Save model if there is improvement
        ####################################
        # if improved:
        save_name = "best_model_{}.pth".format(model_num) if epoch_num is None else "best_model_{}_ep{}.pth".format(model_num, epoch_num)
        torch.save({
            'model_state_dict': model.state_dict(),
            'validation_loss': val_avg_loss
        }, os.path.join(SAVE_PATH, save_name))

        return val_avg_loss, return_loss, return_acc, return_dice, improved


def train():
    # Model loop
    for model_num in range(args["model_num"]):
        log_string("Training classifier #" + str(model_num) + "\n")
        gc.collect()
        classifier = PixelClassifier(num_classes=args["num_classes"], dim=args['featuremaps_dim'][-1])
        classifier.init_weights()
        log_string("Model architecture:\n" + str(classifier) + "\n")

        classifier = nn.DataParallel(classifier).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.SGD(classifier.parameters(), lr=args["pixel_classifier_lr"], momentum=0.9)
        # optimizer = optim.Adam(classifier.parameters(), lr=args["pixel_classifier_lr"])  # lr 0.001

        training_set = PixelFeaturesDataset(args["dataset_save_dir"], split="train")
        validation_set = PixelFeaturesDataset(args["dataset_save_dir"], split="val", val_fold=args["val_fold"])

        log_string("Length of train dataset: " + str(len(training_set)))
        log_string("Length of validation dataset: " + str(len(validation_set)) + "\n")

        log_string("Using WeightedRandomSampler in training dataset to balance classes in batch")
        sample_weights = [training_set.class_samp_weights[training_set.ground_truth[idx // training_set.img_pixel_feat_len][idx % training_set.img_pixel_feat_len]] for idx in range(len(training_set))]
        # sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(training_set), replacement=True)
        sampler = CustomWeightedRandomSampler(torch.DoubleTensor(sample_weights), len(training_set), replacement=True)

        train_loader = DataLoader(training_set, batch_size=args['batch_size'], sampler=sampler, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
        # train_loader = DataLoader(training_set, batch_size=args['batch_size'], shuffle=True, pin_memory=True)
        val_loader = DataLoader(validation_set, batch_size=args['featuremaps_dim'][1], shuffle=False, num_workers=32, pin_memory=True)

        # Training loop
        lowest_validation_loss = 10000000.
        lowest_train_loss = 10000000.
        highest_val_acc = 0.
        highest_val_dice = 0.
        # early_stopper = EarlyStopping(patience=args["early_stopping_patience"], min_delta=0.005)

        for epoch in range(args["epochs"]):
            log_string("Epoch " + str(epoch) + " starting...")
            classifier.train()
            total_train_loss = 0.
            summed_acc = 0.

            for batch_idx, (data, ground_truth) in enumerate(train_loader):
                # Move data and ground truth labels to cuda device, change ground truth labels to dtype long (integers)
                # data is [b, 6128], ground_truth is [64,]
                data, ground_truth = data.float().to(device), ground_truth.long().to(device)
                optimizer.zero_grad()

                pred_logits = classifier(data)  # pred shape [b, 7]  # 7 class output probabilities
                loss = criterion(pred_logits, ground_truth)
                acc = multi_acc(pred_logits, ground_truth)

                total_train_loss += loss.item()
                summed_acc += acc.item()

                loss.backward()
                optimizer.step()

            # Calculate average epoch loss for training set, log losses
            train_avg_loss = total_train_loss / len(train_loader)
            train_avg_acc = summed_acc / len(train_loader)
            tf_writer.add_scalar("Loss/train", train_avg_loss, epoch)
            tf_writer.add_scalar("Accuracy/train", train_avg_acc, epoch)

            if train_avg_loss < lowest_train_loss:
                train_improved_str = "(improved)"
                lowest_train_loss = train_avg_loss
            else:
                train_improved_str = ""

            log_string('Epoch {:03d}: - Train Avg Batch Accuracy: {:.3f}, Train Avg Batch Loss: {:.8f} {}'.format(
                epoch, train_avg_acc, train_avg_loss, train_improved_str))

            # Run validation
            val_avg_loss, lowest_validation_loss, highest_val_acc, highest_val_dice, val_improved = validation(classifier, model_num, val_loader, criterion, lowest_validation_loss, highest_val_acc, highest_val_dice, epoch)

            # Check for early stopping criteria
            # early_stopper(val_avg_loss)
            # if val_improved:
            #     early_stopper.counter = 0
            # if early_stopper.early_stop:
            #     break

        log_string("Done training classifier " + str(model_num) + "\n\n" + "---" * 40)


def main():
    # Log training configuration to training log
    log_string("Training configuration:")
    for key in args:
        log_string(key + ": " + str(args[key]))
    log_string("\n")

    # Train-validation loop, test once at end
    start_time = time.time()
    try:
        train()
    except KeyboardInterrupt:
        pass

    seconds_elapsed = time.time() - start_time
    log_string("Training took %s minutes" % str(seconds_elapsed / 60.))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default="/home/cougarnet.uh.edu/srizvi7/Desktop/Histopathology_Dataset_GAN/experiments/TMA_4096_tile.json")
    opts = parser.parse_args()
    args = json.load(open(opts.experiment, 'r'))

    # Create experiment training directory
    SAVE_PATH = args['experiment_dir']
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
        os.mkdir(os.path.join(SAVE_PATH, "python_file_saves"))
        print('Experiment folder created at: %s' % SAVE_PATH)

    os.system('cp %s %s' % (opts.experiment, SAVE_PATH))
    os.system("cp train_pixel_classifier.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))
    os.system("cp networks/pixel_classifier.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))
    os.system("cp pixel_features_dataset.py {}".format(os.path.join(SAVE_PATH, "python_file_saves")))

    expt_name = args["experiment_dir"].split("/")[-1]
    tf_writer = SummaryWriter('runs/' + expt_name)

    logger = open(os.path.join(SAVE_PATH, "training_log.txt"), "w")
    main()
    logger.close()

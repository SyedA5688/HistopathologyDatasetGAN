import os
import gc
import json
import time
import argparse
import numpy as np
from random import seed

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.utils import multi_acc, EarlyStopping
from utils.visualization_utils import plot_loss_curves, plot_acc_curves
from networks.pixel_classifier import PixelClassifier
from pixel_features_dataset import PixelFeaturesDataset, ClassBalancedPixelFeaturesDataset


chosen_seed = 0
seed(chosen_seed)
np.random.seed(chosen_seed)
torch.manual_seed(chosen_seed)
torch.cuda.manual_seed_all(chosen_seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ToDo: After pixel classifier baseline, consider augmenting scarce pixel classes


def log_string(str1):
    print(str1)
    logger.write(str1 + "\n")
    logger.flush()


def validation(model, model_num, val_loader, criterion, lowest_val_loss, highest_val_acc):
    with torch.no_grad():
        model.eval()

        ################################
        # Evaluate on validation dataset
        ################################
        val_total_loss = 0.
        summed_acc = 0.
        # Consider saving produced mask at every val epoch, or when validation improves

        for batch_idx, (data, ground_truth) in enumerate(val_loader):
            data, ground_truth = data.to(device), ground_truth.long().to(device)

            pred_logits = model(data)
            loss = criterion(pred_logits, ground_truth)
            acc = multi_acc(pred_logits, ground_truth)

            val_total_loss += loss.item()
            summed_acc += acc.item()

        ############################
        # Display validation results
        ############################
        val_avg_loss = val_total_loss / (batch_idx + 1)
        val_avg_acc = summed_acc / (batch_idx + 1)

        if val_avg_acc > highest_val_acc:  # val_avg_loss < lowest_val_loss or
            improved, improved_str = True, "(improved accuracy or val loss)"
        else:
            improved, improved_str = False, ""

        return_loss = val_avg_loss if val_avg_loss < lowest_val_loss else lowest_val_loss
        return_acc = val_avg_acc if val_avg_acc > highest_val_acc else highest_val_acc

        log_string('Validation Avg Batch Acc: {:.4f}, Validation Avg Batch Loss: {:.8f} {}'.format(float(val_avg_acc), float(val_avg_loss), improved_str) + "\n")

        ####################################
        # Save model if there is improvement
        ####################################
        if improved:
            # Overwrite best saved model each time, only keep best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'validation_loss': val_avg_loss
            }, os.path.join(SAVE_PATH, "best_model_" + str(model_num) + ".pth"))

        return val_avg_loss, return_loss, val_avg_acc, return_acc, improved


def train():
    # Model loop
    for model_num in range(args["model_num"]):
        log_string("Training classifier #" + str(model_num) + "\n")
        gc.collect()
        classifier = PixelClassifier(num_classes=args["num_classes"], dim=args['featuremaps_dim'][-1])
        classifier.init_weights()
        log_string("Model architecture:\n" + str(classifier) + "\n")

        classifier = nn.DataParallel(classifier).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(classifier.parameters(), lr=args["pixel_classifier_lr"])

        # Create datasets and dataloaders, specific for this model. Random selection of pixel features
        training_set = PixelFeaturesDataset(args["pixel_feat_save_dir"], split="train")
        validation_set = PixelFeaturesDataset(args["pixel_feat_save_dir"], split="val")

        log_string("Length of train dataset: " + str(len(training_set)))
        log_string("Length of validation dataset: " + str(len(validation_set)) + "\n")

        # log_string("Training Dataset Balanced class pixel counts:")
        # for idx, key in enumerate(training_set.class_counts):
        #     log_string("Class " + str(idx) + ": " + str(training_set.class_counts[idx]))
        # log_string("\n")

        train_loader = DataLoader(training_set, batch_size=args['batch_size'], shuffle=True)
        val_loader = DataLoader(validation_set, batch_size=args['batch_size'], shuffle=False)

        # Training loop
        lowest_validation_loss = 10000000
        lowest_train_loss = 10000000
        highest_val_acc = 0
        train_losses.clear()
        val_losses.clear()
        train_acc_list.clear()
        val_acc_list.clear()
        early_stopper = EarlyStopping(patience=args["early_stopping_patience"], min_delta=0.005)

        for epoch in range(args["epochs"]):
            log_string("Epoch " + str(epoch) + " starting...")
            classifier.train()
            total_train_loss = 0.
            summed_acc = 0.

            for batch_idx, (data, ground_truth) in enumerate(train_loader):
                # Move data and ground truth labels to cuda device, change ground truth labels to dtype long (integers)
                data, ground_truth = data.to(device), ground_truth.long().to(device)  # data is [b, 6080], ground_truth is [64,]
                optimizer.zero_grad()

                pred_logits = classifier(data)  # pred shape [b, 13]  # 13 class output probabilities
                loss = criterion(pred_logits, ground_truth)
                acc = multi_acc(pred_logits, ground_truth)

                total_train_loss += loss.item()
                summed_acc += acc.item()

                loss.backward()
                optimizer.step()

            # Calculate average epoch loss for training set, log losses
            train_avg_loss = total_train_loss / (batch_idx + 1)  # Divide by # of batches
            train_avg_acc = summed_acc / (batch_idx + 1)
            train_losses.append(float(train_avg_loss))
            train_acc_list.append(float(train_avg_acc))

            if train_avg_loss < lowest_train_loss:
                train_improved_str = "(improved)"
                lowest_train_loss = train_avg_loss
            else:
                train_improved_str = ""

            log_string('Epoch {:03d} Overall Results - Train Avg Batch Accuracy: {:.3f}, Train Avg Batch Loss: {:.8f} {}'.format(
                epoch, train_avg_acc, train_avg_loss, train_improved_str))

            # Run validation
            val_avg_loss, lowest_validation_loss, val_avg_acc, highest_val_acc, val_improved = validation(classifier, model_num, val_loader, criterion, lowest_validation_loss, highest_val_acc)
            val_losses.append(float(val_avg_loss))
            val_acc_list.append(float(val_avg_acc))

            # Check for early stopping criteria
            early_stopper(val_avg_loss)
            if val_improved:  # Eary stopper won't count increases in accuracy
                early_stopper.counter = 0
            if early_stopper.early_stop:
                break

        log_string("Done training classifier " + str(model_num) + "\n\n" + "---" * 40)
        plot_loss_curves(train_losses, val_losses, model_num, SAVE_PATH)
        plot_acc_curves(train_acc_list, val_acc_list, model_num, SAVE_PATH)


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
        # Catch if there is a Keyboard Interruption (ctrl-c). Plot train-val loss curves
        plot_loss_curves(train_losses, val_losses, "last", SAVE_PATH)

    seconds_elapsed = time.time() - start_time
    log_string("Training took %s minutes" % str(seconds_elapsed / 60.))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str)
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

    # Declare here so can plot train/val loss curves if training stops by ctrl-c
    train_losses = []
    val_losses = []
    train_acc_list = []
    val_acc_list = []

    logger = open(os.path.join(SAVE_PATH, "training_log.txt"), "w")
    main()
    logger.close()

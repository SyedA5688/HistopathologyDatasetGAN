import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_loss_curves(train_losses, val_losses, model_num, save_path):
    if type(model_num) != str:
        model_num = str(model_num)

    if len(train_losses) != len(val_losses):
        min_len = min(len(train_losses), len(val_losses))
        train_losses = train_losses[0:min_len]
        val_losses = val_losses[0:min_len]

    time = list(range(len(train_losses)))
    visual_df = pd.DataFrame({
        "Train Loss": train_losses,
        "Validation Loss": val_losses,
        "Epoch": time
    })

    sns.lineplot(x='Epoch', y='value', hue='variable', data=pd.melt(visual_df, ['Epoch']))
    plt.title("Pixel Classifier #" + str(model_num) + " Loss Curves")
    plt.savefig(save_path + '/pixel_classifier_' + str(model_num) + '_loss_curves.png')
    plt.clf()


def plot_acc_curves(train_acc, val_acc, model_num, save_path):
    if type(model_num) != str:
        model_num = str(model_num)

    if len(train_acc) != len(val_acc):
        min_len = min(len(train_acc), len(val_acc))
        train_acc = train_acc[0:min_len]
        val_acc = train_acc[0:min_len]

    time = list(range(len(train_acc)))
    visual_df = pd.DataFrame({
        "Train Accuracy": train_acc,
        "Validation Accuracy": val_acc,
        "Epoch": time
    })

    sns.lineplot(x='Epoch', y='value', hue='variable', data=pd.melt(visual_df, ['Epoch']))
    plt.title("Pixel Classifier #" + str(model_num) + " Accuracy Curves")
    plt.savefig(save_path + '/pixel_classifier_' + str(model_num) + '_acc_curves.png')
    plt.clf()

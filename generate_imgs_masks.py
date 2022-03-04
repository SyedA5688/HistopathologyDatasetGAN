import os
import json
import argparse
import functools
from random import seed

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils.utils import oht_to_scalar, colorize_mask, latent_to_image
from utils.data_util import tma_12_palette
from networks.pixel_classifier import PixelClassifier
from load_networks import load_stylegan2_ada, get_upsamplers


chosen_seed = 0
seed(chosen_seed)
np.random.seed(chosen_seed)
torch.manual_seed(chosen_seed)
torch.cuda.manual_seed_all(chosen_seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


def generate_one_image_mask_pair(idx, g_all, upsamplers, classifier):
    print("Generating image and mask", idx, "...")
    latent_np = np.random.RandomState(idx).randn(1, g_all.z_dim)  # shape [1, 512]
    latent_np = torch.from_numpy(latent_np).to(device)
    img, upsampled_featmap = latent_to_image(g_all, upsamplers, latent_np, is_w_latent=False,
                                             dim=args['featuremaps_dim'][1], truncation_psi=args['truncation'],
                                             return_upsampled_featuremaps=True, device=device)
    # Save generated image
    img = Image.fromarray(img[0].cpu().numpy(), 'RGB')
    image_path = os.path.join(SAVE_PATH, "generated_image_{}.jpg".format(idx))
    img.save(image_path)

    # Reshape upsampled featuremap
    b, ch, h, w = upsampled_featmap.shape  # [1, 6080, 1024, 1024]
    pixel_features_list = []
    for row in range(h):
        for col in range(w):
            pixel_features_list.append(upsampled_featmap[:, :, row, col])  # Append [1, 6080]
    pixel_features_list = torch.cat(pixel_features_list, dim=0).to(device)  # [1024*1024, 6080]

    # Get 7-class mask predictions
    all_pixel_preds = []
    with torch.no_grad():
        for i in range(0, len(pixel_features_list), 1024):
            pred_logits = classifier(pixel_features_list[i:i+1024])  # pred shape [b, 7]  # 7 classes
            pixel_preds = oht_to_scalar(pred_logits)
            all_pixel_preds.append(pixel_preds.unsqueeze(0))
        all_pixel_preds = torch.cat(all_pixel_preds, dim=0).detach().cpu().numpy()

    # Create arteriole mask - only task arteriole class predictions in center 512x512
    arteriole_mask = np.zeros((1024, 1024))
    for row in range(0, 1024):
        for col in range(0, 1024):
            if all_pixel_preds[row][col] == 1:  # If predicted Arteriole
                arteriole_mask[row][col] = 255

    # Save generated masks
    np.save(os.path.join(SAVE_PATH, "multiclass_mask_{}".format(idx)), all_pixel_preds)
    colorized_multiclass_mask = colorize_mask(all_pixel_preds, tma_12_palette)
    plt.imsave(os.path.join(SAVE_PATH, "multiclass_mask_img_{}.jpg".format(idx)), colorized_multiclass_mask)

    np.save(os.path.join(SAVE_PATH, "arteriole_mask_{}".format(idx)), arteriole_mask)
    image_path = os.path.join(SAVE_PATH, "arteriole_mask_img_{}.jpg".format(idx))
    plt.imsave(image_path, arteriole_mask)

def main():
    g_all, avg_latent = load_stylegan2_ada(args)
    # Line for inference on CPU
    g_all.forward = functools.partial(g_all.forward, force_fp32=True)
    upsamplers = get_upsamplers(args)

    # Load pretrained pixel classifier
    model = PixelClassifier(num_classes=args["num_classes"], dim=args['featuremaps_dim'][-1])
    checkpoint = torch.load(model_path)['model_state_dict']
    prefix = 'module.'
    n_clip = len(prefix)
    adapted_dict = {k[n_clip:]: v for k, v in checkpoint.items() if k.startswith(prefix)}

    model.load_state_dict(adapted_dict)
    model.to(device)
    model.eval()

    for i in range(NUM_IMAGES):
        generate_one_image_mask_pair(i, g_all, upsamplers, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str,
                        default="/home/cougarnet.uh.edu/srizvi7/Desktop/Histopathology_Dataset_GAN/experiments/TMA_Arteriole_stylegan2_ada.json")

    opts = parser.parse_args()
    args = json.load(open(opts.experiment, 'r'))

    SAVE_PATH = os.path.join(args["dataset_save_dir"], "artificial_dataset_5000")
    model_path = os.path.join(args["experiment_dir"], "best_model_" + str(0) + "_ep0.pth")
    NUM_IMAGES = 5000

    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    main()

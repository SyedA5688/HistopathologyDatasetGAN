import os
import json
import argparse
import functools

import torch
import numpy as np

from PIL import Image
from load_networks import load_stylegan2_ada, get_upsamplers
from utils.utils import latent_to_image

np.random.seed(0)
torch.manual_seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'


def generate_data(num_sample):
    if not os.path.exists(args["dataset_save_dir"]):
        os.system('mkdir -p %s' % args["dataset_save_dir"])
        print('Experiment folder created at: %s' % args["dataset_save_dir"])

    g_all, avg_latent = load_stylegan2_ada(args)
    # Line for inference on CPU
    # g_all.forward = functools.partial(g_all.forward, force_fp32=True)
    # g_all.synthesis.forward = functools.partial(g_all.synthesis.forward, force_fp32=True)
    upsamplers = get_upsamplers(args)

    # Save avg_latent
    mean_latent_save_path = os.path.join(args["dataset_save_dir"], "avg_latent_stylegan2.npy")
    np.save(mean_latent_save_path, avg_latent[0].detach().cpu().numpy())

    with torch.no_grad():
        latent_cache = []
        print("Num_samples to generate: ", num_sample)

        for i in range(num_sample):
            print("Generating", i, "Out of:", num_sample)

            if i == 0:
                latent = avg_latent.to(device)
                img, _ = latent_to_image(g_all, upsamplers, latent, is_w_latent=True, dim=args['featuremaps_dim'][1], return_upsampled_featuremaps=False)
            else:
                # Use index as random seed, for reproducible generation
                latent_np = np.random.RandomState(i).randn(1, g_all.z_dim)
                latent_cache.append(np.copy(latent_np))
                latent = torch.from_numpy(latent_np).to(device)
                img, _ = latent_to_image(g_all, upsamplers, latent, is_w_latent=False, dim=args['featuremaps_dim'][1], truncation_psi=args['truncation'], return_upsampled_featuremaps=False)

            img = Image.fromarray(img[0].cpu().numpy(), 'RGB')
            image_path = os.path.join(args["dataset_save_dir"], "image_%d.png" % i)
            img.save(image_path)

        latent_cache = np.concatenate(latent_cache, 0)
        latent_save_path = os.path.join(args["dataset_save_dir"], "latent_stylegan2.npy")
        np.save(latent_save_path, latent_cache)

        reconstruct_save_path = os.path.join(args["dataset_save_dir"], "reconstructed_images")
        if not os.path.exists(reconstruct_save_path):
            os.system('mkdir -p %s' % reconstruct_save_path)

        for idx, latent_np in enumerate(latent_cache):
            latent = torch.from_numpy(latent_np).to(device).unsqueeze(0)
            img, _ = latent_to_image(g_all, upsamplers, latent, is_w_latent=False, dim=args['featuremaps_dim'][1],
                                     truncation_psi=args['truncation'], return_upsampled_featuremaps=False)

            img = Image.fromarray(img[0].cpu().numpy(), 'RGB')
            image_path = os.path.join(reconstruct_save_path, "reconstructed_image_%d.png" % (idx + 1))
            img.save(image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default="/home/cougarnet.uh.edu/srizvi7/Desktop/Histopathology_Dataset_GAN/experiments/TMA_4096_tile.json")
    parser.add_argument('--num_sample', type=int,  default=500)

    opts = parser.parse_args()
    args = json.load(open(opts.experiment, 'r'))
    generate_data(opts.num_sample)

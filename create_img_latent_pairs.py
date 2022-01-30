"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import json
import argparse

import torch
import numpy as np

from PIL import Image
from load_networks import load_stylegan2_ada, get_upsamplers
from utils.utils import latent_to_image

np.random.seed(0)
torch.manual_seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def generate_data(num_sample, save_path):
    if not os.path.exists(save_path):
        os.system('mkdir -p %s' % save_path)
        print('Experiment folder created at: %s' % save_path)

    g_all, avg_latent = load_stylegan2_ada(args)
    upsamplers = get_upsamplers(args)

    # save avg_latent for reproducibility
    mean_latent_save_path = os.path.join(save_path, "avg_latent_stylegan2.npy")
    np.save(mean_latent_save_path, avg_latent[0].detach().cpu().numpy())

    with torch.no_grad():
        latent_cache = []

        print("Num_samples to generate: ", num_sample)

        for i in range(num_sample):
            if i % 10 == 0:
                print("Generating", i, "Out of:", num_sample)

            if i == 0:
                latent = avg_latent.to(device)
                img, _ = latent_to_image(g_all, upsamplers, latent, is_w_latent=True, dim=args['featuremaps_dim'][1], return_upsampled_featuremaps=False)
            else:
                # Use index as random seed, for reproducible generation
                latent_np = np.random.RandomState(i).randn(1, g_all.z_dim)
                latent_cache.append(np.copy(latent_np))
                latent = torch.from_numpy(latent_np).to(device)
                img, _ = latent_to_image(g_all, upsamplers, latent, is_w_latent=False, dim=args['featuremaps_dim'][1],
                                         truncation_psi=args['truncation'], return_upsampled_featuremaps=False)

            img = Image.fromarray(img[0].cpu().numpy(), 'RGB')
            image_path = os.path.join(save_path, "image_%d.jpg" % i)
            img.save(image_path)

        latent_cache = np.concatenate(latent_cache, 0)
        latent_save_path = os.path.join(save_path, "latent_stylegan2.npy")
        np.save(latent_save_path, latent_cache)

        reconstruct_save_path = os.path.join(save_path, "reconstructed_images")
        if not os.path.exists(reconstruct_save_path):
            os.system('mkdir -p %s' % reconstruct_save_path)

        for idx, latent_np in enumerate(latent_cache):
            latent = torch.from_numpy(latent_np).to(device).unsqueeze(0)
            img, _ = latent_to_image(g_all, upsamplers, latent, is_w_latent=False, dim=args['featuremaps_dim'][1],
                                     truncation_psi=args['truncation'], return_upsampled_featuremaps=False)

            img = Image.fromarray(img[0].cpu().numpy(), 'RGB')
            image_path = os.path.join(reconstruct_save_path, "reconstructed_image_%d.jpg" % (idx + 1))
            img.save(image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--num_sample', type=int,  default=100)
    parser.add_argument('--save_path', type=str)
    cmdline_args = parser.parse_args()

    args = json.load(open(cmdline_args.experiment, 'r'))
    print("Experiment options from file:", args)
    generate_data(cmdline_args.num_sample, cmdline_args.save_path)

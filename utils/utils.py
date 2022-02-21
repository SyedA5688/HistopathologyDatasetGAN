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
import torch
import numpy as np
import torch.nn as nn
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


def multi_acc(y_pred, y_label):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_label).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = acc * 100

    return acc


def oht_to_scalar(y_pred):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    return y_pred_tags


def latent_to_image(g_all, upsamplers, latents, is_w_latent=False, dim=1024, truncation_psi=0.7, return_upsampled_featuremaps=False, process_out=True, noise_mode='const'):
    """
    :param g_all: Generator network, consisting of g_mapping and g_synthesis modules
    :param upsamplers: list of upsampling layers
    :param latents: latent codes to turn into images
    :param is_w_latent: whether or not latent codes are w latents or z latents. If false, latents will be passed
        through mapping network first
    :param return_upsampled_featuremaps: whether or not to compute returned upsampled featuremaps
    :param process_out: whether or not to process output images
    :param dim: dimension to upsample featuremaps to
    :return: images: torch tensor shape [len(latents), dim, dim],
             upsampled_featuremaps: torch tensor shape [sum of featuremaps dimension, dim, dim]
    """
    with torch.no_grad():
        if is_w_latent:
            w_latents = latents
            images, affine_layers = g_all.synthesis(w_latents)
        else:
            # Pass conditioning label here
            images, affine_layers = g_all(latents, c=0, truncation_psi=truncation_psi, noise_mode=noise_mode)

        num_features = 0
        for item in affine_layers:
            num_features += item.shape[1]

        upsampled_featuremaps = None
        if return_upsampled_featuremaps:
            upsampled_featuremaps = torch.FloatTensor(1, num_features, dim, dim).to(device)
            start_channel_index = 0
            for i in range(len(affine_layers)):
                len_channel = affine_layers[i].shape[1]
                upsampled_featuremaps[:, start_channel_index:start_channel_index + len_channel] = upsamplers[i](affine_layers[i])
                start_channel_index += len_channel

            upsampled_featuremaps = upsampled_featuremaps.cpu()
            torch.cuda.empty_cache()

        if process_out:
            images = (images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    return images, upsampled_featuremaps


def get_stylegan_latent(g_all, latents):
    style_latents = g_all.mapping(latents)
    return style_latents


def colorize_mask(mask, palette):
    """
    :param mask: numpy array of the mask
    :param palette:
    :return: colorized mask
    """

    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return np.array(new_mask.convert('RGB'))


def get_label_stas(data_loader):
    count_dict = {}
    for i in range(data_loader.__len__()):
        x, y = data_loader.__getitem__(i)
        if int(y.item()) not in count_dict:
            count_dict[int(y.item())] = 1
        else:
            count_dict[int(y.item())] += 1

    return count_dict


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            # print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                # print('INFO: Early stopping')
                self.early_stop = True

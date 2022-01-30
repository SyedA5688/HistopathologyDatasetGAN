import os

import torch
import torch.nn as nn
import numpy as np
import dnnlib

from torch_utils import misc
from networks import legacy
from utils.utils import Interpolate


np.random.seed(0)
torch.manual_seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_stylegan2_ada(args):
    assert args['stylegan_ver'] == "2_ADA", "Please use StyleGAN2-ADA"
    assert args['category'] is not None, "Please specify a category and accompanying stylegan2_ada hyperparameters"

    # Define dictionary with StyleGan2ADA Parameters
    g_kwargs = args['G_kwargs']
    common_kwargs = dict(c_dim=0, img_resolution=args['resolution'], img_channels=3)
    g_all = dnnlib.util.construct_class_by_name(**g_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    g_all.eval()

    # Load saved pickle file
    with dnnlib.util.open_url(args['stylegan_checkpoint']) as f:
        g_checkpoint = legacy.load_network_pkl(f)['G_ema'].to(device)
        misc.copy_params_and_buffers(g_checkpoint, g_all, require_all=False)

    if args['average_latent'] == '':
        avg_latent = g_all.make_mean_latent(8000, truncation_psi=args['truncation'])
    else:
        avg_latent = np.load(args['average_latent'])
        avg_latent = torch.from_numpy(avg_latent).type(torch.FloatTensor).to(device)

    return g_all, avg_latent


def get_upsamplers(args):
    res = args['resolution']
    mode = args['upsample_mode']
    upsamplers = [nn.Upsample(scale_factor=res / 4, mode=mode),
                  nn.Upsample(scale_factor=res / 4, mode=mode),
                  nn.Upsample(scale_factor=res / 8, mode=mode),
                  nn.Upsample(scale_factor=res / 8, mode=mode),
                  nn.Upsample(scale_factor=res / 16, mode=mode),
                  nn.Upsample(scale_factor=res / 16, mode=mode),
                  nn.Upsample(scale_factor=res / 32, mode=mode),
                  nn.Upsample(scale_factor=res / 32, mode=mode),
                  nn.Upsample(scale_factor=res / 64, mode=mode),
                  nn.Upsample(scale_factor=res / 64, mode=mode),
                  nn.Upsample(scale_factor=res / 128, mode=mode),
                  nn.Upsample(scale_factor=res / 128, mode=mode),
                  nn.Upsample(scale_factor=res / 256, mode=mode),
                  nn.Upsample(scale_factor=res / 256, mode=mode)]

    if res > 256:
        upsamplers.append(nn.Upsample(scale_factor=res / 512, mode=mode))
        upsamplers.append(nn.Upsample(scale_factor=res / 512, mode=mode))

    if res > 512:
        upsamplers.append(Interpolate(res, 'bilinear'))
        upsamplers.append(Interpolate(res, 'bilinear'))

    return upsamplers

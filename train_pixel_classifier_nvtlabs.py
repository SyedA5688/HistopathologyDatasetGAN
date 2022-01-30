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
import gc
import json
import pickle
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import scipy.misc
import scipy.stats

from PIL import Image
from torch.distributions import Categorical
from networks.pixel_classifier import PixelClassifier
from load_networks import load_stylegan2_ada, get_upsamplers
from utils.utils import multi_acc, colorize_mask, get_label_stas, latent_to_image, oht_to_scalar


np.random.seed(0)
torch.manual_seed(0)
device_ids = [0]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152


class TrainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


def generate_data(args, checkpoint_path, num_sample, start_step=0, vis=True):
    """
    Syed's notes: this function loads checkpoints for trained emsemble pixel classifiers
    and predicts masks for num_sample generated images
    Args:
        args:
        checkpoint_path:
        num_sample:
        start_step:
        vis:

    Returns:

    """
    if args['category'] == 'car':
        from utils.data_util import car_20_palette as palette
    elif args['category'] == 'face':
        from utils.data_util import face_palette as palette
    elif args['category'] == 'bedroom':
        from utils.data_util import bedroom_palette as palette
    elif args['category'] == 'cat':
        from utils.data_util import cat_palette as palette
    else:
        assert False
    if not vis:
        result_path = os.path.join(checkpoint_path, 'samples' )
    else:
        result_path = os.path.join(checkpoint_path, 'vis_%d'%num_sample)
    if os.path.exists(result_path):
        pass
    else:
        os.system('mkdir -p %s' % (result_path))
        print('Experiment folder created at: %s' % (result_path))

    g_all, avg_latent = load_stylegan2_ada(args)
    upsamplers = get_upsamplers(args)

    classifier_list = []
    for MODEL_NUMBER in range(args['model_num']):
        print('MODEL_NUMBER', MODEL_NUMBER)

        classifier = PixelClassifier(numpy_class=args['number_class'], dim=args['dim'][-1])
        classifier =  nn.DataParallel(classifier, device_ids=device_ids).cuda()

        checkpoint = torch.load(os.path.join(checkpoint_path, 'model_' + str(MODEL_NUMBER) + '.pth'))

        classifier.load_state_dict(checkpoint['model_state_dict'])


        classifier.eval()
        classifier_list.append(classifier)

    softmax_f = nn.Softmax(dim=1)
    with torch.no_grad():
        latent_cache = []
        image_cache = []
        seg_cache = []
        entropy_calculate = []
        results = []
        np.random.seed(start_step)
        count_step = start_step



        print( "num_sample: ", num_sample)

        for i in range(num_sample):
            if i % 100 == 0:
                print("Genearte", i, "Out of:", num_sample)

            curr_result = {}

            latent = np.random.randn(1, 512)

            curr_result['latent'] = latent


            latent = torch.from_numpy(latent).type(torch.FloatTensor).to(device)
            latent_cache.append(latent)

            img, affine_layers = latent_to_image(g_all, upsamplers, latent, dim=args['dim'][1], return_upsampled_layers=True)

            if args['dim'][0] != args['dim'][1]:
                img = img[:, 64:448][0]
            else:
                img = img[0]

            image_cache.append(img)
            if args['dim'][0] != args['dim'][1]:
                affine_layers = affine_layers[:, :, 64:448]
            affine_layers = affine_layers[0]

            affine_layers = affine_layers.reshape(args['dim'][-1], -1).transpose(1, 0)

            all_seg = []
            all_entropy = []
            mean_seg = None

            seg_mode_ensemble = []
            for MODEL_NUMBER in range(args['model_num']):
                classifier = classifier_list[MODEL_NUMBER]

                img_seg = classifier(affine_layers)

                img_seg = img_seg.squeeze()


                entropy = Categorical(logits=img_seg).entropy()
                all_entropy.append(entropy)

                all_seg.append(img_seg)
                if mean_seg is None:
                    mean_seg = softmax_f(img_seg)
                else:
                    mean_seg += softmax_f(img_seg)

                img_seg_final = oht_to_scalar(img_seg)
                img_seg_final = img_seg_final.reshape(args['dim'][0], args['dim'][1], 1)
                img_seg_final = img_seg_final.cpu().detach().numpy()

                seg_mode_ensemble.append(img_seg_final)

            mean_seg = mean_seg / len(all_seg)

            full_entropy = Categorical(mean_seg).entropy()

            js = full_entropy - torch.mean(torch.stack(all_entropy), 0)

            top_k = js.sort()[0][- int(js.shape[0] / 10):].mean()
            entropy_calculate.append(top_k)


            img_seg_final = np.concatenate(seg_mode_ensemble, axis=-1)
            img_seg_final = scipy.stats.mode(img_seg_final, 2)[0].reshape(args['dim'][0], args['dim'][1])
            del (affine_layers)
            if vis:

                color_mask = 0.7 * colorize_mask(img_seg_final, palette) + 0.3 * img

                scipy.misc.imsave(os.path.join(result_path, "vis_" + str(i) + '.jpg'),
                                  color_mask.astype(np.uint8))
                scipy.misc.imsave(os.path.join(result_path, "vis_" + str(i) + '_image.jpg'),
                                  img.astype(np.uint8))
            else:
                seg_cache.append(img_seg_final)
                curr_result['uncertrainty_score'] = top_k.item()
                image_label_name = os.path.join(result_path, 'label_' + str(count_step) + '.png')
                image_name = os.path.join(result_path,  str(count_step) + '.png')

                js_name = os.path.join(result_path, str(count_step) + '.npy')
                img = Image.fromarray(img)
                img_seg = Image.fromarray(img_seg_final.astype('uint8'))
                js = js.cpu().numpy().reshape(args['dim'][0], args['dim'][1])
                img.save(image_name)
                img_seg.save(image_label_name)
                np.save(js_name, js)
                curr_result['image_name'] = image_name
                curr_result['image_label_name'] = image_label_name
                curr_result['js_name'] = js_name
                count_step += 1


                results.append(curr_result)
                if i % 1000 == 0 and i != 0:
                    with open(os.path.join(result_path, str(i) + "_" + str(start_step) + '.pickle'), 'wb') as f:
                        pickle.dump(results, f)

        with open(os.path.join(result_path, str(num_sample) + "_" + str(start_step) + '.pickle'), 'wb') as f:
            pickle.dump(results, f)


def prepare_data(args, palette):
    g_all, avg_latent = load_stylegan2_ada(args)
    upsamplers = get_upsamplers(args)
    latent_all = np.load(args['annotation_image_latent_path'])
    latent_all = torch.from_numpy(latent_all).cuda()

    # load annotated mask
    mask_list = []
    im_list = []
    latent_all = latent_all[:args['max_training']]
    num_data = len(latent_all)

    for i in range(len(latent_all)):

        if i >= args['max_training']:
            break
        name = 'image_mask%0d.npy' % i

        im_frame = np.load(os.path.join( args['annotation_mask_path'] , name))
        mask = np.array(im_frame)
        mask =  cv2.resize(np.squeeze(mask), dsize=(args['dim'][1], args['dim'][0]), interpolation=cv2.INTER_NEAREST)

        mask_list.append(mask)

        im_name = os.path.join( args['annotation_mask_path'], 'image_%d.jpg' % i)
        img = Image.open(im_name)
        img = img.resize((args['dim'][1], args['dim'][0]))

        im_list.append(np.array(img))

    # delete small annotation error
    for i in range(len(mask_list)):  # clean up artifacts in the annotation, must do
        for target in range(1, 50):
            if (mask_list[i] == target).sum() < 30:
                mask_list[i][mask_list[i] == target] = 0


    all_mask = np.stack(mask_list)


    # 3. Generate ALL training data for training pixel classifier
    all_feature_maps_train = np.zeros((args['dim'][0] * args['dim'][1] * len(latent_all), args['dim'][2]), dtype=np.float16)  # e.g. cars: 512*512*16, 5088
    all_mask_train = np.zeros((args['dim'][0] * args['dim'][1] * len(latent_all),), dtype=np.float16)  # shape [512*512*16]


    vis = []
    for i in range(len(latent_all) ):

        gc.collect()

        latent_input = latent_all[i].float()

        img, feature_maps = latent_to_image(g_all, upsamplers, latent_input.unsqueeze(0), dim=args['dim'][1], return_upsampled_layers=True, use_style_latents=args['annotation_data_from_w'])

        mask = all_mask[i:i + 1]
        feature_maps = feature_maps.permute(0, 2, 3, 1)

        feature_maps = feature_maps.reshape(-1, args['dim'][2])  # [512*512, 5088]
        new_mask =  np.squeeze(mask)

        mask = mask.reshape(-1)  # shape [512*512]

        all_feature_maps_train[args['dim'][0] * args['dim'][1] * i: args['dim'][0] * args['dim'][1] * i + args['dim'][0] * args['dim'][1]] = feature_maps.cpu().detach().numpy().astype(np.float16)
        all_mask_train[args['dim'][0] * args['dim'][1] * i:args['dim'][0] * args['dim'][1] * i + args['dim'][0] * args['dim'][1]] = mask.astype(np.float16)

    #     img_show =  cv2.resize(np.squeeze(img[0]), dsize=(args['dim'][1], args['dim'][1]), interpolation=cv2.INTER_NEAREST)
    #
    #     curr_vis = np.concatenate( [im_list[i], img_show, colorize_mask(new_mask, palette)], 0 )
    #
    #     vis.append( curr_vis )
    #
    #
    # vis = np.concatenate(vis, 1)
    # scipy.misc.imsave(os.path.join(args['exp_dir'], "train_data.jpg"), vis)

    return all_feature_maps_train, all_mask_train, num_data


def main(args
         ):

    if args['category'] == 'car':
        from utils.data_util import car_20_palette as palette
    elif args['category'] == 'face':
        from utils.data_util import face_palette as palette
    elif args['category'] == 'bedroom':
        from utils.data_util import bedroom_palette as palette
    elif args['category'] == 'cat':
        from utils.data_util import cat_palette as palette


    all_feature_maps_train_all, all_mask_train_all, num_data = prepare_data(args, palette)


    train_data = TrainData(torch.FloatTensor(all_feature_maps_train_all),
                           torch.FloatTensor(all_mask_train_all))


    count_dict = get_label_stas(train_data)

    max_label = max([*count_dict])
    print(" *********************** max_label " + str(max_label) + " ***********************")


    print(" *********************** Current number data " + str(num_data) + " ***********************")


    batch_size = args['batch_size']

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    print(" *********************** Current dataloader length " +  str(len(train_loader)) + " ***********************")

    for MODEL_NUMBER in range(args['model_num']):

        gc.collect()

        classifier = pixel_classifier(numpy_class=(max_label + 1), dim=args['dim'][-1])

        classifier.init_weights()

        classifier = nn.DataParallel(classifier, device_ids=device_ids).cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)
        classifier.train()


        iteration = 0
        break_count = 0
        best_loss = 10000000
        stop_sign = 0
        for epoch in range(100):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_batch = y_batch.type(torch.long)
                y_batch = y_batch.type(torch.long)

                optimizer.zero_grad()
                y_pred = classifier(X_batch)
                loss = criterion(y_pred, y_batch)
                acc = multi_acc(y_pred, y_batch)

                loss.backward()
                optimizer.step()

                iteration += 1
                if iteration % 1000 == 0:
                    print('Epoch : ', str(epoch), 'iteration', iteration, 'loss', loss.item(), 'acc', acc)
                    gc.collect()

                if iteration % 5000 == 0:
                    model_path = os.path.join(args['exp_dir'],
                                              'model_20parts_iter' +  str(iteration) + '_number_' + str(MODEL_NUMBER) + '.pth')
                    print('Save checkpoint, Epoch : ', str(epoch), ' Path: ', model_path)

                    torch.save({'model_state_dict': classifier.state_dict()},
                               model_path)

                if epoch > 3:
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        break_count = 0
                    else:
                        break_count += 1

                    if break_count > 50:
                        stop_sign = 1
                        print("*************** Break, Total iters,", iteration, ", at epoch", str(epoch), "***************")
                        break

            if stop_sign == 1:
                break

        gc.collect()
        model_path = os.path.join(args['exp_dir'],
                                  'model_' + str(MODEL_NUMBER) + '.pth')
        MODEL_NUMBER += 1
        print('save to:',model_path)
        torch.save({'model_state_dict': classifier.state_dict()},
                   model_path)
        gc.collect()


        gc.collect()
        torch.cuda.empty_cache()    # clear cache memory on GPU


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str)
    parser.add_argument('--exp_dir', type=str,  default="")
    parser.add_argument('--generate_data', type=bool, default=False)
    parser.add_argument('--save_vis', type=bool, default=False)
    parser.add_argument('--start_step', type=int, default=0)

    parser.add_argument('--resume', type=str,  default="")
    parser.add_argument('--num_sample', type=int,  default=1000)

    args = parser.parse_args()

    opts = json.load(open(args.exp, 'r'))
    print("Opt", opts)

    if args.exp_dir != "":
        opts['exp_dir'] = args.exp_dir


    path =opts['exp_dir']
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))

    os.system('cp %s %s' % (args.exp, opts['exp_dir']))



    if args.generate_data:
        generate_data(opts, args.resume, args.num_sample, vis=args.save_vis, start_step=args.start_step)
    else:
        main(opts)


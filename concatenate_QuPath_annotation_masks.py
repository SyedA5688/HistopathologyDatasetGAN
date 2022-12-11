import os
import glob
import json
import argparse

import numpy as np
from PIL import Image

from utils.data_util import tma_4096_crop_class


def get_segmentation_class_numbers():
    seg_classes_dict = {}
    for idx, annotation_class_obj in enumerate(tma_4096_crop_class):
        seg_classes_dict[annotation_class_obj] = idx
    seg_classes_dict["Cortex"] = -1
    print(len(tma_4096_crop_class), "classes:")
    print(seg_classes_dict)
    return seg_classes_dict


def get_annotation_mask_filenames_for_each_img(image_path, annotation_masks_path):
    class_overlay_order = ["Whitespace", "Cortex", "Glomerulus", "Arteriole", "Artery"]
    img_annot_mask_fnames = {}
    for file in glob.glob(os.path.join(image_path, "*.png")):
        img_filename = file.split("/")[-1]
        fname_list = []
        for class_name in class_overlay_order:
            fname_list += [fname for fname in os.listdir(annotation_masks_path) if img_filename in fname and class_name in fname]
        img_annot_mask_fnames[img_filename] = fname_list

    return img_annot_mask_fnames


def reconstruct_mask_for_one_image(image_path, image_name, annotation_masks_path, img_annotation_mask_fnames, seg_classes):
    img = Image.open(os.path.join(image_path, image_name))
    img_w, img_h = img.size

    mask_file_list = img_annotation_mask_fnames[image_name]
    concatenated_mask = np.full((img_h, img_w), 0)  # initialize with 0s for whitespace
    for annotation_mask_file in mask_file_list:
        mask_class_name = annotation_mask_file.split(".png_")[-1].split("_(")[0]
        if mask_class_name not in seg_classes:
            raise Exception("Found unknown class:", mask_class_name)
        mask_coords_str = annotation_mask_file.split("(1.00,")[-1].split(")-mask")[0]
        mask_coords_str_list = mask_coords_str.split(",")
        mask_coords_list = [int(coord_str) for coord_str in mask_coords_str_list]

        annotation_mask = Image.open(os.path.join(annotation_masks_path, annotation_mask_file))  # shape: (width, height)
        annotation_mask_np = np.array(annotation_mask)  # shape: (height, width)
        assert len(annotation_mask_np.shape) == 2, "Unknown mask shape, not 2D"

        #--- Correct for if starting x or y coordinate went negative. Want it to start at 0 or above ---#
        mask_start_x, mask_start_y = mask_coords_list[0], mask_coords_list[1]
        if mask_start_x < 0:
            annotation_mask_np = annotation_mask_np[:, abs(mask_start_x):]
            mask_start_x = 0

        if mask_start_y < 0:
            annotation_mask_np = annotation_mask_np[abs(mask_start_y):, :]
            mask_start_y = 0

        #--- Correct for if height or width of the annotation mask is larger than the image
        # dimensions because of 300-pixel padding ---#
        mask_h, mask_w = annotation_mask_np.shape
        if mask_start_y + mask_h > img_h:
            overshoot = mask_start_y + mask_h - img_h
            overshoot *= -1
            annotation_mask_np = annotation_mask_np[:overshoot, :]
            mask_h = img_h
        if mask_start_x + mask_w > img_w:
            overshoot = mask_start_x + mask_w - img_w
            overshoot *= -1
            annotation_mask_np = annotation_mask_np[:, :overshoot]
            mask_w = img_w

        #---  ---# # Overlay annotation mask onto image
        annotation_class_idx = seg_classes[mask_class_name]
        mask = annotation_mask_np == 255
        concatenated_mask[mask_start_y:mask_start_y + mask_h, mask_start_x:mask_start_x + mask_w] = np.where(mask, annotation_class_idx, concatenated_mask[mask_start_y:mask_start_y + mask_h, mask_start_x:mask_start_x + mask_w])
        # Note: last annotation overlayed on top has priority if annotation classes overlap on an image.

    #--- Create Cortical Tubulointerstitium mask ---#
    interstitial_mask = concatenated_mask == -1
    concatenated_mask = np.where(interstitial_mask, seg_classes["Cortical Tubulointerstitium"], concatenated_mask)
    return concatenated_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default="/path/to/Histopathology_Dataset_GAN/experiments/TMA_4096_tile.json")

    opts = parser.parse_args()
    args = json.load(open(opts.experiment, 'r'))

    seg_classes = get_segmentation_class_numbers()
    img_annotation_mask_fnames = get_annotation_mask_filenames_for_each_img(args["dataset_save_dir"], args["qupath_annotation_mask_dir"])

    for file in os.listdir(args["dataset_save_dir"]):
        if file.split(".")[-1] == "png":
            print("Processing file", file)
            concat_mask = reconstruct_mask_for_one_image(args["dataset_save_dir"], file, args["qupath_annotation_mask_dir"], img_annotation_mask_fnames, seg_classes)

            np.save(os.path.join(args["dataset_save_dir"], file.split(".png")[0] + "_mask.npy"), concat_mask)

    print("Done.")


if __name__ == "__main__":
    main()

# HistopathologyDatasetGAN

Official implementation of HD_GAN by Syed Rizvi.
This repository updates the DatasetGAN annotation and segmentation process to use StyleGAN2-ADA, 
and makes some improvements on memory consumption.

## Quick Start:
1. Train a StyleGAN2-ADA model using the official NVIDIA repository, save pickle file
2. Use the script create_img_latent_pairs.py to generate images and save the corresponding latents for later use.
3. Annotate the generated images in QuPath.
4. Run QuPath batch processing script on QuPath projects to create compartment masks.
5. Run the script generated_datasets/concatenate_QuPath_annotation_masks.py (point paths to correct directories) in order to stitch compartment masks together into ground truth masks.
6. Create pixel feature dataset that will be used in training by dataloaders, use script process_pixel_dataset.py
7. Run pixel classification training, script is train_pixel_classifier.py

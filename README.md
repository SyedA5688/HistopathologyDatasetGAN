# HistopathologyDatasetGAN

Official PyTorch implementation of the paper [Histopathology DatasetGAN: Synthesizing Large-Resolution Histopathology Datasets](https://arxiv.org/abs/2207.02712).
Improvements are proposed and implemented on the original DatasetGAN framework for annotation-efficient image-annotation generation.

## Quick Start:
To recreate the HDGAN environment, run:
```
conda env create -f environment.yml
```

## HDGAN Framework Steps:
1. Train a StyleGAN2-ADA model using the official NVIDIA repository, saving the pickle file of the best trained checkpoint.
2. Use the script create_img_latent_pairs.py to generate images and save the corresponding latents for later use.
3. Annotate the generated images in QuPath or another annotation software of choice.
4. Create compartment masks of structures of interest using chosen annotation software.
5. [If using QuPath and compartment masks are separated by class] Run the script concatenate_QuPath_annotation_masks.py (point paths to correct directories) in order to stitch compartment masks together into ground truth masks.
6. Create pixel feature dataset that will be used in training by dataloaders with the process_pixel_dataset.py script.
7. Run pixel classification training: ```python train_pixel_classifier.py```


## Citing
```
@article{rizvi2022histopathology,
  title={Histopathology DatasetGAN: Synthesizing Large-Resolution Histopathology Datasets},
  author={Rizvi, SA and Cicalese, P and Seshan, SV and Sciascia, S and Becker, JU and Nguyen, HV},
  journal={arXiv preprint arXiv:2207.02712},
  year={2022}
}
```

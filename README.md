# FACTS

Adversarial Co-Training with Fuzzy Boundary Regularization for Semi-Supervised Medical Image Segmentation

FACTS is a semi-supervised segmentation framework that combines adversarial co-training and a fuzzy boundary loss. The FB loss assigns higher weights to pixels near class boundaries using a Gaussian membership function over the distance-to-boundary map.

This repository contains the ACDC cardiac MRI implementation. The same framework can be adapted to other 2D medical image segmentation datasets, including ISIC and ultrasound data.

## Code

The main training script is:

```bash
code/train_facts.py
```

The UNet architecture is implemented in:

```bash
code/networks/unet.py
```

Important folders:

```text
code/dataloaders/     data loading and sampling utilities
code/networks/        UNet and network factory
code/utils/           losses, metrics, and ramp-up functions
data/ACDC/            expected ACDC data location
example_data/ACDC_tiny/ synthetic example data
```

## Requirements

Install the Python packages using:

```bash
pip install -r requirements.txt
```

Install PyTorch according to your CUDA version from [pytorch.org](https://pytorch.org/). The current training code expects a CUDA-enabled PyTorch installation.

## Data

The processed ACDC data should be arranged as:

```text
data/ACDC/
  train_slices.list
  val.list
  test.list
  data/
    slices/
      <slice_name>.h5
    <case_name>.h5
```

Each `.h5` file should contain:

```text
image
label
```

Here, `image` is the input MRI slice or volume, and `label` is the corresponding segmentation mask with class labels `0, 1, 2, 3`.

Training slices are loaded from `data/ACDC/data/slices/`. Validation and test cases are loaded from `data/ACDC/data/`.

## Demo Data

The folder `example_data/ACDC_tiny/` contains synthetic `.h5` files with the same structure expected by the dataloader:

- 8 training slices
- 2 validation volumes
- 2 testing volumes

These files are only for checking the data format and code execution. They are not real medical images.

## Execution

To train FACTS on processed ACDC data, run:

```bash
cd code
python train_facts.py --root_path ../data/ACDC --model unet --num_classes 4
```

To check the loader and code path using the synthetic demo data, run:

```bash
cd code
python train_facts.py --root_path ../example_data/ACDC_tiny --max_iterations 1 --batch_size 2 --labeled_bs 1 --labeled_num 1
```

## FB Loss

For each class `k`, FACTS computes a one-vs-all boundary distance map:

```text
b_k(p) = N_k(p) d_bg_k(p) + (1 - N_k(p)) d_fg_k(p)
```

The fuzzy boundary membership is:

```text
phi_k(p) = exp(-B_k(p)^2 / (2 sigma^2))
```

Pixels closer to the boundary have larger membership values and therefore receive stronger boundary-focused supervision.

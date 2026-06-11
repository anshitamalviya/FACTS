# FACTS

Official research code release for **FACTS**, a semi-supervised medical image segmentation framework using Cross Pseudo Supervision with a fuzzy boundary loss (**FB loss**).

This public release contains the ACDC cardiac MRI segmentation implementation, including the final training script, UNet architecture, data loading utilities, and a small synthetic example dataset that documents the expected file format.

## Repository Structure

```text
FACTS/
  code/
    train_facts.py              # main training entrypoint
    val_2D.py                   # validation helper
    dataloaders/                # ACDC dataloader and samplers
    networks/                   # UNet and network factory
    utils/                      # losses, metrics, ramps
  data/
    ACDC/                       # expected location for processed ACDC data
  example_data/
    ACDC_tiny/                  # synthetic example data for format checking
  requirements.txt
  README.md
```

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

Install PyTorch using the command recommended for your CUDA version from [pytorch.org](https://pytorch.org/). The current training code uses `.cuda()`, so a CUDA-capable PyTorch environment is expected.

## Dataset Preparation

The full ACDC dataset is not included in this repository. Please download and preprocess ACDC according to its official license and place the processed `.h5` files under:

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

- `image`: image array for a 2D slice or 3D volume.
- `label`: integer segmentation mask with labels `0, 1, 2, 3`.

Training slices are read from `data/ACDC/data/slices/`. Validation and test volumes are read from `data/ACDC/data/`.

## Example Data

`example_data/ACDC_tiny/` contains synthetic `.h5` files with the same keys and folder structure expected by the dataloader:

- 8 synthetic training slices.
- 2 synthetic validation volumes.
- 2 synthetic testing volumes.

This data is provided only for checking the repository structure and loader format. It is not real medical data and must not be used for reporting segmentation performance.

## Training

From the repository root:

```bash
cd code
python train_facts.py --root_path ../data/ACDC --model unet --num_classes 4
```

For a quick format smoke test with the synthetic example data:

```bash
cd code
python train_facts.py --root_path ../example_data/ACDC_tiny --max_iterations 1 --batch_size 2 --labeled_bs 1 --labeled_num 1
```

The smoke test is intended only to verify that the data layout is readable. It is not a meaningful training run.

## Method Summary

FACTS uses a fuzzy boundary loss to emphasize supervision near anatomical boundaries. For each class, the boundary distance is computed using a one-vs-all opposite-region distance:

```text
b_k(p) = N_k(p) d_bg_k(p) + (1 - N_k(p)) d_fg_k(p)
```

The fuzzy boundary membership is then defined as:

```text
phi_k(p) = exp(-B_k(p)^2 / (2 sigma^2))
```

These boundary memberships weight the loss around class interfaces and are used together with cross pseudo supervision for semi-supervised segmentation.

## Other Modalities

The FACTS framework is designed for 2D medical image segmentation and can be adapted to additional modalities. In our broader experiments, the framework was also prepared for ISIC skin lesion segmentation and ultrasound segmentation. This repository currently provides the ACDC release; modality-specific code and dataset instructions can be added in separate folders when released.

## Citation

This repository is being prepared for a manuscript submission to **IEEE Transactions on Medical Imaging (TMI)**. Citation information will be added after the paper is available.

## Acknowledgements

This implementation builds on common semi-supervised segmentation components, including Cross Pseudo Supervision and UNet-style segmentation networks. Please also follow the dataset licenses and citation requirements for ACDC or any other dataset used with this code.

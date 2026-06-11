# Synthetic ACDC Example

This folder contains synthetic `.h5` files that match the data loader format.

It is only for checking the expected structure:

- `train_slices.list` points to files under `data/slices/`.
- `val.list` and `test.list` point to files under `data/`.
- Each `.h5` file contains `image` and `label`.

This is not real ACDC data and should not be used for reporting model performance.

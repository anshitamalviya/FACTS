# ACDC Data

Place the processed ACDC H5 files here before training.

Expected structure:

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

Training slice files are read from `data/slices/<name>.h5`, where `<name>` comes from `train_slices.list`.

Validation files are read from `data/<name>.h5`, where `<name>` comes from `val.list`.

The `.h5` files should contain datasets named `image` and `label`.

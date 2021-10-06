# Feature extraction with [DeepProfiler](https://github.com/cytomining/DeepProfiler)

This folder contains the config files to run DeepProfiler feature extraction
using a pretrained model.

## Install DeepProfiler

Follow the [Installation guide](https://github.com/cytomining/DeepProfiler#quick-guide).

## Download images

Copy the `tiff` images to the `inputs/images/` folder.

## Download locations

Copy the `csv` files containing the cell locations to the `inputs/locations`
folder.

## Download pretrained model

Download the pretrained model and move it to `outputs/efn_pretrained/checkpoint/efficientnetb0_notop.h5`

## Extract single-cell features

Extract features using the `profile` option:

```bash
$ python3 -m deepprofiler --exp efn_pretrained --root ./ --config jump.json profile
```

# Build profiles with single-cell features

Run the aggregation script to build well-based profiles:

```bash
$ python3 utils/build_profiles.py
```

It will generate the `efn_pretrained.parquet` file with the following
structure:


|    | Metadata_Plate   | Metadata_Well   |         0 |          1 |         2 | ...    |      6397 |       6398 |     6399 |
|---:|:-----------------|:----------------|----------:|-----------:|----------:|-------:| ---------:|-----------:|---------:|
|  0 | BR00116996       | A01             | -0.145242 | -0.0976836 | -0.110939 | ...    | -0.188828 | -0.0994691 | 1.11783  |
|  1 | BR00116996       | A02             | -0.16149  | -0.0870829 | -0.130767 | ...    | -0.185635 | -0.11242   | 1.04221  |
|  2 | BR00116996       | A03             | -0.146293 | -0.0802944 | -0.123345 | ...    | -0.180474 | -0.105045  | 1.01936  |
|  3 | BR00116996       | A04             | -0.153452 | -0.0818477 | -0.133035 | ...    | -0.187012 | -0.103294  | 0.992679 |
|  4 | BR00116996       | A05             | -0.167078 | -0.0862786 | -0.134154 | ...    | -0.187428 | -0.109548  | 1.11078  |

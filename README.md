# Cell images
Cell images are available on a S3 bucket (link).

# Plate map and Metadata
Plate map and Metadata are available from the `metadata/' folder and also from https://github.com/jump-cellpainting/JUMP-Target.

# CellProfiler feature extraction
Use the CellProfiler pipelines in `pipelines/2020_11_04_CPJUMP1` and follow the instructions in the [profiling handbook](https://cytomining.github.io/profiling-handbook/) up until chapter 6.2 to generate the well level aggregated profiles from the cell images.

# Profile processing with pycytominer
Clone this repo and activate the conda environment, after installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html), with the commands

```bash
git clone https://github.com/jump-cellpainting/neurips-cpjump1
cd neurips-cpjump1
git submodule update --init --recursive
conda env create --force --file environment.yml
conda activate profiling
```

Then run the pycytominer workflow with the command

```bash
./run.sh
```

This creates the profiles in the `profiles/` folder for all the plates in each batch. The folder for each plate contains the following files

| File name                                                  | Description                                              |
| ---------------------------------------------------------- | -------------------------------------------------------- |
| `<plate_ID>.csv.gz`                                        | Aggregated profiles                                      |
| `<plate_ID>_augmented.csv`                                 | Metadata annotated profiles                              |
| `<plate_ID>_normalized.csv.gz`                             | MAD robustized to whole plate profiles                   |
| `<plate_ID>_normalized_negcon.csv.gz`                      | MAD robustized to negative control profiles              |
| `<plate_ID>_normalized_feature_select_plate.csv.gz`        | Feature selected normalized to whole plate profiles      |
| `<plate_ID>_normalized_feature_select_negcon_plate.csv.gz` | Feature selected normalized to negative control profiles |

# Running the analysis script
To run the analysis script activate the conda environment in `analysis/`

```bash
conda env create --force --file analysis/environment.yml
conda activate analysis
```

Then run the jupyter notebook (`analysis/0.percent_matching.ipynb`) to create the figures in `analysis/figues/`.

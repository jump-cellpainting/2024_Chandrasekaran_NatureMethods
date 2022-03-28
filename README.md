- [Step 1: Download cell images](#step-1--download-cell-images)
  * [Batch and Plate metadata](#batch-and-plate-metadata)
  * [Image metadata](#image-metadata)
  * [Plate map and Perturbation Metadata](#plate-map-and-perturbation-metadata)
- [Step 2: Extract features using CellProfiler and DeepProfiler](#step-2-extract-features-using-cellprofiler-and-deepprofiler)
- [Step 3: Process the profiles using pycytominer](#step-3-process-the-profiles-using-pycytominer)
- [Step 4: Run the analysis script](#step-4-run-the-analysis-script)
- [Data Organization](#data-organization)
- [Maintenance plan](#maintenance-plan)
- [Compute resources](#compute-resources)
- [License](#license)
- [Manuscript](#manuscript)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

Features from the cell images were extracted using [CellProfiler](https://cellprofiler.org/) and the single cell profiles were aggregated, annotated, normalized and feature selected using [pycytominer](https://github.com/cytomining/pycytominer). Features were also extracted using [DeepProfiler](https://github.com/cytomining/DeepProfiler) which were annotated and spherized. The resulting profiles were analyzed using the [notebooks in this repo](https://github.com/jump-cellpainting/neurips-cpjump1/tree/main/benchmark). Steps for reproducing the data in this repository are outlined below.

# Step 1: Download cell images

Cell images are available on a S3 bucket. The images can be downloaded using the command

```bash
batch = <BATCH NAME>
aws s3 cp \
  --no-sign-request \
  --recursive \
  s3://cellpainting-gallery/jump-pilot/source_4/images/${batch}/ . 
```

The `<BATCH NAME>` is one of the six batches [mentioned below](#batch-and-plate-metadata).

You can test out download for a single file using:

```
suffix=images/2020_11_04_CPJUMP1/images/BR00117010__2020-11-08T18_18_00-Measurement1/Images/r01c01f01p01-ch1sk1fk1fl1.tiff

aws s3 cp \
  --no-sign-request \
  s3://cellpainting-gallery/jump-pilot/source_4/${suffix} \
  .
```

See this [wiki](https://github.com/carpenterlab/2016_bray_natprot/wiki/What-do-Cell-Painting-features-mean%3F) for sample Cell Painting images and the meaning of (CellProfiler-derived) Cell Painting features. 

## Batch and Plate metadata
There are six batches of data - `2020_11_04_CPJUMP1`, `2020_11_18_CPJUMP1_TimepointDay1`, `2020_11_19_TimepointDay4`, `2020_12_02_CPJUMP1_2WeeksTimePoint`, `2020_12_07_CPJUMP1_4WeeksTimePoint` and `2020_12_08_CPJUMP1_Bleaching`.

[experimental-metadata.tsv](benchmark/output/experiment-metadata.tsv) contains all the experimental metadata for each plate in each batch.

## Image metadata

The folder for each 384-well plate typically contains images from nine sites for each well (for some wells 7,8 or 16 sites were imaged). 
The (x,y) coordinates of sites are available in the `Metadata_PositionX` and `Metadata_PositionY` columns of the `load_data.csv.gz` files in the `load_data_csv` folder. 
There are eight images per site (five from the fluorescent channels and three brightfield images). 
The names of the image files follow the naming convention - `rXXcXXfXXp01-chXXsk1fk1fl1.tiff` where
- `rXX` is the row number of the well that was imaged. `rXX` ranges from `r01` to `r16`.
- `cXX` is the column number of the well that was imaged. `cXX` ranges from `c01` to `c24`.
- `fXX` corresponds to the site that was imaged. `fXX` ranges from `f01` to `f16`.
- `chXX` corresponds to the fluorescent channels imaged. `chXX` ranges from `ch01` to `ch08`.
    - `ch01` - Alexa 647
    - `ch02` - Alexa 568
    - `ch03` - Alexa 488 long
    - `ch04` - Alexa 488
    - `ch05` - Hoechst 33342
    - `ch06-8` - three brighfield z planes.

Cell bounding boxes and segmentation masks have not been provided. 

## Plate map and Perturbation Metadata
Plate map and Metadata are available in the `metadata/` folder and also from https://github.com/jump-cellpainting/JUMP-Target.

# Step 2: Extract features using CellProfiler and DeepProfiler
Use the CellProfiler pipelines in `pipelines/2020_11_04_CPJUMP1` and follow the instructions in the [profiling handbook](https://cytomining.github.io/profiling-handbook/) up until chapter 5.3 to generate the well-level aggregated CellProfiler profiles from the cell images. 

Instead of regenerating the single cell CellProfiler features, they can also be downloaded from the S3 bucket

```bash
batch = <BATCH NAME>
aws s3 cp \
  --no-sign-request \
  --recursive \
  s3://cellpainting-gallery/jump-pilot/source_4/workspace/backend/${batch}/ . 
```

where `<BATCH NAME>` is one of the six batches [mentioned above](#batch-and-plate-metadata).

Follow the [README.md](deep_profiles/README.md) to extract features from a
pretrained neural network using [DeepProfiler](https://github.com/cytomining/DeepProfiler)

# Step 3: Process the profiles using pycytominer
Pycytominer adds metadata from `metadata/moa` to the well-level aggregated profiles, normalizes the profiles to the whole plate and to the negative controls, separately and filters out invariant and redundant features.

To reproduce the profiles, clone this repo, download the files and activate the conda environment, after installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html), with the commands

```bash
git clone https://github.com/jump-cellpainting/neurips-cpjump1
cd neurips-cpjump1
git lfs pull
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

# Step 4: Run the benchmark script
The benchmark scripts compute `Percent Replicating` which is a measure of signature strength, `Percent Matching across modalities` which is a measure of how well the chemical and genetic perturbation profiles match. These metrics are calculated using the `Feature selected normalized to negative control profiles` (well-level profiles).

In the case of features extracted using DeepProfiler, the annotated features are spherized and `Percent Replicating` is calculated on these profiles.

To run the benchmark script activate the conda environment in `benchmark/`

```bash
conda env create --force --file benchmark/environment.yml
conda activate benchmark
```

Then run the jupyter notebooks (`benchmark/0.percent_matching.ipynb` and `benchmark/1.percent_matching_across_modalities.ipynb`) to create the figures in `benchmark/figues/` and the tables in `benchmark/README.md`.

# Data Organization
The following is the description of contents of the relevant folders in this repo.

- `benchmark` - contains the notebooks for reproducing the benchmark scores and figures
- `config_files` - contains the config files required for processing the profiles with pycytominer
- `datasplit` - contains the recommended data splits
- `deep_profiles` - contains the config files, functions and instructions for extracting DeepProfiler features
- `example_images` - contains single-site, all channel images from ten example wells
- `load_data_csv` - contains file location and other image metadata for each plate in all batches
- `metadata` - contains the perturbation metadata and plate maps
- `pipelines` - contains the CellProfiler pipelines for cell segmentation and feature extraction
- `profiles` - contains both CellProfiler and DeepProfiler extracted features for all batches
- `profiling-recipe` - contains the scripts that for running the pycytominer pipeline for processing profiles
- `visualization` - contains notebooks for generating plate map and clinical phase status visualization figures
- `environment.yml` - conda environment for running pycytominer pipeline
- `run.sh` - runs the pycytominer pipeline for processing profiles
- `maintenance_plan.md` - contains our maintenance plan for this dataset

# Maintenance plan
We have provided our maintenance plan in [maintenance_plan.md](maintenance_plan.md).

# Compute resources
For segmentation and feature extraction by CellProfiler, each plate of images took on average 30 minutes to process, using a fleet of 200 m4.xlarge spot instances (800 vCPUs), which cost approximately $10 per plate.  Aggregation into mean profiles takes 12-18 hours, though can be parallelized onto a single large machine, at the total cost of <$1 per plate. For profile processing with pycytominer, each plate took under two minutes, using a local machine (Intel Core i9 with 16 GB memory)

DeepProfiler took around 8 hours to extract features from ~280.000 images in a
p3.2xlarge with a single Tesla V100-SXM2 GPU. Note that cell locations were
previously precomputed with the CellProfiler segmentation pipeline.

# License
We use a dual license in this repository.
We license the source code as [BSD 3-Clause](LICENSE_BSD3.md), and license the data, results, and figures as [CC0 1.0](LICENSE_CC0.md).

# Manuscript
A manuscript describing the contents of this repository is on [biorxiv](https://www.biorxiv.org/content/10.1101/2022.01.05.475090v1).

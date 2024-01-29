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

Image-based features from the cell images were extracted using [CellProfiler](https://cellprofiler.org/) and assembled as single cell profiles, which were aggregated, annotated, normalized and feature selected using [pycytominer](https://github.com/cytomining/pycytominer). Image-based features were also extracted using [DeepProfiler](https://github.com/cytomining/DeepProfiler) which were annotated and [spherized](https://en.wikipedia.org/wiki/Whitening_transformation). The resulting profiles were analyzed using the [notebooks in this repo](https://github.com/jump-cellpainting/2023_Chandrasekaran_submitted/tree/main/benchmark). Steps for reproducing the data in this repository are outlined below.

# Step 1: Download cell images

Cell images are available on a S3 bucket. The images can be downloaded using the command

```bash
batch=<BATCH NAME>
aws s3 sync \
  --no-sign-request \
  s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/images/${batch}/ . 
```

The `${batch}` is one of the six batches [mentioned below](#batch-and-plate-metadata).

You can test out download for a single file using:

```
suffix=images/2020_11_04_CPJUMP1/images/BR00117010__2020-11-08T18_18_00-Measurement1/Images/r01c01f01p01-ch1sk1fk1fl1.tiff

aws s3 cp \
  --no-sign-request \
  s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/${suffix} \
  .
```

Note: If you'd like to just browse the data, it's a lot easier [to do so using a storage browser](https://stackoverflow.com/a/72143198/1094109). 

The following are various kinds of image-related or experiment-related metadata.

## Batch and Plate metadata
There are six batches of data - `2020_11_04_CPJUMP1`, `2020_11_18_CPJUMP1_TimepointDay1`, `2020_11_19_TimepointDay4`, `2020_12_02_CPJUMP1_2WeeksTimePoint`, `2020_12_07_CPJUMP1_4WeeksTimePoint` and `2020_12_08_CPJUMP1_Bleaching`. Each batch either contains a single experiment of multiple experiments. Details about all the experimental conditions ia available in the [associated manuscript](#manuscript).

[experimental-metadata.tsv](benchmark/output/experiment-metadata.tsv) contains all the experimental metadata for each plate in each batch. The following is the description of each of the columns in the file

- `Batch` - name of the batch the plate belongs to. Could be one of the six batches.
- `Plate_Map_Name` - name of the plate layout file associated with the plate. All plate layouts are in `metadata/platemaps`.
- `Assay_Plate_Barcode` - name of the plate.
- `Perturbation` - type of perturbation on the plate. Could be one of `compound`, `crispr` or `orf`.
- `Cell_type` - type of cell. Could be one of `A549` or `U2OS`.
- `Time` -  amount of time the cells were subjected to a pertubation.
- `Density` - cell seeding density. 100 is the baseline cell sending density percentage while 80 and 120 are 20% lower and higher cell seeding density, respectively.
- `Antibiotics` - status of antibiotics added to the cells. Could be one of `Blasticidin`, `Puromycin` or `absent` (no antibiotics added).
- `Cell_line` - whether or not the cell contain Cas9. Some compound experiments were performed using cells containing Cas9.
- `Time_delay` - how long after staining the plates were stored, before they were imaged.
- `Times_imaged` - number of times a plate or a subset of wells on the plate was imaged.
- `Anomaly` - whether the plate contained experimental anomalies. Most anomalies are related to differences in the amount of dye added to the plate. Though these may impact the profiles, we don't expect a significant effect.
- `Number_of_images` - number of total images for each plate. The number may be higher than the median because additional fields of view were captured, or lower because only a subset of wells on the plate were imaged.

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
After downloading the images, use the CellProfiler pipelines in `pipelines/2020_11_04_CPJUMP1` and follow the instructions in the [profiling handbook](https://cytomining.github.io/profiling-handbook/) up until chapter 5.3 to generate the well-level aggregated CellProfiler profiles. 

Instead of regenerating the CellProfiler features, they can also be downloaded from the S3 bucket

```bash
batch = <BATCH NAME>
aws s3 cp \
  --no-sign-request \
  --recursive \
  s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/workspace/backend/${batch}/ . 
```

where `${batch}` is one of the six batches [mentioned above](#batch-and-plate-metadata).

The `.sqlite` files contain single-cell image-based profiles while the `.csv` files contain the well-level aggregated profiles.

See this [blog post](https://carpenter-singh-lab.broadinstitute.org/blog/help-interpreting-image-based-profiles) for the meaning of (CellProfiler-derived) Cell Painting features. Samples Cell Painting images can be found in the `example_images` folder.

To extract features using a pretrained neural network using [DeepProfiler](https://github.com/cytomining/DeepProfiler), follow the [README.md](deep_profiles/README.md) instructions, which creates well-level profiles.

# Step 3: Process the profiles using pycytominer
After generating the well-level CellProfiler-based features, use Pycytominer to add metadata from `metadata/moa`, normalize the profiles to the whole plate and to the negative controls, separately, and the filter out invariant and redundant features.

To regenerate all the profiles, clone this repo, download the files and activate the conda environment. Before issuing the following commands, Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

```bash
git clone https://github.com/jump-cellpainting/2023_Chandrasekaran_submitted
cd 2023_Chandrasekaran_submitted
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

Annotated DeepProfiler profiles are spherized using [this notebook](https://github.com/jump-cellpainting/2023_Chandrasekaran_submitted/blob/main/benchmark/old_notebooks/3.spherize_profiles.ipynb). 

# Step 4: Run the benchmark script
The benchmark scripts compute `Average Precision (AP)` for various retrieval tasks, such as, retrieving replicates against negative controls, retrieving perturbation pairs against non-pairs, and retrieving gene-compound pairs against non-pairs. `AP` was calculated using the `Feature selected normalized to negative control profiles` (well-level profiles).

To run the benchmark script activate the conda environment in `benchmark/`

```bash
conda env create --force --file benchmark/environment.yml
conda activate analysis
```

Then run the jupyter notebooks (`benchmark/1.calculate-map-cp.ipynb`, `benchmark/2.calculate-map-dp.ipynb`, and `benchmark/3.generate-map-figure.ipynb`) to create the figures in `benchmark/figues/`.

# Data Organization
The following is the description of relevant files and contents of the relevant folders in this repo.

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

DeepProfiler took around 8 hours to extract features from ~280.000 images in a p3.2xlarge with a single Tesla V100-SXM2 GPU. Note that cell locations were previously precomputed with the CellProfiler segmentation pipeline.

Running the benchmark notebooks took an hour in a local machine (Intel Core i9 with 16 GB memory).

# License
We use a dual license in this repository.
We license the source code as [BSD 3-Clause](LICENSE_BSD3.md), and license the data, results, and figures as [CC0 1.0](LICENSE_CC0.md).

# Manuscript
A manuscript describing the contents of this repository is on [biorxiv](https://www.biorxiv.org/content/10.1101/2022.01.05.475090v1).

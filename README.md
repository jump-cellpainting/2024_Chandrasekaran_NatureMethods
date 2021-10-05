- [Step 1: Download cell images](#step-1--download-cell-images)
  * [Batch and Plate metadata](#batch-and-plate-metadata)
    + [2020_11_04_CPJUMP1](#2020_11_04_cpjump1)
    + [2020_11_18_CPJUMP1_TimepointDay1](#2020_11_18_cpjump1_timepointday1)
    + [2020_11_19_TimepointDay4](#2020_11_19_timepointday4)
    + [2020_12_02_CPJUMP1_2WeeksTimePoint](#2020_12_02_cpjump1_2weekstimepoint)
    + [2020_12_07_CPJUMP1_4WeeksTimePoint](#2020_12_07_cpjump1_4weekstimepoint)
    + [2020_12_08_CPJUMP1_Bleaching](#2020_12_08_cpjump1_bleaching)
  * [Image metadata](#image-metadata)
  * [Plate map and Perturbation Metadata](#plate-map-and-perturbation-metadata)
- [Step 2: Extract features using CellProfiler and DeepProfiler](#step-2-extract-features-using-cellprofiler-and-deepprofiler)
- [Step 3: Process the profiles using pycytominer](#step-3-process-the-profiles-using-pycytominer)
- [Step 4: Run the analysis script](#step-4-run-the-analysis-script)
- [Data Organization](#data-organization)
- [Compute resources](#compute-resources)
- [License](#license)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

Features from the cell images were extracted using [CellProfiler](https://cellprofiler.org/) and the single cell profiles were aggregated, annotated, normalized and feature selected using [pycytominer](https://github.com/cytomining/pycytominer). Features were also extracted using [DeepProfiler](https://github.com/cytomining/DeepProfiler) which were annotated and spherized. The resulting profiles were analyzed using the [notebooks in this repo](https://github.com/jump-cellpainting/neurips-cpjump1/tree/main/benchmark). Steps for reproducing the data in this repository are outlined below.

# Step 1: Download cell images

Cell images are available on a S3 bucket. The images can be downloaded using the command

```bash
aws s3 cp \
  --recursive \
  s3://cell-painting-gallery/jump-pilots/source_4/images . 
```

(We are in the process of moving the images to the `cellpainting-gallery` bucket. The images will be available for download shortly)

See this [wiki](https://github.com/carpenterlab/2016_bray_natprot/wiki/What-do-Cell-Painting-features-mean%3F) for sample Cell Painting images and the meaning of (CellProfiler-derived) Cell Painting features. 

## Batch and Plate metadata
There are six batches of data - `2020_11_04_CPJUMP1`, `2020_11_18_CPJUMP1_TimepointDay1`, `2020_11_19_TimepointDay4`, `2020_12_02_CPJUMP1_2WeeksTimePoint`, `2020_12_07_CPJUMP1_4WeeksTimePoint` and `2020_12_08_CPJUMP1_Bleaching`.

### 2020_11_04_CPJUMP1

51 384-well plates of cells from two cell lines treated with different types of perturbations at different time points.

<details>
<summary>Click to expand</summary>

| Barcode    | Description                                     | Number_of_images |
| ---------- | ----------------------------------------------- | ---------------- |
| BR00118049 | A549 96-hour ORF w/ Blasticidin Plate 1         | 27648            |
| BR00118050 | A549 96-hour ORF Plate 1                        | 27648            | 
| BR00117006 | A549 96-hour ORF Plate 2                        | 27648            |
| BR00118039 | U2OS 96-hour ORF Plate 1                        | 27648            |
| BR00118040 | U2OS 96-hour ORF Plate 2                        | 27648            |
| BR00117020 | A549 48-hour ORF Plate 1                        | 27648            |
| BR00117021 | A549 48-hour ORF Plate 2                        | 27648            |
| BR00117022 | U2OS 48-hour ORF Plate 1                        | 27648            |
| BR00117023 | U2OS 48-hour ORF Plate 2                        | 27648            |
| BR00118041 | A549 96-hour CRISPR Plate 1                     | 27560            |
| BR00118042 | A549 96-hour CRISPR Plate 2                     | 27648            |
| BR00118043 | A549 96-hour CRISPR Plate 3                     | 27648            |
| BR00118044 | A549 96-hour CRISPR Plate 4                     | 27648            |
| BR00118045 | U2OS 96-hour CRISPR Plate 1                     | 27648            |
| BR00118046 | U2OS 96-hour CRISPR Plate 2                     | 27640            |
| BR00118047 | U2OS 96-hour CRISPR Plate 3                     | 27648            |
| BR00118048 | U2OS 96-hour CRISPR Plate 4                     | 27648            |
| BR00117003 | A549 144-hour CRISPR Plate 1                    | 27632            |
| BR00117004 | A549 144-hour CRISPR Plate 2                    | 27648            |
| BR00117005 | A549 144-hour CRISPR Plate 3                    | 27568            |
| BR00117000 | A549 144-hour CRISPR Plate 4                    | 27640            |
| BR00117002 | A549 144-hour CRISPR w/ Puromycin Plate 1       | 27648            |
| BR00117001 | A549 144-hour CRISPR w/ Puromycin Plate 2       | 27648            |
| BR00116997 | U2OS 144-hour CRISPR Plate 1                    | 27648            |
| BR00116998 | U2OS 144-hour CRISPR Plate 2                    | 27648            |
| BR00116999 | U2OS 144-hour CRISPR Plate 3                    | 27648            |
| BR00116996 | U2OS 144-hour CRISPR Plate 4                    | 27648            |
| BR00116991 | A549 24-hour Compound Plate 1                   | 27648            |
| BR00116992 | A549 24-hour Compound Plate 2                   | 27640            |
| BR00116993 | A549 24-hour Compound Plate 3                   | 27352            |
| BR00116994 | A549 24-hour Compound Plate 4                   | 27576            |
| BR00116995 | U2OS 24-hour Compound Plate 1                   | 27648            |
| BR00117024 | U2OS 24-hour Compound Plate 2                   | 27648            |
| BR00117025 | U2OS 24-hour Compound Plate 3                   | 27648            |
| BR00117026 | U2OS 24-hour Compound Plate 4                   | 27648            |
| BR00117017 | A549 48-hour Compound Plate 1                   | 49144            |
| BR00117019 | A549 48-hour Compound Plate 2                   | 49152            |
| BR00117015 | A549 48-hour Compound Plate 3                   | 49144            |
| BR00117016 | A549 48-hour Compound Plate 4                   | 49152            |
| BR00117012 | U2OS 48-hour Compound Plate 1                   | 27648            |
| BR00117013 | U2OS 48-hour Compound Plate 2                   | 27648            |
| BR00117010 | U2OS 48-hour Compound Plate 3                   | 27648            |
| BR00117011 | U2OS 48-hour Compound Plate 4                   | 27648            |
| BR00117054 | A549 48-hour +20% Seed Density Compound Plate 1 | 27648            |
| BR00117055 | A549 48-hour +20% Seed Density Compound Plate 2 | 27648            |
| BR00117008 | A549 48-hour -20% Seed Density Compound Plate 1 | 27648            |
| BR00117009 | A549 48-hour -20% Seed Density Compound Plate 2 | 27648            |
| BR00117052 | A549 Cas9 48-hour Compound Plate 1              | 27648            |
| BR00117053 | A549 Cas9 48-hour Compound Plate 2              | 27648            |
| BR00117050 | A549 Cas9 48-hour Compound Plate 3              | 27648            |
| BR00117051 | A549 Cas9 48-hour Compound Plate 4              | 27648            |
</details>

### 2020_11_18_CPJUMP1_TimepointDay1

Eight ORF plates imaged one day after staining.

<details>
<summary>Click to expand</summary>

| Barcode    | Description              | Number_of_images |
| ---------- | ------------------------ | ---------------- |
| BR00118050 | A549 96-hour ORF Plate 1 | 12280            |
| BR00117006 | A549 96-hour ORF Plate 2 | 12272            |
| BR00118039 | U2OS 96-hour ORF Plate 1 | 12288            |
| BR00118040 | U2OS 96-hour ORF Plate 2 | 12280            | 
| BR00117020 | A549 48-hour ORF Plate 1 | 12288            |
| BR00117021 | A549 48-hour ORF Plate 2 | 12288            |
| BR00117022 | U2OS 48-hour ORF Plate 1 | 12288            |
| BR00117023 | U2OS 48-hour ORF Plate 2 | 12288            |

</details>

### 2020_11_19_TimepointDay4

Eight ORF plates imaged four days after staining.

<details>
<summary>Click to expand</summary>

| Barcode    | Description              | Number_of_images |
| ---------- | ------------------------ | ---------------- |
| BR00118050 | A549 96-hour ORF Plate 1 | 12288            | 
| BR00117006 | A549 96-hour ORF Plate 2 | 12288            |
| BR00118039 | U2OS 96-hour ORF Plate 1 | 12288            |
| BR00118040 | U2OS 96-hour ORF Plate 2 | 12288            |
| BR00117020 | A549 48-hour ORF Plate 1 | 12288            |
| BR00117021 | A549 48-hour ORF Plate 2 | 12288            |
| BR00117022 | U2OS 48-hour ORF Plate 1 | 12208            |
| BR00117023 | U2OS 48-hour ORF Plate 2 | 12280            |

</details>

### 2020_12_02_CPJUMP1_2WeeksTimePoint

Eight ORF plates imaged 14 days after staining.

<details>
<summary>Click to expand</summary>

| Barcode    | Description              | Number_of_images |
| ---------- | ------------------------ | ---------------- |
| BR00118050 | A549 96-hour ORF Plate 1 | 27648            | 
| BR00117006 | A549 96-hour ORF Plate 2 | 27648            |
| BR00118039 | U2OS 96-hour ORF Plate 1 | 27648            |
| BR00118040 | U2OS 96-hour ORF Plate 2 | 27648            |
| BR00117020 | A549 48-hour ORF Plate 1 | 27320            |
| BR00117021 | A549 48-hour ORF Plate 2 | 27632            |
| BR00117022 | U2OS 48-hour ORF Plate 1 | 27648            |
| BR00117023 | U2OS 48-hour ORF Plate 2 | 27648            |

</details>

### 2020_12_07_CPJUMP1_4WeeksTimePoint

Eight ORF plates imaged 28 days after staining.

<details>
<summary>Click to expand</summary>

| Barcode    | Description              | Number_of_images |
| ---------- | ------------------------ | ---------------- |
| BR00118050 | A549 96-hour ORF Plate 1 | 27648            | 
| BR00117006 | A549 96-hour ORF Plate 2 | 27648            |
| BR00118039 | U2OS 96-hour ORF Plate 1 | 27648            |
| BR00118040 | U2OS 96-hour ORF Plate 2 | 27648            |
| BR00117020 | A549 48-hour ORF Plate 1 | 27648            |
| BR00117021 | A549 48-hour ORF Plate 2 | 27640            |
| BR00117022 | U2OS 48-hour ORF Plate 1 | 27648            |
| BR00117023 | U2OS 48-hour ORF Plate 2 | 27648            |

</details>

### 2020_12_08_CPJUMP1_Bleaching

Four compound plates imaged and additional six times (`A`, `B`, `C`, `D`, `E` and `F`).

<details>
<summary>Click to expand</summary>

| Barcode     | Description                   | Number_of_images |
| ----------- | ----------------------------- | ---------------- |
| BR00116991A | A549 24-hour Compound Plate 1 | 21504            |
| BR00116992A | A549 24-hour Compound Plate 2 | 21504            |
| BR00116993A | A549 24-hour Compound Plate 3 | 27648            |
| BR00116994A | A549 24-hour Compound Plate 4 | 27648            |
| BR00116991B | A549 24-hour Compound Plate 1 | 21504            |
| BR00116992B | A549 24-hour Compound Plate 2 | 21504            |
| BR00116993B | A549 24-hour Compound Plate 3 | 27648            |
| BR00116994B | A549 24-hour Compound Plate 4 | 27648            |
| BR00116991C | A549 24-hour Compound Plate 1 | 21504            |
| BR00116992C | A549 24-hour Compound Plate 2 | 21504            |
| BR00116993C | A549 24-hour Compound Plate 3 | 27648            |
| BR00116994C | A549 24-hour Compound Plate 4 | 27648            |
| BR00116991D | A549 24-hour Compound Plate 1 | 21504            |
| BR00116992D | A549 24-hour Compound Plate 2 | 21504            |
| BR00116993D | A549 24-hour Compound Plate 3 | 27648            |
| BR00116994D | A549 24-hour Compound Plate 4 | 27648            |
| BR00116991E | A549 24-hour Compound Plate 1 | 21504            |
| BR00116992E | A549 24-hour Compound Plate 2 | 21504            |
| BR00116993E | A549 24-hour Compound Plate 3 | 27648            |
| BR00116994E | A549 24-hour Compound Plate 4 | 27648            |
| BR00116991F | A549 24-hour Compound Plate 1 | 21504            |
| BR00116992F | A549 24-hour Compound Plate 2 | 21504            |
| BR00116993F | A549 24-hour Compound Plate 3 | 27648            |
| BR00116994F | A549 24-hour Compound Plate 4 | 27648            | 

</details>

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

**DeepProfiler instructions will go here**

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
- `example_images` - contains single-site, all channel images from ten example wells
- `load_data_csv` - contains file location and other image metadata for each plate in all batches
- `metadata` - contains the perturbation metadata and plate maps
- `pipelines` - contains the CellProfiler pipelines for cell segmentation and feature extraction
- `profiling-recipe` - contains the scripts that for running the pycytominer pipeline for processing profiles
- `visualization` - contains notebooks for generating plate map and clinical phase status visualization figures
- `environment.yml` - conda environment for running pycytominer pipeline
- `run.sh` - runs the pycytominer pipeline for processing profiles

# Datasheet
We have also provided a [DataSheet](DATASHEET.md) with additional details about this dataset.

# Compute resources
For segmentation and feature extraction, each plate of images took on average 30 minutes to process, using a fleet of 200 m4.xlarge spot instances (800 vCPUs), which cost approximately $10 per plate.  Aggregation into mean profiles takes 12-18 hours, though can be parallelized onto a single large machine, at the total cost of <$1 per plate. For profile processing with pycytominer, each plate took under two minutes, using a local machine (Intel Core i9 with 16 GB memory)

**DeepProfiler statistics will go here**

# License
We use a dual license in this repository.
We license the source code as [BSD 3-Clause](LICENSE_BSD3.md), and license the data, results, and figures as [CC0 1.0](LICENSE_CC0.md).

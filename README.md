# Cell images
Cell images are available on a S3 bucket. The images can be downloaded using the command

```bash
aws s3 cp \
  --recursive \
  s3://jump-cellpainting-public/projects/2019_07_11_JUMP-CP-pilots/ . --request-payer 
```

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

# CellProfiler pipeline
Pipelines for cell segmentation and profile extraction is available is the folder `pipelines`.

# Plate map and Metadata
Plate map and Metadata are available from the `metadata/' folder and also from https://github.com/jump-cellpainting/JUMP-Target.

# Batch and plate metadata 
There are six batches of data - `2020_11_04_CPJUMP1`, `2020_11_18_CPJUMP1_TimepointDay1`, `2020_11_19_TimepointDay4`, `2020_12_02_CPJUMP1_2WeeksTimePoint`, `2020_12_07_CPJUMP1_4WeeksTimePoint` and `2020_12_08_CPJUMP1_Bleaching`.

## 2020_11_04_CPJUMP1

51 384-well plates of cells from two cell lines treated with different types of perturbations at different time points.

<details>
<summary>Click to expand</summary>

| Barcode    | Description                                     |
| ---------- | ----------------------------------------------- |
| BR00118049 | A549 96-hour ORF w/ Blasticidin Plate 1         |
| BR00118050 | A549 96-hour ORF Plate 1                        |
| BR00117006 | A549 96-hour ORF Plate 2                        |
| BR00118039 | U2OS 96-hour ORF Plate 1                        |
| BR00118040 | U2OS 96-hour ORF Plate 2                        |
| BR00117020 | A549 48-hour ORF Plate 1                        |
| BR00117021 | A549 48-hour ORF Plate 2                        |
| BR00117022 | U2OS 48-hour ORF Plate 1                        |
| BR00117023 | U2OS 48-hour ORF Plate 2                        |
| BR00118041 | A549 96-hour CRISPR Plate 1                     |
| BR00118042 | A549 96-hour CRISPR Plate 2                     |
| BR00118043 | A549 96-hour CRISPR Plate 3                     |
| BR00118044 | A549 96-hour CRISPR Plate 4                     |
| BR00118045 | U2OS 96-hour CRISPR Plate 1                     |
| BR00118046 | U2OS 96-hour CRISPR Plate 2                     |
| BR00118047 | U2OS 96-hour CRISPR Plate 3                     |
| BR00118048 | U2OS 96-hour CRISPR Plate 4                     |
| BR00117003 | A549 144-hour CRISPR Plate 1                    |
| BR00117004 | A549 144-hour CRISPR Plate 2                    |
| BR00117005 | A549 144-hour CRISPR Plate 3                    |
| BR00117000 | A549 144-hour CRISPR Plate 4                    |
| BR00117002 | A549 144-hour CRISPR w/ Puromycin Plate 1       |
| BR00117001 | A549 144-hour CRISPR w/ Puromycin Plate 2       |
| BR00116997 | U2OS 144-hour CRISPR Plate 1                    |
| BR00116998 | U2OS 144-hour CRISPR Plate 2                    |
| BR00116999 | U2OS 144-hour CRISPR Plate 3                    |
| BR00116996 | U2OS 144-hour CRISPR Plate 4                    |
| BR00116991 | A549 24-hour Compound Plate 1                   |
| BR00116992 | A549 24-hour Compound Plate 2                   |
| BR00116993 | A549 24-hour Compound Plate 3                   |
| BR00116994 | A549 24-hour Compound Plate 4                   |
| BR00116995 | U2OS 24-hour Compound Plate 1                   |
| BR00117024 | U2OS 24-hour Compound Plate 2                   |
| BR00117025 | U2OS 24-hour Compound Plate 3                   |
| BR00117026 | U2OS 24-hour Compound Plate 4                   |
| BR00117017 | A549 48-hour Compound Plate 1                   |
| BR00117019 | A549 48-hour Compound Plate 2                   |
| BR00117015 | A549 48-hour Compound Plate 3                   |
| BR00117016 | A549 48-hour Compound Plate 4                   |
| BR00117012 | U2OS 48-hour Compound Plate 1                   |
| BR00117013 | U2OS 48-hour Compound Plate 2                   |
| BR00117010 | U2OS 48-hour Compound Plate 3                   |
| BR00117011 | U2OS 48-hour Compound Plate 4                   |
| BR00117054 | A549 48-hour +20% Seed Density Compound Plate 1 |
| BR00117055 | A549 48-hour +20% Seed Density Compound Plate 2 |
| BR00117008 | A549 48-hour -20% Seed Density Compound Plate 1 |
| BR00117009 | A549 48-hour -20% Seed Density Compound Plate 2 |
| BR00117052 | A549 Cas9 48-hour Compound Plate 1              |
| BR00117053 | A549 Cas9 48-hour Compound Plate 2              |
| BR00117050 | A549 Cas9 48-hour Compound Plate 3              |
| BR00117051 | A549 Cas9 48-hour Compound Plate 4              |

</details>

## 2020_11_18_CPJUMP1_TimepointDay1

Eight ORF plates imaged one day after staining.

<details>
<summary>Click to expand</summary>

| Barcode    | Description                                     |
| ---------- | ----------------------------------------------- |
| BR00118050 | A549 96-hour ORF Plate 1                        |
| BR00117006 | A549 96-hour ORF Plate 2                        |
| BR00118039 | U2OS 96-hour ORF Plate 1                        |
| BR00118040 | U2OS 96-hour ORF Plate 2                        |
| BR00117020 | A549 48-hour ORF Plate 1                        |
| BR00117021 | A549 48-hour ORF Plate 2                        |
| BR00117022 | U2OS 48-hour ORF Plate 1                        |
| BR00117023 | U2OS 48-hour ORF Plate 2                        |

</details>

## 2020_11_19_TimepointDay4

Eight ORF plates imaged four days after staining.

<details>
<summary>Click to expand</summary>

| Barcode    | Description                                     |
| ---------- | ----------------------------------------------- |
| BR00118050 | A549 96-hour ORF Plate 1                        |
| BR00117006 | A549 96-hour ORF Plate 2                        |
| BR00118039 | U2OS 96-hour ORF Plate 1                        |
| BR00118040 | U2OS 96-hour ORF Plate 2                        |
| BR00117020 | A549 48-hour ORF Plate 1                        |
| BR00117021 | A549 48-hour ORF Plate 2                        |
| BR00117022 | U2OS 48-hour ORF Plate 1                        |
| BR00117023 | U2OS 48-hour ORF Plate 2                        |

</details>

## 2020_12_02_CPJUMP1_2WeeksTimePoint

Eight ORF plates imaged 14 days after staining.

<details>
<summary>Click to expand</summary>

| Barcode    | Description                                     |
| ---------- | ----------------------------------------------- |
| BR00118050 | A549 96-hour ORF Plate 1                        |
| BR00117006 | A549 96-hour ORF Plate 2                        |
| BR00118039 | U2OS 96-hour ORF Plate 1                        |
| BR00118040 | U2OS 96-hour ORF Plate 2                        |
| BR00117020 | A549 48-hour ORF Plate 1                        |
| BR00117021 | A549 48-hour ORF Plate 2                        |
| BR00117022 | U2OS 48-hour ORF Plate 1                        |
| BR00117023 | U2OS 48-hour ORF Plate 2                        |

</details>

## 2020_12_07_CPJUMP1_4WeeksTimePoint

Eight ORF plates imaged 28 days after staining.

<details>
<summary>Click to expand</summary>

| Barcode    | Description                                     |
| ---------- | ----------------------------------------------- |
| BR00118050 | A549 96-hour ORF Plate 1                        |
| BR00117006 | A549 96-hour ORF Plate 2                        |
| BR00118039 | U2OS 96-hour ORF Plate 1                        |
| BR00118040 | U2OS 96-hour ORF Plate 2                        |
| BR00117020 | A549 48-hour ORF Plate 1                        |
| BR00117021 | A549 48-hour ORF Plate 2                        |
| BR00117022 | U2OS 48-hour ORF Plate 1                        |
| BR00117023 | U2OS 48-hour ORF Plate 2                        |

</details>

## 2020_12_08_CPJUMP1_Bleaching

Four compound plates imaged and additional six times (`A`, `B`, `C`, `D`, `E` and `F`).

<details>
<summary>Click to expand</summary>

| Barcode    | Description                                     |
| ---------- | ----------------------------------------------- |
| BR00116991 | A549 24-hour Compound Plate 1                   |
| BR00116992 | A549 24-hour Compound Plate 2                   |
| BR00116993 | A549 24-hour Compound Plate 3                   |
| BR00116994 | A549 24-hour Compound Plate 4                   |

</details>

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

# License

We use a dual license in this repository.
We license the source code as [BSD 3-Clause](LICENSE_BSD3.md), and license the data, results, and figures as [CC0 1.0](LICENSE_CC0.md).

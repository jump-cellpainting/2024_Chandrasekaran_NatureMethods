# Percent Replicating

| Description      | Perturbation   | time   | Cell   |   Percent_Replicating |
|:-----------------|:---------------|:-------|:-------|----------------------:|
| compound_A549_24 | compound       | short  | A549   |                  87.6 |
| compound_A549_48 | compound       | long   | A549   |                  95.1 |
| compound_U2OS_24 | compound       | short  | U2OS   |                  80.1 |
| compound_U2OS_48 | compound       | long   | U2OS   |                  74.5 |
| crispr_U2OS_144  | crispr         | long   | U2OS   |                  62.3 |
| crispr_U2OS_96   | crispr         | short  | U2OS   |                  72.8 |
| crispr_A549_144  | crispr         | long   | A549   |                  41   |
| crispr_A549_96   | crispr         | short  | A549   |                  42   |
| orf_A549_96      | orf            | long   | A549   |                  28.7 |
| orf_A549_48      | orf            | short  | A549   |                  35.6 |
| orf_U2OS_48      | orf            | short  | U2OS   |                  52.5 |
| orf_U2OS_96      | orf            | long   | U2OS   |                  42.5 |

# Percent Matching across modalities

| Chemical_Perturbation   | Genetic_Perturbation   |   Percent_Matching |
|:------------------------|:-----------------------|-------------------:|
| compound_A549_24        | crispr_A549_144        |                6.7 |
| compound_A549_24        | crispr_A549_96         |                7   |
| compound_A549_24        | orf_A549_96            |               11.1 |
| compound_A549_24        | orf_A549_48            |                8.2 |
| compound_A549_48        | crispr_A549_144        |                7.4 |
| compound_A549_48        | crispr_A549_96         |               10   |
| compound_A549_48        | orf_A549_96            |                8.2 |
| compound_A549_48        | orf_A549_48            |                6.5 |
| compound_U2OS_24        | crispr_U2OS_144        |               12.2 |
| compound_U2OS_24        | crispr_U2OS_96         |                7.4 |
| compound_U2OS_24        | orf_U2OS_48            |               10.1 |
| compound_U2OS_24        | orf_U2OS_96            |                8.5 |
| compound_U2OS_48        | crispr_U2OS_144        |                8.9 |
| compound_U2OS_48        | crispr_U2OS_96         |                8.2 |
| compound_U2OS_48        | orf_U2OS_48            |               11.1 |
| compound_U2OS_48        | orf_U2OS_96            |                7.8 |

# Percent Replicating DL features

| Description      | Perturbation   | time   | Cell   |   Percent_Replicating |
|:-----------------|:---------------|:-------|:-------|----------------------:|
| compound_U2OS_48 | compound       | long   | U2OS   |                  65.7 |
| crispr_U2OS_144  | crispr         | long   | U2OS   |                  39.3 |
| orf_U2OS_96      | orf            | long   | U2OS   |                  51.2 |

# Percent Replicating for DMSO wells in the same well position across plates

| Description             | Perturbation   | time   | Cell   |   Percent_Replicating |
|:------------------------|:---------------|:-------|:-------|----------------------:|
| CellProfiler_normalized | compound       | long   | U2OS   |                  40.6 |
| CellProfiler_spherized  | compound       | long   | U2OS   |                  26.6 |
| DeepProfiler_spherized  | compound       | long   | U2OS   |                  32.8 |

# Percent Replicating for compounds in the same well positon across plates

| Description             | Perturbation   | time   | Cell   |   Percent_Replicating |
|:------------------------|:---------------|:-------|:-------|----------------------:|
| CellProfiler_normalized | Compound       | long   | U2OS   |                  93.8 |
| CellProfiler_spherized  | Compound       | long   | U2OS   |                  61.9 |
| DeepProfiler_spherized  | Compound       | long   | U2OS   |                  62.2 |

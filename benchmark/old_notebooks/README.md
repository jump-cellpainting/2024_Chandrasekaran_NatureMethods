# Percent Replicating

| Description      | Perturbation   | time   | Cell   |   Percent_Replicating |
|:-----------------|:---------------|:-------|:-------|----------------------:|
| compound_A549_24 | compound       | short  | A549   |                  88.2 |
| compound_A549_48 | compound       | long   | A549   |                  96.4 |
| compound_U2OS_24 | compound       | short  | U2OS   |                  80.7 |
| compound_U2OS_48 | compound       | long   | U2OS   |                  74.5 |
| crispr_U2OS_144  | crispr         | long   | U2OS   |                  60   |
| crispr_U2OS_96   | crispr         | short  | U2OS   |                  72.8 |
| crispr_A549_144  | crispr         | long   | A549   |                  39   |
| crispr_A549_96   | crispr         | short  | A549   |                  42.6 |
| orf_A549_96      | orf            | long   | A549   |                  32.5 |
| orf_A549_48      | orf            | short  | A549   |                  33.1 |
| orf_U2OS_48      | orf            | short  | U2OS   |                  54.4 |
| orf_U2OS_96      | orf            | long   | U2OS   |                  45.6 |

# Percent Matching across modalities

| Chemical_Perturbation   | Genetic_Perturbation   |   Percent_Matching |
|:------------------------|:-----------------------|-------------------:|
| compound_A549_24        | crispr_A549_144        |                6.5 |
| compound_A549_24        | crispr_A549_96         |                7.4 |
| compound_A549_24        | orf_A549_96            |               10.8 |
| compound_A549_24        | orf_A549_48            |                7.2 |
| compound_A549_48        | crispr_A549_144        |                7.6 |
| compound_A549_48        | crispr_A549_96         |                9.6 |
| compound_A549_48        | orf_A549_96            |               10.1 |
| compound_A549_48        | orf_A549_48            |                6.2 |
| compound_U2OS_24        | crispr_U2OS_144        |               11.9 |
| compound_U2OS_24        | crispr_U2OS_96         |                6.7 |
| compound_U2OS_24        | orf_U2OS_48            |               11.4 |
| compound_U2OS_24        | orf_U2OS_96            |                7.8 |
| compound_U2OS_48        | crispr_U2OS_144        |                9.3 |
| compound_U2OS_48        | crispr_U2OS_96         |                9.3 |
| compound_U2OS_48        | orf_U2OS_48            |               11.1 |
| compound_U2OS_48        | orf_U2OS_96            |                7.8 |

# Percent Replicating DL features

| Description      | Perturbation   | time   | Cell   |   Percent_Replicating |
|:-----------------|:---------------|:-------|:-------|----------------------:|
| compound_U2OS_48 | compound       | long   | U2OS   |                  68   |
| crispr_U2OS_144  | crispr         | long   | U2OS   |                  39.3 |
| orf_U2OS_96      | orf            | long   | U2OS   |                  50   |

# Percent Replicating for DMSO wells in the same well position across plates

| Description             | Perturbation   | time   | Cell   |   Percent_Replicating |
|:------------------------|:---------------|:-------|:-------|----------------------:|
| CellProfiler_normalized | compound       | long   | U2OS   |                  42.2 |
| CellProfiler_spherized  | compound       | long   | U2OS   |                  26.6 |
| DeepProfiler_spherized  | compound       | long   | U2OS   |                  31.2 |

# Percent Replicating for compounds in the same well positon across plates

| Description             | Perturbation   | time   | Cell   |   Percent_Replicating |
|:------------------------|:---------------|:-------|:-------|----------------------:|
| CellProfiler_normalized | compound       | long   | U2OS   |                  79.7 |
| CellProfiler_spherized  | compound       | long   | U2OS   |                  73.8 |
| DeepProfiler_spherized  | compound       | long   | U2OS   |                  69.7 |

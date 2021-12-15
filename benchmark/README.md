# Percent Replicating

| Description      | Perturbation   | time   | Cell   |   Percent_Replicating |
|:-----------------|:---------------|:-------|:-------|----------------------:|
| compound_A549_24 | compound       | short  | A549   |                  87.6 |
| compound_A549_48 | compound       | long   | A549   |                  85   |
| compound_U2OS_24 | compound       | short  | U2OS   |                  80.1 |
| compound_U2OS_48 | compound       | long   | U2OS   |                  74.5 |
| crispr_U2OS_144  | crispr         | long   | U2OS   |                  62.3 |
| crispr_U2OS_96   | crispr         | short  | U2OS   |                  72.8 |
| crispr_A549_144  | crispr         | long   | A549   |                  43   |
| crispr_A549_96   | crispr         | short  | A549   |                  42   |
| orf_A549_96      | orf            | long   | A549   |                  26.2 |
| orf_A549_48      | orf            | short  | A549   |                  35.6 |
| orf_U2OS_48      | orf            | short  | U2OS   |                  52.5 |
| orf_U2OS_96      | orf            | long   | U2OS   |                  42.5 |

# Percent Matching across modalities

| Chemical_Perturbation   | Genetic_Perturbation   |   Percent_Matching |
|:------------------------|:-----------------------|-------------------:|
| Compound_A549_24        | ORF_A549_48            |                7.8 |
| Compound_A549_24        | ORF_A549_96            |                9.5 |
| Compound_A549_24        | CRISPR_A549_96         |                6.4 |
| Compound_A549_24        | CRISPR_A549_144        |                7.7 |
| Compound_A549_48        | ORF_A549_48            |                6.2 |
| Compound_A549_48        | ORF_A549_96            |                9.5 |
| Compound_A549_48        | CRISPR_A549_96         |               11.3 |
| Compound_A549_48        | CRISPR_A549_144        |                9.5 |
| Compound_U2OS_24        | ORF_U2OS_48            |               10.8 |
| Compound_U2OS_24        | ORF_U2OS_96            |                7.5 |
| Compound_U2OS_24        | CRISPR_U2OS_96         |                7.7 |
| Compound_U2OS_24        | CRISPR_U2OS_144        |               10.5 |
| Compound_U2OS_48        | ORF_U2OS_48            |                9.8 |
| Compound_U2OS_48        | ORF_U2OS_96            |                9.8 |
| Compound_U2OS_48        | CRISPR_U2OS_96         |                9.8 |
| Compound_U2OS_48        | CRISPR_U2OS_144        |                7.2 |

# Percent Replicating DL features

| Description      | Perturbation   | time   | Cell   |   Percent_Replicating |
|:-----------------|:---------------|:-------|:-------|----------------------:|
| ORF_U2OS_96      | ORF            | long   | U2OS   |                  41.9 |
| CRISPR_U2OS_144  | CRISPR         | long   | U2OS   |                  30.8 |
| Compound_U2OS_48 | Compound       | long   | U2OS   |                  59.5 |

# Percent Replicating for DMSO wells in the same well position across plates

| Description             | Perturbation   | time   | Cell   |   Percent_Replicating |
|:------------------------|:---------------|:-------|:-------|----------------------:|
| CellProfiler_normalized | Compound       | long   | U2OS   |                  53.1 |
| CellProfiler_spherized  | Compound       | long   | U2OS   |                  17.2 |
| DeepProfiler_spherized  | Compound       | long   | U2OS   |                  15.6 |

# Percent Replicating for compounds in the same well positon across plates

| Description             | Perturbation   | time   | Cell   |   Percent_Replicating |
|:------------------------|:---------------|:-------|:-------|----------------------:|
| CellProfiler_normalized | Compound       | long   | U2OS   |                  93.8 |
| CellProfiler_spherized  | Compound       | long   | U2OS   |                  61.9 |
| DeepProfiler_spherized  | Compound       | long   | U2OS   |                  62.2 |

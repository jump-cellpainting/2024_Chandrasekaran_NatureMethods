# Data splits

See [`1.create-data-splits.md`](1.create-data-splits.md) for details on creating these data splits.

## Representation learning

**Strategy:** Per cell line, train on CRISPRs and ORFs, validate on Compounds (one time point), and test on Compounds (the other time point).

Splits are saved at
[`representation_learning_split_example_1.csv`](representation_learning_split_example_1.csv);
each row corresponds to a unique perturbation/cell line/time point.

There are 3276 rows in total.

## Gene-compound matching

**Strategy:** Per cell line and time-point level (i.e. low or high), split by gene target into 60-20-20 for train-validate-test.

Splits are saved at [`gene_compound_split_example_1.csv`](gene_compound_split_example_1.csv);
each row corresponds to a CRISPR-ORF-Compounds triplet. 

There are 320 rows in total.

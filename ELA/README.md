# ELA: Extensible Learned Metadata Extraction for Data Lakes

This Repo contains the code and data for our data programming framework ELA.

## Data Preparation
### Public BI
Download the Public BI benchmark data set:
https://github.com/cwida/public_bi_benchmark/tree/dev/master

### Turl

## Valid Semantic Types
All used semantic types are defined in `data/extract/out/valid_types/types.json`. Here the list `type78` defines all types used for the Public BI benchmark and the list `type_turl` contains all 105 types used for the turl data corpus.

The assignment of semantic types to the columns for each data corpus can be found in `data/extract/out/valid_headers`. We have one CSV-File and one JSON-File for each data corpus, but both contain the same information of the assignment. The reason for this is that our system ELA works mostly with the JSON-File and the existing neuronal network Sato, which we used as prediction model currently, with the CSV-File.

## Labeled/Unlabeled/Test Split
To run the labeled/unlabeled/test split for the Public BI or for the Turl data corpus as explained in our paper, you have to run: 
`data/extract/data_split_labeled_unlabeled_test_absolut.py`

When executing the .py script, you must set corresponding required arguments:
```
python data_split_labeled_unlabeled_test_absolut.py --labeled_size [l_size] --test_size [t_size] --valid_headers [v_headers] --corpus [corp] --random_state [r_state]

# l_size: desired number of columns per semantic type in the labeled split (1-5 used in our paper)
# t_size: percentage size of the test split (default 0.2 (20%))
# v_headers: valid header CSV-File in data/extract/out/valid_headers/
# corp: name of the corpus "public_bi" for Public BI "turl" for Turl corpus
# random_state: the random state (seed) value. In the paper we uese 1-5    
```

## Generate Additional Training Data with ELA
### Run all LFs
To run all LF you have to do the follwing two steps. For the LF "Embedding Clustering" we provided a seperate .py script.  For the other LFs we have made one .py script which executes one LF after the other automatically.
- LF: Embedding Clustering

First, you must execute the script to calculate the embedded vector represantation for each column in `emb_clus/word_embedding/`:
```
python word_embedding.py --corpus [corp] --fraction [frac] --num_processes [n]

# corp: name of the corpus "public_bi" for Public BI "turl" for Turl corpus
# frac: fraction of the column-values to build the vector representation for one column.
# n: number of parallel processes
```

Second, run the following .py script in `emb_clus/without_knn/`. 
```
python run_cluster_n_classify.py
```

- LFs: [Column Headers, Value-Overlap, Value-Pattern]

Run the following .py script in `labeling_functions/`
```
python run_all_LFs.py
```

### Combine the outputs of the LFs
To combine the outputs of the four different LFs as described in our paper, run the following .py script in `labeling_functions/`
```
python run_combine_LFs_labels.py
```
This script will output all generated training data which we can now use to train or re-train an learned model in `labeling_functions/combined_LFs/gen_training_data`

## Train/Retrain existing learned model
We re-trained the existing Sato model with the additional generated training data.
To do this you have to run the following script in ELA/Sato/ela_integration/:
```
python run_exp_4_without_knn_combinedLFs.py

#### Note
## Please set-up first the variables from codeline 5-24 before runining the script!
# BASEPATH: Path to the Sato dir
# TYPENAME: "type78" for Public BI / "type_turl" for TURL
# corpus: "public_bi" / "turl"

## You also have to set-up the variable in codeline 33 & 35 for specifying your filepaths
```


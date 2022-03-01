# semantic_data_lake

## Setting Environment

- Environment Variables: (must all be adjusted in the .evn file in the root directory)
  - "PUBLIC_BI_BENCHMARK" => Absolute Path to the Public BI Benchmark Data Corpus
  - "TURL" => Absolute Path to TURL Data Corpus
  - "WORKING_DIR" => Absolute Path to the working directory "semantic_data_lake"
  - "TYPENAME" => equivalent to SATOs "TYPENAME". Name of the current types specified in the \data\exctract\out\valid_types.json file (e.g. if you want run experiment with TURL this variable should be "type_turl")

## To-Do´s
- delete all "os.environ["WORKING DIR"] =" lines in jupyter notebooks
- make use of the defines "CORPUS" Env-Variable in the different scripts
- adding the whole Gittables corpus as additional Data Lake (so far only abstraction_tables of it is integrated)
- complete labeling performance measure for appr3 with the usage of all columns (/labeling_performance/appr3)

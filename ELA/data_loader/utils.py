import os
import sys
import glob
import pandas as pd
from dotenv import load_dotenv
load_dotenv(override=True)

def get_all_gittables_tables_in_a_dir(domain="abstraction_tables", only_table_names=True):
    list_of_tables = glob.glob(
        os.path.join(os.environ["GITTABLES_DIR"], domain, "*.parquet"))
    if only_table_names:
        list_of_tables = list(
            map(lambda x: os.path.basename(x)[:-8], list_of_tables))
    return list_of_tables


def get_all_publicbi_domains(only_domain_names: bool = False):
    list_of_domains = glob.glob(
        os.path.join(os.environ.get("PUBLIC_BI_BENCHMARK"), "*/"))
    if only_domain_names:
        list_of_domains = list(
            map(lambda x: os.path.split(os.path.split(x)[0])[-1],
                list_of_domains))
    return list_of_domains


def get_all_publicbi_tables(domain: str, only_table_names: bool = False):
    """
    Function that returns a list of all Table-Paths for a given domain in Public BI Benchmark Data Corpus

    Parameters
    ----------
    domain: str
        The domain of the tables you would like to have the list of all tables available in Public BI Benchmark
    only_table_names: bool
        if true the returned only consist the tablename and not the complete absolute table-path

    Returns
    -------
    The list of all absolute table-paths
    """
    list_of_tables = glob.glob(
        os.path.join(os.environ.get("PUBLIC_BI_BENCHMARK"), domain, "*.csv"))
    if only_table_names:
        list_of_tables = list(
            map(lambda x: os.path.basename(x)[:-4], list_of_tables))
    return list_of_tables


def load_public_bi_table(domain, tablename, frac, with_header=True):
    df = pd.read_csv(os.path.join(os.environ.get("PUBLIC_BI_BENCHMARK"),
                                  domain, tablename + ".csv"),
                     sep="|",
                     header=None).sample(frac=frac)
    if with_header:
        df_header = pd.read_csv(os.path.join(
            os.environ.get("PUBLIC_BI_BENCHMARK"), domain, "samples",
            tablename + ".header.csv"),
                                sep="|")
        df.columns = df_header.columns
        return df
    else:
        return df


def get_all_turl_tables(only_table_names: bool = False):
    list_of_tables = glob.glob(os.path.join(os.environ["TURL"], "tables", "*.csv"))
    if only_table_names:
        list_of_tables = list(
            map(lambda x: os.path.basename(x)[:-4], list_of_tables))
    return list_of_tables

### table colum loader of raw data
def load_turl_tablecolumn(dataset_id:str, sample=5) -> list:
    """ Function which loads a tablecolum of the turl data corpus and returns it as list. 
    It only returns random samples of the given number from the specified list 
    
    Parameters
    ----------
    dataset_id: str
        dataset_id with the construction table+column_number (eg. 23122.csv+column_0 to load column 0 from table 23122)
    sample: int
        number of samples to select randomly from the column
        
    Returns
    -------
    column_values: list
        all 
    """
    
    table_id = dataset_id.split("+")[0]
    column_id = dataset_id.split("+")[1].split("_")[1]
    df_column = pd.read_csv(join(os.environ["TURL"], "tables",table_id), usecols=[int(column_id)]).sample(n=sample,replace=True, random_state=42)
    return df_column.iloc[:,0].values.tolist()

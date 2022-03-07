import os
from dotenv import load_dotenv

load_dotenv(override=True)
import numpy as np

gen_train_data = True
absolute = True

if os.environ["CORPUS"] == "public_bi":
    valid_headers = "public_bi_type78.json"
    word_embeddings = "word_embeddings_all_df.p"
elif os.environ["CORPUS"] == "turl":
    valid_headers = "turl_type_turl_header_valid.json"
    word_embeddings = "turl_1.0_df.p"

if __name__ == "__main__":
    if absolute:
        for labeled_data_size in [1, 2, 3, 4, 5]:
            for distance_threshold in [1e-2]:
                for random_state in [1, 2, 3, 4, 5]:
                    if gen_train_data:
                        os.system(
                            f"{os.environ['PYTHON']} cluster_n_classify.py --corpus {os.environ['CORPUS']} --gen_train_data {True} --valid_headers {valid_headers} --word_embeddings {word_embeddings} --distance_threshold {distance_threshold} --labeled_data_size {labeled_data_size} --absolute_numbers {True} --test_data_size 20 --random_state {random_state}"
                        )
                    else:
                        os.system(
                            f"{os.environ['PYTHON']} cluster_n_classify.py --corpus {os.environ['CORPUS']} --valid_headers {valid_headers} --word_embeddings {word_embeddings} --distance_threshold {distance_threshold} --labeled_data_size {labeled_data_size} --absolute_numbers {True} --test_data_size 20 --random_state {random_state}"
                        )
    else:
        for labeled_data_size in np.around(np.arange(0.2, 2.2, 0.2), 2):
            for distance_threshold in [1e-2]:
                for random_state in [2]:
                    if gen_train_data:
                        os.system(
                            f"{os.environ['PYTHON']} cluster_n_classify.py --corpus {os.environ['CORPUS']} --gen_train_data {True} --valid_headers {valid_headers} --word_embeddings {word_embeddings} --distance_threshold {distance_threshold} --labeled_data_size {labeled_data_size} --unlabeled_data_size {100.00-20.0-labeled_data_size} --test_data_size 20 --random_state {random_state}"
                        )
                    else:
                        os.system(
                            f"{os.environ['PYTHON']} cluster_n_classify.py --corpus {os.environ['CORPUS']} --valid_headers {valid_headers} --word_embeddings {word_embeddings} --distance_threshold {distance_threshold} --labeled_data_size {labeled_data_size} --unlabeled_data_size {100.00-20.0-labeled_data_size} --test_data_size 20 --random_state {random_state}"
                        )

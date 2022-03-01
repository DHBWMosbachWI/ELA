import numpy as np
import os
import sys

# set env-var
os.environ['BASEPATH'] = 'D:\\ELA\\Sato'
# path to the raw data
os.environ['RAW_DIR'] = '/mnt/batch/tasks/shared/LS_root/mounts/clusters/dhbwmlc-ds12-v2/code/Users/svenlangenecker/viznet-master/raw'
os.environ['SHERLOCKPATH'] = os.environ['BASEPATH']+'\\sherlock'
os.environ['EXTRACTPATH'] = os.environ['BASEPATH']+'\\extract'
#os.environ['PYTHONPATH'] = os.environ['PYTHONPATH']+':'+os.environ['SHERLOCKPATH']
#os.environ['PYTHONPATH'] = os.environ['PYTHONPATH']+':'+os.environ['BASEPATH']
os.environ['TYPENAME'] = 'type78'

sys.path.append("..")
sys.path.append(os.environ['BASEPATH'])

test_data_size = 20.0

corpus = "public_bi"


if __name__ == "__main__":

    # for index,percent in enumerate(np.arange(5,55,5)):
    for index, percent in enumerate([2,3,4,5]):
        for distance_threshold in [1e-6]:
            comment = f"embclus_{distance_threshold}_labeled{percent}_unlabeledAbsolute_test{test_data_size}"
            column_level_split_file_path = f"D:\\ELA\\data\\extract\\out\\labeled_unlabeled_test_split\\{corpus}_{percent}_absolute_{test_data_size}.json"
            embclus_gen_train_data_path = f"D:\\ELA\\emb_clus\\knn_classifier\\out\\gen_training_data\\{corpus}_gen_training_data_1_{distance_threshold}_{percent}_absolute_{test_data_size}.csv"
            pretrained_shelock_path = f"sherlock_retrain_embclus_{distance_threshold}_labeled{percent}_unlabeledAbsolute_test{test_data_size}.pt"
            pretrained_CRF_LDA_path = f"CRF+LDA_retrain_embclus_{distance_threshold}_labeled{percent}_unlabeledAbsolute_test{test_data_size}.pt"

            # retrain sherlock
            os.system(
                f"python ../model/train_sherlock.py -c ../model/params/publicbi/sherlock_retrain.txt  --comment {comment} --column_level_split_file_path {column_level_split_file_path} --embclus_gen_train_data_path {embclus_gen_train_data_path}"
            )

            # retrain sato
            os.system(
                f"python ../model/train_CRF_LC.py -c ../model/params/publicbi/CRF+LDA_retrain.txt --pre_trained_sherlock_path {pretrained_shelock_path} --comment {comment} --column_level_split_file_path {column_level_split_file_path} --embclus_gen_train_data_path {embclus_gen_train_data_path}"
            )

            # validate sato
            os.system(
                f"python ../model/train_CRF_LC.py -c ../model/params/publicbi/CRF+LDA_eval.txt --model_list {pretrained_CRF_LDA_path} --comment eval_{comment} --column_level_split_file_path {column_level_split_file_path}"
            )

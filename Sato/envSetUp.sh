#export BASEPATH=/mnt/batch/tasks/shared/LS_root/mounts/clusters/$HOSTNAME/code/Users/svenlangenecker/sato
export BASEPATH=D:\\20120321_DHBW_AZUREML\\sato
# RAW_DIR can be empty if using extracted feature files.
export RAW_DIR=/mnt/batch/tasks/shared/LS_root/mounts/clusters/$HOSTNAME/code/Users/svenlangenecker/viznet-master/raw
export SHERLOCKPATH=$BASEPATH/sherlock
export EXTRACTPATH=$BASEPATH/extract
export PYTHONPATH=$PYTHONPATH:$SHERLOCKPATH
export PYTHONPATH=$PYTHONPATH:$BASEPATH
export TYPENAME='type_turl'
export PUBLICBIBENCHMARK_DIR=D:\\semantic_data_lake\\semantic_data_lake\\data\\benchmark
export TURL_DIR=D:\\TURL\\tables

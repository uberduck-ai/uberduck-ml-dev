#!/bin/bash
gcloud auth activate-service-account --key-file=/secrets/gcloud_key.json
export GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcloud_key.json
export BUCKET=$1
export UBMLEXP_GIT=$2
export WSN=$3
export FILELIST_TRAIN=$4
export FILELIST_VAL=$5
export NAME=$6
export CONFIG=$7
export RESULT_DIR=${BUCKET}/results
export DATA_DIR=${BUCKET}/data
export MODEL_DIR=${BUCKET}/models
export LOG_DIR=${RESULT_DIR}/${NAME}/logs
export CHECKPOINT_PATH=${RESULT_DIR}/${NAME}/checkpoints
export WARM_START_NAME=${MODEL_DIR}/${WSN}
#./uberduck-ml-exp/experiments/taco2_lj_lachow/config.json
git clone -b sam-exp $UBMLEXP_GIT
gcsfuse $BUCKET /bucket
python -m uberduck_ml_dev.exec.train_tacotron2 --config $CONFIG --log_dir $LOG_DIR --checkpoint_path $CHECKPOINT_PATH --training_audiopaths_and_text $FILELIST_TRAIN --val_audiopaths_and_text $FILELIST_VAL --warm_start_name $WARM_START_NAME

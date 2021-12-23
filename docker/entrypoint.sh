#!/bin/bash
gcloud auth activate-service-account --key-file=/secrets/gcloud_key.json
#sleep 3600
export GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcloud_key.json
export BUCKET=$1
export UBMLEXP_GIT=$2
export WSN=$3
export FILELIST_TRAIN=$4
export FILELIST_VAL=$5
export NAME=$6
export CONFIG=$7
export BUCKET_LOCAL=$8
export RESULT_DIR=${BUCKET_LOCAL}/results
export DATA_DIR=${BUCKET_LOCAL}/data
export MODEL_DIR=${BUCKET_LOCAL}/models
export LOG_DIR=${RESULT_DIR}/${NAME}/logs
export CHECKPOINT_PATH=${RESULT_DIR}/${NAME}/checkpoints
export WARM_START_NAME=${MODEL_DIR}/${WSN}

git clone $UBMLEXP_GIT
gcsfuse --implicit-dirs $BUCKET $BUCKET_LOCAL
echo "BUCKET: $BUCKET"
echo "UBMLEXP_GIT: $UBMLEXP_GIT"
echo "WSN: $WSN"
echo "FILELIST_TRAIN: $FILELIST_TRAIN"
echo "FILELIST_VAL: $FILELIST_VAL"
echo "NAME: $NAME"
echo "CONFIG: $CONFIG"
echo "BUCKET_LOCAL: $BUCKET_LOCAL"
echo "RESULT_DIR: $RESULT_DIR"
echo "MODEL_DIR: $MODEL_DIR"
echo "LOG_DIR: $LOG_DIR"
echo "CHECKPOINT_PATH: $CHECKPOINT_PATH"
echo "WARM_START_NAME: $WARM_START_NAME"

python -m uberduck_ml_dev.exec.train_tacotron2 --config $CONFIG --log_dir $LOG_DIR --checkpoint_path $CHECKPOINT_PATH --training_audiopaths_and_text $FILELIST_TRAIN --val_audiopaths_and_text $FILELIST_VAL --warm_start_name $WARM_START_NAME
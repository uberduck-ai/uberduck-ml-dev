#!/bin/bash
gcloud auth activate-service-account --key-file=/secrets/gcloud_key.json
#sleep 3600
export GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcloud_key.json
export BUCKET=$1
export BUCKET_LOCAL=$2
export UBMLEXP_GIT=$3
export WARM_START_NAME=$4
export FILELIST_TRAIN=$5
export FILELIST_VAL=$6
export CONFIG=$7
export LOG_PATH=$8
export CHECKPOINT_PATH=$9

git clone $UBMLEXP_GIT
gcsfuse --implicit-dirs $BUCKET $BUCKET_LOCAL
echo "BUCKET: $BUCKET"
echo "BUCKET_LOCAL: $BUCKET_LOCAL"
echo "UBMLEXP_GIT: $UBMLEXP_GIT"
echo "FILELIST_TRAIN: $FILELIST_TRAIN"
echo "FILELIST_VAL: $FILELIST_VAL"
echo "NAME: $NAME"
echo "CONFIG: $CONFIG"
echo "LOG_PATH: $LOG_PATH"
echo "CHECKPOINT_PATH: $CHECKPOINT_PATH"
echo "WARM_START_NAME: $WARM_START_NAME"
sleep 3600
python -m uberduck_ml_dev.exec.train_tacotron2 --config $CONFIG --log_dir $LOG_PATH --checkpoint_path $CHECKPOINT_PATH --training_audiopaths_and_text $FILELIST_TRAIN --val_audiopaths_and_text $FILELIST_VAL --warm_start_name $WARM_START_NAME

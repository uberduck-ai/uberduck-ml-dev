#!/bin/bash
gcloud auth activate-service-account --key-file=/secrets/gcloud_key.json
export GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcloud_key.json
if [[ -z "$1" ]]; then
    echo "Enter BUCKET: "
    read inp
    export BUCKET=$inp
else
    export BUCKET=$1
fi
if [[ -z "$2" ]]; then
    echo "Enter Experiemnts repo to clone: "
    read inp
    export UBMLEXP_GIT=$inp
else
    export UBMLEXP_GIT=$2
fi
if [[ -z "$3" ]]; then
    echo "Enter warm start name: "
    read inp
    export WARM_START_NAME=$inp
else
    export WARM_START_NAME=$3
fi
if [[ -z "$4" ]]; then
    echo "Enter training filelist: "
    read inp
    export FILELIST_TRAIN=$inp
else
    export FILELIST_TRAIN=$4
fi
if [[ -z "$5" ]]; then
    echo "Enter validation filelist: "
    read inp
    export FILELIST_VAL=$inp
else
    export FILELIST_VAL=$5
fi
if [[ -z "$6" ]]; then
    echo "Enter config: "
    read inp
    export CONFIG=$inp
else
    export CONFIG=$6
fi

git clone $UBMLEXP_GIT
# gcsfuse --implicit-dirs $BUCKET $BUCKET_LOCAL
gcsfuse --implicit-dirs $BUCKET /root/bucket

echo "AIP_TENSORBOARD_LOG_DIR: $AIP_TENSORBOARD_LOG_DIR"
echo "AIP_CHECKPOINT_DIR: $AIP_CHECKPOINT_DIR"
echo "AIP_MODEL_DIR: $AIP_MODEL_DIR"
echo "BUCKET: $BUCKET"
# Mount Tensorboard log dir
proto="gs://"
# url=$(echo $AIP_TENSORBOARD_LOG_DIR | sed -e s,$proto,,g)
# path="$(echo $url | grep / | cut -d/ -f2-)"
# host=$(echo $url | cut -d/ -f1)
# echo "Mounting path: $path. host: $host" 
# gcsfuse --only-dir $path $host /root/logs
# Mount checkpoint dir
url=$(echo $AIP_CHECKPOINT_DIR | sed -e s,$proto,,g)
path="$(echo $url | grep / | cut -d/ -f2-)"
host=$(echo $url | cut -d/ -f1)
echo "Mounting path: $path. host: $host" 
gcsfuse --only-dir $path $host /root/checkpoints
# Mount model dir
url=$(echo $AIP_MODEL_DIR | sed -e s,$proto,,g)
path="$(echo $url | grep / | cut -d/ -f2-)"
host=$(echo $url | cut -d/ -f1)
echo "Mounting path: $path. host: $host" 
gcsfuse --only-dir $path $host /root/checkpoints

echo "UBMLEXP_GIT: $UBMLEXP_GIT"
echo "FILELIST_TRAIN: $FILELIST_TRAIN"
echo "FILELIST_VAL: $FILELIST_VAL"
echo "CONFIG: $CONFIG"
echo "WARM_START_NAME: $WARM_START_NAME"
python -m uberduck_ml_dev.exec.train_tacotron2 --config $CONFIG \
	--log_dir $AIP_TENSORBOARD_LOG_DIR  \
	--checkpoint_path /root/checkpoints \
	--training_audiopaths_and_text $FILELIST_TRAIN \
	--val_audiopaths_and_text $FILELIST_VAL \
	--warm_start_name $WARM_START_NAME

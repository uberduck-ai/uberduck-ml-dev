#!/bin/bash
gcloud auth activate-service-account --key-file=/secrets/gcloud_key.json
#sleep 3600
export GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcloud_key.json
if [[ -z "$1" ]] then
    echo "Enter BUCKET: "
    read inp
    export BUCKET=$inp
else
    export BUCKET=$1
fi
if [[ -z "$2" ]] then
    echo "Enter BUCKET_LOCAL: "
    read inp
    export BUCKET_LOCAL=$inp
else
    export BUCKET_LOCAL=$2
fi
if [[ -z "$3" ]] then
    echo "Enter Experiemnts repo to clone: "
    read inp
    export UBMLEXP_GIT=$inp
else
    export UBMLEXP_GIT=$3
fi
if [[ -z "$4" ]] then
    echo "Enter warm start name: "
    read inp
    export WARM_START_NAME=$inp
else
    export WARM_START_NAME=$4
fi
if [[ -z "$5" ]] then
    echo "Enter training filelist: "
    read inp
    export FILELIST_TRAIN=$inp
else
    export FILELIST_TRAIN=$5
fi
if [[ -z "$6" ]] then
    echo "Enter validation filelist: "
    read inp
    export FILELIST_VAL=$inp
else
    export FILELIST_VAL=$6
fi
if [[ -z "$7" ]] then
    echo "Enter config: "
    read inp
    export CONFIG=$inp
else
    export CONFIG=$7
fi
if [[ -z "$8" ]] then
    echo "Enter log path: "
    read inp
    export LOG_PATH=$inp
else
    export LOG_PATH=$8
fi
if [[ -z "$9" ]] then
    echo "Enter checkpoint path: "
    read inp
    export CHECKPOINT_PATH=$inp
else
    export CHECKPOINT_PATH=$9
fi

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

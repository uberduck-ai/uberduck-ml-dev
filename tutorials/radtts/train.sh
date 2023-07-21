#!/bin/bash

cd /path/to/uberduck_ml_dev
# remember to set training and eval filelists, heteronyms_path and phoneme_dict_path vocoder_config_path and vocoder_checkpoint_path in demo_config.json
python uberduck_ml_dev/exec/train_radtts_with_ray.py --config /path/to/uberduck_ml_dev/tutorials/radtts/demo_config.json

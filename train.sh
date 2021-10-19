#!/bin/bash

# This is a basic training quick-start script.
# Code credits: https://unix.stackexchange.com/a/505342


helpFunction()
{
   echo ""
   echo "Usage: $0 ConfigPath"
   echo -e "\tConfig (typically JSON) path containing training HParams."
   exit 1 # Exit script after printing help
}

# Print helpFunction in case parameters are empty
if [ -z "$1" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi


# Basic training quickstart. $1 is config path.

echo "Using configuration path: $1"

python -m uberduck_ml_dev.exec.train_tacotron2 \
                                              --config "$1"

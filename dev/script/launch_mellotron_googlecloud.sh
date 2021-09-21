#launch gpu... not fininalized
export IMAGE_FAMILY="pytorch-latest-gpu"
export ZONE="us-west-2-b"
export INSTANCE_NAME="uberduck-sam"

gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --custom-memory=16384MB \
  --custom-cpu=8 \
  --image-family=$IMAGE_FAMILY \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --accelerator="type=nvidia-tesla-t4,count=1" \
  --metadata="install-nvidia-driver=True"

#log in
gcloud compute ssh $INSTANCE_NAME

git clone https://github.com/NVIDIA/mellotron.git
cd mellotron
git submodule init; git submodule update

cd ..
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
cd ..

wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xf LJSpeech-1.1.tar.bz2 -v 

pip install librosa
sudo apt-get install libsndfile1-dev
pip install tensorboardX
pip install inflect
pip install tensorflow==1.15.5
pip install gdown
pip install soundfile
conda install libsndfile #deactivate log out and log in

cd mellotron
mkdir outdir
mkdir logdir

cd filelists
rm lib*
vim ljs_audiopaths_text_sid_train_filelist.txt #add LJspeech path (cntl+v, shft+)
vim ljs_audiopaths_text_sid_val_filelist.txt #add LJspeech path
cd ..

python train.py --output_directory=outdir --log_directory=logdir

#get existing mellotron
pip install gdown
gdown --id 1UwDARlUl8JvB2xSuyMFHFsIWELVpgQD4

# #launch gpu... not fininalized
# export IMAGE_FAMILY="pytorch-latest-gpu"
# export ZONE="us-west1-b"
# export INSTANCE_NAME="uberduck-1"



dvc remote modify --local uberduck user pub_qwmyyewvnpxhjwwufv
dvc remote modify --local uberduck password pk_731b4c9c-0d17-4537-a6a5-4900bacd540c

echo "set enable-bracketed-paste off" >> ~/.inputrc
# gcloud compute instances create $INSTANCE_NAME \
#   --zone=$ZONE \
#   --custom-memory=16384MB \
#   --custom-cpu=8 \
#   --image-family=$IMAGE_FAMILY \
#   --image-project=deeplearning-platform-release \
#   --maintenance-policy=TERMINATE \
#   --accelerator="type=nvidia-tesla-a100,count=1" \
#   --metadata="install-nvidia-driver=True"

# export INSTANCE_NAME="uberduck-100"
# gcloud compute instances create $INSTANCE_NAME \
#   --zone=$ZONE \
#   --machine-type=a2-highgpu-1g \
#   --image-family=$IMAGE_FAMILY \
#   --image-project=deeplearning-platform-release \
#   --maintenance-policy=TERMINATE
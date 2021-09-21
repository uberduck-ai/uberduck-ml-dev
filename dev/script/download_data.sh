#modifying .bash shared between sessions... 5 gb base storage...  maybe need venv?

conda install -c conda-forge mamba # installs much faster than conda
mamba install -c conda-forge dvc

#export UBERDUCK_ACCESS_TOKEN=your-github-access-token
export UBERDUCK_ACCESS_TOKEN="5dc7e850038038abdceecab854cd2bc0a375d2ad"

git clone https://sjkoelle:$UBERDUCK_ACCESS_TOKEN@git.uberduck.ai/uberduck-internal/eminem.git

export API_KEY="pub_qwmyyewvnpxhjwwufv"
export API_SECRET="pk_731b4c9c-0d17-4537-a6a5-4900bacd540c"

dvc remote modify --local uberduck user $API_KEY
dvc remote modify --local uberduck password $API_SECRET



mkdir data

cd data
wget https://www.openslr.org/resources/60/dev-clean.tar.gz & 
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 & 
wget https://datashare.ed.ac.uk/download/DS_10283_3443.zip & 

tar -xvf dev-clean.tar.gz
tar -xvf LJSpeech-1.1.tar.bz2

mkdir uberduck
cd uberduck

git clone https://git.uberduck.ai/uberduck-internal/eminem.git
cd eminem
dvc pull

git clone https://git.uberduck.ai/uberduck-internal/kanye-rap.git
git clone https://git.uberduck.ai/uberduck-internal/alex-trebek.git
git clone https://git.uberduck.ai/uberduck-internal/ben-shapiro.git
git clone https://git.uberduck.ai/uberduck-internal/michael-rosen.git
git clone https://git.uberduck.ai/uberduck-internal/steve-harvey.git

#https://cloud.google.com/compute/docs/disks/add-persistent-disk
gcloud compute disks create uberduck-experiments-v0 \
  --size 250 \
  --type https://www.googleapis.com/compute/v1/projects/uberduck/zones/us-west2-b/diskTypes/pd-ssd

gcloud compute instances attach-disk uberduck-sam \
  --disk uberduck-experiments-v0

gcloud compute ssh $INSTANCE_NAME
 sudo lsblk

#sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb
 

sudo mkdir -p /mnt/disks/uberduck-experiments-v0

sudo mount -o discard,defaults /dev/sdb /mnt/disks/uberduck-experiments-v0

sudo mount /dev/sdb /mnt/disks/uberduck-experiments-v0

sudo chmod a+w /mnt/disks/uberduck-experiments-v0


git clone https://github.com/uberduck-ai/uberduck-ml-dev.git
#follow commands to automount disk
#UUID="a2ced836-f22f-479f-839f-e0fbedef78ec" TYPE="ext4"

gcloud compute instances detach-disk uberduck-sam-util --disk=uberduck-experiments-v0 --zone us-west2-b

gcloud compute instances move uberduck-sam-util \
    --zone us-west2-b --destination-zone us-west2-c




gcloud compute snapshots delete backup-uberduck-experiments-v0
gcloud compute snapshots delete backup-uberduck-sam-util

gcloud compute snapshots delete uberduck-experiments-v0
gcloud compute snapshots delete uberduck-sam-util


gcloud compute disks snapshot uberduck-experiments-v0  \
    --snapshot-names backup-uberduck-experiments-v0  \
    --zone us-west2-b
    
gcloud compute disks snapshot uberduck-sam-util  \
    --snapshot-names backup-uberduck-sam-util  \
    --zone us-west2-b
    
gcloud compute disks create backup-uberduck-sam-util --source-snapshot backup-uberduck-sam-util \
    --zone us-west2-c

gcloud compute disks create backup-uberduck-experiments-v0 --source-snapshot backup-uberduck-experiments-v0 \
    --zone us-west2-c


gcloud compute instances delete uberduck-sam-util-west2c


export IMAGE_FAMILY="pytorch-latest-gpu"

gcloud compute instances create uberduck-sam-util-west2c \
    --zone us-west2-c \
    --custom-memory=16384MB \
    --custom-cpu=8 \
    --image-family=$IMAGE_FAMILY \
    --disk name=backup-uberduck-sam-util,boot=yes,mode=rw \
    --disk name=backup-uberduck-experiments-v0,mode=rw \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=False"



# gcloud compute instances create uberduck-sam-util-west2c \
#     --zone us-west2-c \
#     --custom-memory=16384MB \
#     --custom-cpu=8 \
#     --image-family=$IMAGE_FAMILY \
#     --disk name=uberduck-experiments-v0,mode=rw \
#     --image-project=deeplearning-platform-release \
#     --maintenance-policy=TERMINATE \
#     --metadata="install-nvidia-driver=False"

gcloud compute instances detach-disk uberduck-sam-util --disk=uberduck-experiments-v0 --zone us-west2-c

gcloud compute instances detach-disk uberduck-sam-util-west2c --disk=backup-uberduck-experiments-v0 --zone us-west2-c

gcloud compute instances detach-disk uberduck-sam-util-west2c --disk=backup-uberduck-sam-util --zone us-west2-c

export IMAGE_FAMILY="pytorch-latest-gpu"
gcloud compute instances create uberduck-sam-t4-west2c \
    --zone us-west2-c \
    --custom-memory=16384MB \
    --custom-cpu=8 \
    --image-family=$IMAGE_FAMILY \
    --disk name=backup-uberduck-sam-util,boot=yes,mode=rw \
    --disk name=backup-uberduck-experiments-v0,mode=rw \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    --metadata="install-nvidia-driver=True"


sudo mount -o discard,defaults /dev/sdb /mnt/disks/uberduck-experiments-v0
sudo chmod a+w /mnt/disks/uberduck-experiments-v0

pip install -r requirements.txt
#maybe also
###pip install librosa
#pip install inflect
#pip install tensorboardX==1.1
#pip install tensorflow==1.15.2
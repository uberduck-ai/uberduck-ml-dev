from src.utils import get_dataset
from src.models import TTSModel
from src.models import TTSTrainer

experiment_filelist_dir

source_files = [experiment_filelist_dir + 'eminem_all.txt',
            '/mnt/disk/uberduck-experiments-v0/data/uberduck/kanye-rap/all.txt',
            '/mnt/disk/uberduck-experiments-v0/data/uberduck/alex-trebek/all.txt',
            '/mnt/disk/uberduck-experiments-v0/data/uberduck/ben-shapiro/all.txt',
            '/mnt/disk/uberduck-experiments-v0/data/uberduck/michael-rosen/all.txt',
            '/mnt/disk/uberduck-experiments-v0/data/uberduck/steve-harvey/all.txt',
            '/mnt/disk/uberduck-experiments-v0/data/uberduck/LJSpeech-1.1/metadata.csv',
            '/mnt/disk/uberduck-experiments-v0/data/uberduck/LibriTTS/metadata_mellotron.txt',
            '/mnt/disk/uberduck-experiments-v0/data/uberduck/vctk/metadata.txt']

synthesize_speakerids(source_files) #need to change
reindex()

train,val  = get_testval_split()

TTSModel('mellotron')
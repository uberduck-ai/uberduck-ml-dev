from src.utils import add_speakerid, parse_vctk, parse_libritts_mellotron, parse_lj7
import numpy as np

filelist_dir = '/mnt/disks/uberduck-experiments-v0/uberduck-ml-dev/experiments/processed_metadata'
metalist_dir = '/mnt/disks/uberduck-experiments-v0/uberduck-ml-dev/experiments/metadata_collections'
vctk_folder = '/mnt/disks/uberduck-experiments-v0/data/vctk/'
libritts_folder = '/mnt/disks/uberduck-experiments-v0/data/LibriTTS'
lj7_folder = '/mnt/disks/uberduck-experiments-v0/data/LJSpeech-1.1'
mellotron_filelist = '/mnt/disks/uberduck-experiments-v0/uberduck-ml-dev/src/vendor/mellotron/libritts_train_clean_100_audiopath_text_sid_shorterthan10s_atleast5min_train_filelist.txt',
# ljs_audiopaths_text_sid_train_filelist.txt
#'/mnt/disk/uberduck-experiments-v0/models/mellotron/vendor/mellotron_libritts.pt' #mellotron_ljs.pt
uberduck_folder = '/mnt/disks/uberduck-experiments-v0/data/uberduck/'
#uberduck_names = ['eminem','kanye-rap','alex-trebek','ben-shapiro','michael-rosen','steve-harvey']
uberduck_names = ['eminem']

uberduck_filelists_multispeaker = {}
nuberduck = len(uberduck_names)
uberduck_processed_files = np.zeros(nuberduck, dtype = object)
for d in range(nuberduck):
    uberduck_filelists_multispeaker[d] = add_speakerid(uberduck_folder + uberduck_names[d] + '/all.txt')
    uberduck_processed_files[d] = filelist_dir  +'/'+uberduck_names[d] + '_all_processed.txt'
    uberduck_filelists_multispeaker[d].to_csv(uberduck_processed_files[d])

np.save(metalist_dir + 'uberduck_processed_files', uberduck_processed_files)

vctk_filelist = parse_vctk(vctk_folder)
vctk_processed_file = filelist_dir + 'vctk' + 'all_processed.txt')
vctk_filelist.to_csv(vctk_processed_file)
np.save(metalist_dir + 'vctk_processed_file', vctk_processed_file)

#go thorugh mellotron folder and build filelist
#alternative is using existing filelist
#in this case we alter index for other experiments
libritts_filelist_mellotron = parse_libritts_mellotron(libritts_folder, mellotron_filelist)
libritts_filelist = parse_libritts(libritts_folder)
libritts_processed_file = filelist_dir + 'libritts' + 'all_processed.txt'
np.save(metalist_dir + 'libritts_processed_file', libritts_processed_file)

lj7_processed = add_speaker_id(lj7_folder)
lj7_processed_file = filelist_dir + 'lj7' + 'all_processed.txt'
lj7_processed.to_csv(lj7_processed_file)
np.save(metalist_dir + 'lj7_processed_file', lj7_processed_file)
import pandas as pd

def add_speakerid(source_file, speaker_key = 0):

    # if speaker_ids == None:
    #     speaker_ids = np.asarray(np.ones(data.shape[0], dtype = int) * speaker_key, dtype = int)
    data = pd.read_csv(source_file, sep = "|",header=None, error_bad_lines=False)
    if data.shape[1] == 3:
        if type(data[2]]) == int:
            pass

    if data.shape[1] == 2:
        speaker_ids = np.asarray(np.ones(data.shape[0], dtype = int) * speaker_key, dtype = int)
        data[2] = speaker_ids

    return(data)
    #data.to_csv(source_file + '_multispeaker.txt', sep = "|",header=None,index= False)

#def synthesize_speakerids(source_files):
def synthesize_speakerids(data_dict):

    source_files = list(data_dict.keys())
    nspeakers_cumulative = 0
    speaker_offset = {}
    for source_file in source_files:
        data = data_dict[source_file]
        nspeakers = len(np.unique(data[3]))
        data[3] = data[3] + nspeakers_cumulative
        data_dict[source_file] = data
        speaker_offset[source_file] = nspeakers_cumulative
        nspeakers_cumulative = nspeakers_cumulative + nspeakers

    return(data_dict)  

def get_dataset(source_files, wav_folders,data_id, data_folder):
    '''
    source_files: the identities of the multispeaker metadatasets
    '''
    os.mkdir(data_folder + '/' + data_id)
    data_dict = {}

    for s in range(len(source_files)):
        source_file = source_files[s]
        wav_folder = wav_folders[s]
        data_dict[source_file] = add_speakerid(source_file)
        npoints = data_dict[source_file].shape[0]
        for i in range(npoints):
            data_dict[source_file][:,0] = wav_folder + data_dict[source_file][i,0]
    data_dict = synthesize_speakerids(data_dict)
    combined_data = pd.concat(list(data_dict.values()))
    combined_data.to_csv(data_folder + '/' + data_id + source_file, sep = "|",header=None, error_bad_lines=False)
    

def parse_libritts():
    #wavs /LibriTTS/dev-clean/1272/135031

#folder = '/mnt/disks/uberduck-experiments-v0/data/vctk/'
def parse_vctk(folder):

    wav_dir = folder + 'wav48_silence_trimmed'
    txt_dir = folder + 'txt'
    speakers = os.listdir(wav_dir):
    data_dict = {}
    for f in wav_dir:
        wav_files_speaker = os.listdir(wav_dir + f):
        data_dict[wav_dir] = pd.DataFrame()
        text = np.asarray([])
        wav_file = np.asarray([])
        for g in range(len(files)):
            text = np.append(text, pd.read_csv(f + files[g]))
            wav_file = np.append(text, pd.read_csv(txt_dir + files[g]))
            data_dict[wav_dir][:,0] = pd.DataFrame()
    txt_folders = os.listdir(folder + 'txt'):
    
    #flacs (wav like) vctk/wav48_silence_trimmed/p230/
    #folder is speaker_id

    # source_file in source_files:
    #     data_dict[source_file].to_csv(data_folder + '/' + data_id + source_file, sep = "|",header=None, error_bad_lines=False)
#def compute_statistics(data_dict):


# def add_speakerid(source_file, speaker_ids = None, speaker_key = None):
#     data = pd.read_csv(source_file, sep = "|",header=None, error_bad_lines=False)
#     if speaker_ids == None:
#         speaker_ids = np.asarray(np.ones(data.shape[0], dtype = int) * speaker_key, dtype = int)
#     for i in range(data.shape[0]):
#         data[0][i] = '/Users/samsonkoelle/Downloads/eminem_14/' + data[0][i] 
#     data[2] = speaker_ids
#     data.to_csv(source_file + '_multispeaker.txt', sep = "|",header=None,index= False)

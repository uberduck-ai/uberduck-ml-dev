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
        #data = pd.read_csv(source_file, sep = "|",header=None, error_bad_lines=False)
        nspeakers = len(np.unique(data[3]))
        #speakers_cumulative = data[3] + nspeakers_cumulative
        data[3] = data[3] + nspeakers_cumulative
        data_dict[source_file] = data
        speaker_offset[source_file] = nspeakers_cumulative
        nspeakers_cumulative = nspeakers_cumulative + nspeakers

    return(data_dict)  

def get_dataset(source_files, data_id, data_folder):

    os.mkdir(data_folder + '/' + data_id)
    data_dict = {}
    for source_file in source_files:
        data_dict[source_file] = add_speakerid(source_file)
    data_dict = synthesize_speakerids(data_dict)
    combined_data = pd.concat(list(data_dict.values()))
    combined_data.to_csv(data_folder + '/' + data_id + source_file, sep = "|",header=None, error_bad_lines=False)
    
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

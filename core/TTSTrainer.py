import uberduck as ud

class TTSTrainer():

    def __init__(model_type, model_opts, data, warm_start = False):

        self.data = data

        if warm_start == False:
            model = ud.TTSModel.create(model_type, model_options)


    def train():

        for batch in enumerate(data):
            #fill in 

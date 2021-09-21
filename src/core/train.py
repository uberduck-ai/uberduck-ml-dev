import uberduck as ud

#parser method

#model = TTSModel.create(model_name, model_options)
trainer = ud.TTSTrainer("taco",model_opts, data, save_model = True) #
model = trainer.go()
#save model

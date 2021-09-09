import uberduck as ud

#model = TTSModel.create(model_name, model_options)
trainer = ud.TTSTrainer("taco", data, save_model = True)
model = trainer.go()
#TTSTrainer.
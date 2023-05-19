# 40k config
# TODO (Sam): move to separate files
DEFAULTS = {
  "train": {
    "log_interval": 200,
    "seed": 1234,
    "epochs": 20000,
    "learning_rate": 1e-4,
    "betas": [0.8, 0.99],
    "eps": 1e-9,
    "batch_size": 4,
    "fp16_run": True,
    "lr_decay": 0.999875,
    "segment_size": 12800,
    "init_lr_ratio": 1,
    "warmup_epochs": 0,
    "c_mel": 45,
    "c_kl": 1.0
  },
  "data": {
    "max_wav_value": 32768.0,
    "sampling_rate": 40000,
    "filter_length": 2048,
    "hop_length": 400,
    "win_length": 2048,
    "n_mel_channels": 125,
    "mel_fmin": 0.0,
    "mel_fmax": None
  },
  "model": {
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0,
    "resblock": "1",
    "resblock_kernel_sizes": [3,7,11],
    "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
    "upsample_rates": [10,10,2,2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16,16,4,4],
    "use_spectral_norm": False,
    "gin_channels": 256,
    "spk_embed_dim": 109
  }
}
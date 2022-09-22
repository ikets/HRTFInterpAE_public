# IWAENC2022's config
config_0 = {
    'fft_length': 256,
    'max_frequency': 16000,

    'model': 'AE',
    'channel_En_z': 128,
    'dim_z': 64,
    'channel_De_z': 128,

    'channel_hyper': 64,
    'hlayers_hyper': 1,
    'droprate': 0.0,

    'num_sub': 77, # sub for train
    'batch_size': 32, # set < VRAM(GB)/0.3
    'epochs': 1000,
    'save_frequency': 1500,
    'learning_rate': 0.001,
    'loss_weights': {
                        'lsd': 1,
                        'cdintra_z': 1
                    }, 
    'num_gpus': 1,
    'num_pts': 440, # B'(# measurement positions) for train/valid
    'use_itd_gt': True, # use true itd
    'fs_upsampling': 8*48000, # freq to obtain ITD
    'green': False, 
}

# debug
config_1 = {
    'fft_length': 256,
    'max_frequency': 16000,

    'model': 'AE',
    'channel_En_z': 128,
    'dim_z': 64,
    'channel_De_z': 128,

    'channel_hyper': 64,
    'hlayers_hyper': 1,
    'droprate': 0.0,

    'num_sub': 77, # sub for train
    'batch_size': 32, # set < VRAM(GB)/0.3
    'epochs': 1,
    'save_frequency': 400,
    'learning_rate': 0.001,
    'loss_weights': {
                        'lsd': 1,
                        'cdintra_z': 1
                    }, 
    'num_gpus': 1,
    'num_pts': 440, # B'(# measurement positions) for train/valid
    'use_itd_gt': True, # use true itd
    'fs_upsampling': 8*48000, # freq to obtain ITD
    'green': False, 
}
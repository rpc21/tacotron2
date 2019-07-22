import tensorflow as tf
from text import symbols


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500,
        iters_per_checkpoint=1000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:34323",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=True, #False,
#        training_files='filelists/ljs_audio_text_train_filelist.txt',
#        training_files='/scratch/speech/datasets/Tacotron_LibriTTS/LibriTTS_train_100.txt',
#        training_files='/scratch/speech/datasets/IEMOCAP_happy_sad_train.pkl',
        training_files='/scratch/speech/datasets/IEMOCAP_fru_train.pkl',
#        training_files='/scratch/speech/datasets/Tacotron_LibriTTS/LibriTTS_train_100_1028_subset.txt',
#        training_files='/scratch/speech/datasets/Tacotron_LibriTTS/LibriTTS_train_small.txt',
#        validation_files='/scratch/speech/datasets/Tacotron_LibriTTS/LibriTTS_validation_100.txt',
#        validation_files='/scratch/speech/datasets/IEMOCAP_happy_sad_val.pkl',
        validation_files='/scratch/speech/datasets/IEMOCAP_fru_val.pkl',
#        validation_files='/scratch/speech/datasets/Tacotron_LibriTTS/LibriTTS_validation_small.txt',
        text_cleaners=['english_cleaners'],

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        #sampling_rate=22050,
        sampling_rate=16000,  #24000, #LibriTTS uses sampling rate of 24000
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Latent model parameters
        latent_kernel_size=3,
        latent_n_convolutions=2,
        latent_embedding_dim=512,
        latent_out_dim=16,
        num_lables=4,
        num_of_mixtues=5, 

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        # Loading latent model
        latent_model_checkpoint='/scratch/speech/output/checkpoint_220000',

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=6,  #8,
        mask_padding=True  # set model's padded outputs to padded values
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams

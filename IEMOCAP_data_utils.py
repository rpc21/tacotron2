import pickle
import random
import numpy as np
import torch
import torch.utils.data
import pdb

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence


class IEMOCAPLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text, self.labels = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft_80 = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        self.stft_512 = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            512, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)

    def load_filepaths_and_text(self):
        with open('/scratch/speech/datasets/IEMOCAP.pkl','rb') as f:
            data = pickle.load(f)
        return zip(data['filename'], data['text']), data['label']

    def get_mel_text_label(self, audiopath_and_text, label):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        mel_80 = self.get_mel(audiopath, self.stft_80)
        mel_512 = self.get_mel(audiopath, self.stft_512)
        return text, mel_80, mel_512, label

    def get_mel(self, filename, stft):
        try:
            melspec = torch.from_numpy(np.load(filename[:-4] + '_' + str(stft.n_mel_channels) + '.npy'))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))
        except:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
            with open(filename[:-4] + '_' + str(stft.n_mel_channels) + '.npy', 'wb+') as f:
                np.save(f, melspec.numpy())
        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_label(self.audiopaths_and_text[index], self.labels[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """

    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_80_normalized, mel_512_normalized, label]
        """
        # Right zero-pad all one-hot text sequences to max input length
        #        print(batch)
        #        print('this is what a batch looks like')
        #        pdb.set_trace()
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        labels = [batch[ids_sorted_decreasing[x]][3] for x in range(len(ids_sorted_decreasing))]
        gate_padded, mel_padded, output_lengths = self.prepare_mel_specs(batch, 1, ids_sorted_decreasing)
        gate_padded_512, mel_padded_512, output_lengths_512 = self.prepare_mel_specs(batch, 2, ids_sorted_decreasing)

        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths, mel_padded_512, gate_padded_512, output_lengths_512, labels

    def prepare_mel_specs(self, batch, mel_index, ids_sorted_decreasing):
        # Right zero-pad mel-spec_80
        num_mels = batch[0][mel_index].size(0)
        #        pdb.set_trace()
        max_target_len = max([x[mel_index].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0
        # include mel padded and gate padded
        #        print('This is the max_target_len, look for 1381')
        #        print(max_target_len)
        #        pdb.set_trace()
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][mel_index]
            #            print(mel)
            #            print(mel.shape)
            #            print(mel.size(1))
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1:] = 1
            output_lengths[i] = mel.size(1)
        return gate_padded, mel_padded, output_lengths

import numpy as np
import random
import itertools
import librosa
import IPython.display as ipd


class data_augmentation:
#  noise
    def noise_aug(data,power,rate)
        n = np.random.randn(len(data))
        data_noise = data + power*n
        ipd.Audio(data_noise, rate=rate)
# Shifting
    def shift_aug(data,shift,rate)
        data_roll = np.roll(data, shift)
        ipd.Audio(data_roll, rate=rate)	 
# Speed
    def stretch_aug(data, rate=1):
        return  librosa.effects.time_stretch(data, rate)
# Pitch
    def pitch_aug(data, rate, pitch):
        return librosa.effects.pitch_shift(data, rate, pitch)
	

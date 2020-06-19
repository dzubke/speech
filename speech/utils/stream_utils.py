import numpy as np


def make_full_window(audio_data:np.ndarray, feature_window:int, feature_step:int):
    """
    Takes in a 1d numpy array as input and add appends zeros
    until it is divisible by the feature_step input
    """
    assert audio_data.shape[0] == audio_data.size, "inpute data is not 1-d"
    remainder = (audio_data.shape[0] - feature_window) % feature_step
    num_zeros = feature_step - remainder
    zero_steps = np.zeros((num_zeros, ), dtype=np.float32)
    return np.concatenate((audio_data, zero_steps), axis=0)
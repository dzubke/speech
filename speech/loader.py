from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
import random
import scipy.signal
import torch
import torch.autograd as autograd
import torch.utils.data as tud
import python_speech_features

from speech.utils import wave

class Preprocessor():

    END = "</s>"
    START = "<s>"

    def __init__(self, data_json, max_samples=100, start_and_end=True, use_mfcc=False):
        """
        Builds a preprocessor from a dataset.
        Arguments:
            data_json (string): A file containing a json representation
                of each example per line.
            max_samples (int): The maximum number of examples to be used
                in computing summary statistics.
            start_and_end (bool): Include start and end tokens in labels.
	    use_mfcc (bool): if true, mfcc processing will be used
        """
        data = read_data_json(data_json)
        self.use_mfcc = use_mfcc           #boolean if true, mfcc processing will be used

        # Compute data mean, std from sample
        audio_files = [d['audio'] for d in data]
        random.shuffle(audio_files)
        # the mean and std are of the log of the spectogram of the audio files
        self.mean, self.std = compute_mean_std(audio_files[:max_samples], self.use_mfcc)
        self._input_dim = self.mean.shape[0]

        # Make char map
        chars = list(set(t for d in data for t in d['text']))
        if start_and_end:
            # START must be last so it can easily be
            # excluded in the output classes of a model.
            chars.extend([self.END, self.START])
        self.start_and_end = start_and_end
        self.int_to_char = dict(enumerate(chars))
        self.char_to_int = {v : k for k, v in self.int_to_char.items()}

    def encode(self, text):
        text = list(text)
        if self.start_and_end:
            text = [self.START] + text + [self.END]
        return [self.char_to_int[t] for t in text]

    def decode(self, seq):
        text = [self.int_to_char[s] for s in seq]
        if not self.start_and_end:
            return text

        s = text[0] == self.START
        e = len(text)
        if text[-1] == self.END:
            e = text.index(self.END)
        return text[s:e]

    def preprocess(self, wave_file, text):
        # if use_mfcc is true, use mfcc values
        if self.use_mfcc: 
            inputs = mfcc_from_file(wave_file)
        else: 
            inputs = log_specgram_from_file(wave_file)
            # print(f"log spec size: {inputs.shape}")
        inputs = (inputs - self.mean) / self.std
        targets = self.encode(text)
        return inputs, targets

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def vocab_size(self):
        return len(self.int_to_char)

def compute_mean_std(audio_files, use_mfcc: bool):
    if use_mfcc:        # if use_mfcc true, use mfcc processing
        samples = [mfcc_from_file(af)
               for af in audio_files]
    else:              # else, use log_specgram processing
        samples = [log_specgram_from_file(af)
                for af in audio_files]
    samples = np.vstack(samples)
    #print(f"cms samples shape: {samples.shape}")
    mean = np.mean(samples, axis=0)
    #print(f"cms mean shape: {mean.shape}")
    std = np.std(samples, axis=0)
    #print(f"cms std shape: {std.shape}")
    return mean, std

class AudioDataset(tud.Dataset):

    def __init__(self, data_json, preproc, batch_size):

        data = read_data_json(data_json)
        self.preproc = preproc

        bucket_diff = 4
        max_len = max(len(x['text']) for x in data)
        num_buckets = max_len // bucket_diff
        buckets = [[] for _ in range(num_buckets)]
        for d in data:
            bid = min(len(d['text']) // bucket_diff, num_buckets - 1)
            buckets[bid].append(d)

        # Sort by input length followed by output length
        sort_fn = lambda x : (round(x['duration'], 1),
                              len(x['text']))
        for b in buckets:
            b.sort(key=sort_fn)
        data = [d for b in buckets for d in b]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        datum = self.preproc.preprocess(datum["audio"],
                                        datum["text"])
        return datum


class BatchRandomSampler(tud.sampler.Sampler):
    """
    Batches the data consecutively and randomly samples
    by batch without replacement.
    """

    def __init__(self, data_source, batch_size):
        it_end = len(data_source) - batch_size + 1
        self.batches = [range(i, i + batch_size)
                for i in range(0, it_end, batch_size)]
        self.data_source = data_source

    def __iter__(self):
        random.shuffle(self.batches)
        return (i for b in self.batches for i in b)

    def __len__(self):
        return len(self.data_source)

def make_loader(dataset_json, preproc,
                batch_size, num_workers=4):
    dataset = AudioDataset(dataset_json, preproc,
                           batch_size)
    sampler = BatchRandomSampler(dataset, batch_size)
    loader = tud.DataLoader(dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=lambda batch : zip(*batch),
                drop_last=True)
    return loader

def log_specgram_from_file(audio_file):
    audio, sr = wave.array_from_wave(audio_file)
    return log_specgram(audio, sr)

def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(window_size * sample_rate / 1e3)
    noverlap = int(step_size * sample_rate / 1e3)
    _, _, spec = scipy.signal.spectrogram(audio,
                    fs=sample_rate,
                    window='hann',
                    nperseg=nperseg,
                    noverlap=noverlap,
                    detrend=False)
    return np.log(spec.T.astype(np.float32) + eps)

def mfcc_from_file(audio_file: str):
    """Computes the Mel Frequency Cepstral Coefficients (MFCC) from an audio file path by calling the mfcc method

    Arguments
    ----------
    audio_file: str, the filename of the audio file

    Returns
    -------
        np.ndarray, the transposed log of the spectrogram as returned by mfcc
    """
    audio, sample_rate = wave.array_from_wave(audio_file)
    #print(f"audio_file: {audio_file}")
    #print(f"audio shape: {audio.shape}, sample rate {sample_rate}")

    if len(audio.shape)>1:     # if there are multiple channels, take the first channel
        audio = audio[:,0]
   
    return create_mfcc(audio, sample_rate)

def create_mfcc(audio, sample_rate: int, esp=1e-10):
    """Calculates the mfcc using python_speech_features and can return the mfcc's and its derivatives, if desired. 
    If num_mfcc is set to 13 or less: Output consists of 12 MFCC and 1 energy
    if num_mfcc is set to 26 or less: ouput consists of 12 mfcc, 1 energy, as well as the first derivative of these
    if num_mfcc is set to 39 or less: ouput consists of above as well as the second derivative of these
    """

    num_mfcc = 39   # the number of mfcc's in the output
    mfcc = python_speech_features.mfcc(audio, sample_rate, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, preemph=0.97, appendEnergy=True)
    out = mfcc
    
    # the if-statement waterfall appends the desired number of derivatives to the output value
    if num_mfcc > 13:
        derivative = np.zeros(mfcc.shape)
        for i in range(1, mfcc.shape[0] - 1):
            derivative[i, :] = mfcc[i + 1, :] - mfcc[i - 1, :] 

        mfcc_derivative = np.concatenate((mfcc, derivative), axis=1)
        out = mfcc_derivative
        if num_mfcc > 26:
            derivative2 = np.zeros(derivative.shape)
            for i in range(1, derivative.shape[0] - 1):
                derivative2[i, :] = derivative[i + 1, :] - derivative[i - 1, :]

            out = np.concatenate((mfcc, derivative, derivative2), axis=1)
            if num_mfcc > 39:
                derivative3 = np.zeros(derivative2.shape)
                for i in range(1, derivative2.shape[0] - 1):
                    derivative3[i, :] = derivative2[i + 1, :] - derivative2[i - 1, :]

                out = np.concatenate((mfcc, derivative, derivative2, derivative3), axis=1)

    #print(f"mfcc shape: {out.shape}")
    return out.astype(np.float32)



def read_data_json(data_json):
    with open(data_json) as fid:
        return [json.loads(l) for l in fid]

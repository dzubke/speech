# compatibility libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# standard libraries
import json
import random
# third-party libraries
import matplotlib.pyplot as plt
import numpy as np
import python_speech_features
import scipy.signal
import torch
import torch.autograd as autograd
import torch.utils.data as tud
# project libraries
from speech.utils import wave
from speech.utils.io import read_data_json
from speech.utils.signal_augment import apply_pitch_perturb, speed_vol_perturb, inject_noise
from speech.utils.signal_augment import synthetic_gaussian_noise_inject
from speech.utils.feature_augment import apply_spec_augment



class Preprocessor():

    END = "</s>"
    START = "<s>"

    def __init__(self, data_json, preproc_cfg, logger=None, max_samples=100, start_and_end=True):
        """
        Builds a preprocessor from a dataset.
        Arguments:
            data_json (string): A file containing a json representation
                of each example per line.
            preproc_json: A json file defining the preprocessing with attributes
                preprocessor: "log_spec" or "mfcc" to determine the type of preprocessing
                window_size: the size of the window in the spectrogram transform
                step_size: the size of the step in the spectrogram transform
            max_samples (int): The maximum number of examples to be used
                in computing summary statistics.
            start_and_end (bool): Include start and end tokens in labels.
        """
        data = read_data_json(data_json)

        # Compute data mean, std from sample
        audio_files = [sample['audio'] for sample in data]
        random.shuffle(audio_files)

        # if true, data augmentation will be applied
        self.train_status = True
        
        assert preproc_cfg['preprocessor'] in ['mfcc', 'log_spectrogram'], "preprocessor string not accepted"
        self.preprocessor = preproc_cfg['preprocessor']
        self.window_size = preproc_cfg['window_size']
        self.step_size = preproc_cfg['step_size']
        self.normalize =  preproc_cfg['normalize']

        self.speed_vol_perturb = preproc_cfg['speed_vol_perturb']
        self.tempo_range = preproc_cfg['tempo_range']
        self.gain_range = preproc_cfg['gain_range']
        
        self.pitch_perturb =  preproc_cfg['pitch_perturb']
        self.pitch_range = preproc_cfg['pitch_range']

        self.synthetic_gaussian_noise = preproc_cfg['synthetic_gaussian_noise']
        self.signal_to_noise_range_db=preproc_cfg['signal_to_noise_range_db']

        self.inject_noise = preproc_cfg['inject_noise']
        self.noise_dir = preproc_cfg['noise_directory']
        self.noise_prob = preproc_cfg['noise_prob']
        self.noise_levels = preproc_cfg['noise_levels']       
        
        self.spec_augment = preproc_cfg['use_spec_augment']


        self.mean, self.std = compute_mean_std(audio_files[:max_samples], 
                                                preprocessor = self.preprocessor,
                                                window_size = self.window_size, 
                                                step_size = self.step_size)
        self._input_dim = self.mean.shape[0]
        self.use_log = (logger is not None)
        self.logger = logger


        # Make char map
        chars = list(set(t for d in data for t in d['text']))
        if start_and_end:
            # START must be last so it can easily be
            # excluded in the output classes of a model.
            chars.extend([self.END, self.START])
        self.start_and_end = start_and_end
        self.int_to_char = dict(enumerate(chars))
        self.char_to_int = {v : k for k, v in self.int_to_char.items()}
    
    
    def preprocess(self, wave_file, text):
        if self.use_log: self.logger.info(f"preproc: ======= Entering preprocess =====")
        if self.use_log: self.logger.info(f"preproc: wave_file: {wave_file}")
        if self.use_log: self.logger.info(f"preproc: text: {text}") 

        audio_data, samp_rate = self.signal_augmentations(wave_file)

        # processing method
        preprocessing_function = eval(self.preprocessor + "_from_data")
        feature_data = preprocessing_function(audio_data, samp_rate, self.window_size, self.step_size)
        
        # normalization
        if self.normalize == "batch_normalize":
            feature_data = self.batch_normalize(feature_data)
        elif self.normalize == "sample_normalize":
            feature_data = self.feature_normalize(feature_data)
        else: 
           raise ValueError("preproc config normalize value must be: 'batch_normalize' or 'sample_normalize'")
        if self.use_log: self.logger.info(f"preproc: normalized")
        
        feature_data = self.feature_augmentations(feature_data)

        # target encoding
        targets = self.encode(text)
        if self.use_log: self.logger.info(f"preproc: text encoded")
        if self.use_log: self.logger.info(f"preproc: ======= Exiting preprocess =====")

        return feature_data, targets
    
    def signal_augmentations(self, wave_file:str)-> tuple:
        """
        Performs all of the augmtations to the raw audio signal. The audio data is in pcm16 format.
        Arguments:
            wave_file - str: the path to the audio sample
        Returns:
            audio_data - np.ndarray: augmented np-array
            samp_rate - int: sample rate of the audio recording
        """
        # sox-based tempo, gain, pitch augmentations
        if self.use_log: self.logger.info(f"preproc: audio_data read: {wave_file}")
        if self.speed_vol_perturb and self.train_status:
            audio_data, samp_rate = speed_vol_perturb(wave_file, tempo_range=self.tempo_range, 
                                            gain_range=self.gain_range, logger=self.logger)
        else:
            audio_data, samp_rate = wave.array_from_wave(wave_file)

        # synthetic gaussian noise
        if self.synthetic_gaussian_noise and self.train_status:
            if self.use_log: self.logger.info(f"preproc: synthetic_gaussian_noise_inject")
            audio_data = synthetic_gaussian_noise_inject(audio_data, 
                                self.signal_to_noise_range_db, logger=self.logger)

        # pitch perturb
        if self.pitch_perturb and self.train_status: 
            audio_data = apply_pitch_perturb(audio_data, samp_rate, 
                            pitch_range=self.pitch_range, logger=self.logger)
        
        # noise injection
        if self.inject_noise and self.train_status:
            add_noise = np.random.binomial(1, self.noise_prob)
            if add_noise:
                audio_data =  inject_noise(audio_data, samp_rate, self.noise_dir, 
                                    self.logger, self.noise_levels) 
            if self.use_log: self.logger.info(f"preproc: noise injected")
        
        return audio_data, samp_rate

    def feature_augmentations(self, feature_data:np.ndarray)->np.ndarray:
        """
        Performs feature augmentations to the 2d array of features
        """
        # spec-augment
        if self.spec_augment and self.train_status:
            feature_data = apply_spec_augment(feature_data, self.logger)
            if self.use_log: self.logger.info(f"preproc: spec_aug applied")

        return feature_data


    def batch_normalize(self, np_arr:np.ndarray)->np.ndarray:
        output = (np_arr - self.mean) / self.std
        return output.astype(np.float32)

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

    def update(self):
        """
        updates an instance with new attributes
        """
        if not hasattr(self, 'pitch_perturb'):
            self.pitch_perturb = False
        if not hasattr(self, 'speed_vol_perturb'):
            self.speed_vol_perturb = False
        if not hasattr(self, 'train_status'):
            self.train_status = True
        if not hasattr(self, 'synthetic_gaussian_noise'):
            self.synthetic_gaussian_noise = False
        if not hasattr(self, "signal_to_noise_range_db"):
            self.signal_to_noise_range_db=(100, 100)
        if self.preprocessor == "log_spec":
            self.preprocessor = "log_spectrogram"

    def set_eval(self):
        """
        turns off the data augmentation for evaluation
        """
        self.train_status = False

    def set_train(self):
        """
        turns on data augmentation for training
        """
        self.train_status = True

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def vocab_size(self):
        return len(self.int_to_char)

    def __str__(self):
        try: 
            attribute_names = ["preprocessor", "window_size", "step_size", "SPEC_AUGMENT_STATIC", "spec_augment",
            "INJECT_NOISE_STATIC", "inject_noise", "noise_dir", "noise_prob", "noise_levels", "_input_dim", 
            "start_and_end", "int_to_char", "char_to_int"]
            string="Showing up-to-date attributes"
            for name in attribute_names:
                string += "\n" + name + ": " + str(eval("self."+name))
            return string
        except AttributeError:
            attribute_names = ["_input_dim", "start_and_end", "int_to_char", "char_to_int"]
            string="Showing limited attributes as not all new attributes are supported\n"
            for name in attribute_names:
                string += "\n" + name +": " + str(eval("self."+name))
            return string

    @staticmethod
    def feature_normalize(feature:np.ndarray)->np.ndarray:
        """
        Normalizes the features so that the entire 2d input array
        has zero mean and unit (1) std deviation
        """
        mean = feature.mean()
        std = feature.std()
        feature -= mean
        feature /= std
        assert feature.dtype == np.float32, "feature is not float32"
        return feature


def compute_mean_std(audio_files, preprocessor, window_size, step_size):
    assert preprocessor in ['mfcc', 'log_spectrogram'], "preprocessor string not accepted"
    samples = []
    preprocessing_function  =  eval(preprocessor + "_from_data")
    for audio_file in audio_files: 
        data, samp_rate = wave.array_from_wave(audio_file)
        samples.append(preprocessing_function(data, samp_rate, window_size, step_size))
     
    samples = np.vstack(samples)
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    return mean, std


class AudioDataset(tud.Dataset):

    def __init__(self, data_json, preproc, batch_size):
        """
        this code sorts the samples in data based on the length of the transcript lables and the audio
        sample duration. It does this by creating a number of buckets and sorting the samples
        into different buckets based on the length of the labels. It then sorts the buckets based 
        on the duration of the audio sample.
        """

        data = read_data_json(data_json)        #loads the data_json into a list
        self.preproc = preproc                  # assign the preproc object

        bucket_diff = 4                             # number of different buckets
        max_len = max(len(x['text']) for x in data) # max number of phoneme labels in data
        num_buckets = max_len // bucket_diff        # the number of buckets
        buckets = [[] for _ in range(num_buckets)]  # creating an empy list for the buckets
        for sample in data:                          
            bucket_id = min(len(sample['text']) // bucket_diff, num_buckets - 1)
            buckets[bucket_id].append(sample)

        # Sort by input length followed by output length
        sort_fn = lambda x : (round(x['duration'], 1),
                              len(x['text']))
        for bucket in buckets:
            bucket.sort(key=sort_fn)
        
        # unpack the data in the buckets into a list
        data = [sample for bucket in buckets for sample in bucket]
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
        
        if len(data_source) < batch_size:
            raise ValueError("batch_size is greater than data length")

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


def mfcc_from_data(audio: np.ndarray, samp_rate:int, window_size=20, step_size=10):
    """
    Computes the Mel Frequency Cepstral Coefficients (MFCC) from an audio file path by calling the mfcc method
    Arguments:
        audio - np.ndarray: an array of audio data in pcm16 format
    Returns:
        np.ndarray: the transposed log of the spectrogram as returned by mfcc
    """

    if len(audio.shape)>1:     # there are multiple channels
        if audio.shape[1] == 1:
            audio = audio.squeeze()
        else:
            audio = audio.mean(axis=1)  # multiple channels, average
   
    return create_mfcc(audio, samp_rate, window_size, step_size)

def create_mfcc(audio, sample_rate: int, window_size, step_size, esp=1e-10):
    """Calculates the mfcc using python_speech_features and can return the mfcc's and its derivatives, if desired. 
    If num_mfcc is set to 13 or less: Output consists of 12 MFCC and 1 energy
    if num_mfcc is set to 26 or less: ouput consists of 12 mfcc, 1 energy, as well as the first derivative of these
    if num_mfcc is set to 39 or less: ouput consists of above as well as the second derivative of these
    
    TODO (dustin): this fuction violates DRY principle. Clean it up. 
    """

    num_mfcc = 39   # the number of mfcc's in the output
    mfcc = python_speech_features.mfcc(audio, sample_rate, winlen=window_size/1000, winstep=step_size/1000, numcep=13, nfilt=26, preemph=0.97, appendEnergy=True)
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

    return out.astype(np.float32)


def log_spectrogram_from_file(audio_path:str, window_size=32, step_size=16):
    
    audio_data, samp_rate = wave.array_from_wave(audio_path)
    return log_spectrogram_from_data(audio_data, samp_rate, window_size=window_size, step_size=step_size)

def log_spectrogram_from_data(audio: np.ndarray, samp_rate:int, window_size=32, step_size=16, plot=False):
    """
    Computes the log of the spectrogram from from a input audio file string
    Arguments:
        audio_data (np.ndarray)
    `Returns:
        np.ndarray, the transposed log of the spectrogram as returned by log_specgram
    """
    
    if len(audio.shape)>1:     # there are multiple channels
        if audio.shape[1] == 1:
            audio = audio.squeeze()
        else:
            audio = audio.mean(axis=1)  # multiple channels, average
    return log_spectrogram(audio, samp_rate, window_size, step_size, plot=plot)

def log_spectrogram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10, plot=False):
    nperseg = int(window_size * sample_rate / 1e3)
    noverlap = int(step_size * sample_rate / 1e3)
    f, t, spec = scipy.signal.spectrogram(audio,
                    fs=sample_rate,
                    window='hann',
                    nperseg=nperseg,
                    noverlap=noverlap,
                    detrend=False)
    if plot==True:
        plot_spectrogram(f,t, spec)
    return np.log(spec.T.astype(np.float32) + eps)


def compare_log_spec_from_file(audio_file_1: str, audio_file_2: str, plot=False):
    """
    This function takes in two audio paths and calculates the difference between the spectrograms 
        by subtracting them. 
    """
    audio_1, sr_1 = wave.array_from_wave(audio_file_1)
    audio_2, sr_2 = wave.array_from_wave(audio_file_2)

    if len(audio_1.shape)>1:
        audio_1 = audio_1[:,0]  # take the first channel
    if len(audio_2.shape)>1:
        audio_2 = audio_2[:,0]  # take the first channel
    
    window_size = 20
    step_size = 10

    nperseg_1 = int(window_size * sr_1 / 1e3)
    noverlap_1 = int(step_size * sr_1 / 1e3)
    nperseg_2 = int(window_size * sr_2 / 1e3)
    noverlap_2 = int(step_size * sr_2 / 1e3)

    freq_1, time_1, spec_1 = scipy.signal.spectrogram(audio_1,
                    fs=sr_1,
                    window='hann',
                    nperseg=nperseg_1,
                    noverlap=noverlap_1,
                    detrend=False)

    freq_2, time_2, spec_2 = scipy.signal.spectrogram(audio_2,
                    fs=sr_2,
                    window='hann',
                    nperseg=nperseg_2,
                    noverlap=noverlap_2,
                    detrend=False)
    
    spec_diff = spec_1 - spec_2 
    freq_diff = freq_1 - freq_2
    time_diff = time_1 - time_2

    if plot:
        plot_spectrogram(freq_diff, time_diff, spec_diff)
        #plot_spectrogram(freq_1, time_1, spec_2)
        #plot_spectrogram(freq_2, time_2, spec_2)
    
    return spec_diff


def plot_spectrogram(f, t, Sxx):
    """This function plots a spectrogram using matplotlib

    Arguments
    ----------
    f: the frequency output of the scipy.signal.spectrogram
    t: the time series output of the scipy.signal.spectrogram
    Sxx: the spectrogram output of scipy.signal.spectrogram

    Returns
    --------
    None

    Note: the function scipy.signal.spectrogram returns f, t, Sxx in that order
    """
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()



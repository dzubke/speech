from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# standard libraries
import json
import numpy as np
import random

# third-party libraries
import scipy.signal
import torch
import torch.autograd as autograd
import torch.utils.data as tud
import matplotlib.pyplot as plt
import python_speech_features

# project libraries
from speech.utils import wave, spec_augment
from speech.utils.noise_injector import inject_noise
from speech.utils.speed_vol_perturb import speed_vol_perturb



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

        Note:if using mfcc processing is desired, change method log_specgram_from_file to mfcc_from_file

        """
        data = read_data_json(data_json)

        # Compute data mean, std from sample
        audio_files = [d['audio'] for d in data]
        random.shuffle(audio_files)


        self.preprocessor = preproc_cfg['preprocessor']
        self.window_size = preproc_cfg['window_size']
        self.step_size = preproc_cfg['step_size']
        self.SPEC_AUGMENT_STATIC = preproc_cfg['use_spec_augment']
        self.spec_augment = preproc_cfg['use_spec_augment']
        self.INJECT_NOISE_STATIC = preproc_cfg['inject_noise']
        self.inject_noise = preproc_cfg['inject_noise']
        self.noise_dir = preproc_cfg['noise_directory']
        self.noise_prob = preproc_cfg['noise_prob']
        self.noise_levels = preproc_cfg['noise_levels']
        self.speed_vol_perturb = preproc_cfg['speed_vol_perturb']
        self.tempo_range = preproc_cfg['tempo_range']
        self.gain_range = preproc_cfg['gain_range']

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
        
        if self.speed_vol_perturb:
            audio_data, samp_rate = speed_vol_perturb(wave_file, tempo_range=self.tempo_range)
        else:
            audio_data, samp_rate = wave.array_from_wave(wave_file)
        if self.use_log: self.logger.info(f"preproc: audio_data read: {wave_file}")
        
        if self.inject_noise:
            add_noise = np.random.binomial(1, self.noise_prob)
            if add_noise:
                audio_data =  inject_noise(audio_data, samp_rate, self.noise_dir, self.logger, self.noise_levels) 
            if self.use_log: self.logger.info(f"preproc: noise injected")

        if self.preprocessor == "log_spec":
            inputs = log_specgram_from_data(audio_data, samp_rate, self.window_size, self.step_size)
            if self.use_log: self.logger.info(f"preproc: log_spec calculated")
        elif self.preprocessor == "mfcc":
           inputs = mfcc_from_data(audio_data, samp_rate, self.window_size, self.step_size)
        else: 
           raise ValueError("preprocessing config preprocessor value must be 'log_spec' or 'mfcc'")
        
        inputs = (inputs - self.mean) / self.std
        if self.use_log: self.logger.info(f"preproc: normalized")

        if self.spec_augment:
            inputs = apply_spec_augment(inputs, self.logger)
            if self.use_log: self.logger.info(f"preproc: spec_aug applied")

        targets = self.encode(text)
        if self.use_log: self.logger.info(f"preproc: text encoded")

        return inputs, targets


    def set_eval(self):
        """
            turns off the data augmentation for evaluation
        """
        if self.SPEC_AUGMENT_STATIC:
            self.spec_augment = False
        if self.INJECT_NOISE_STATIC:
            self.inject_noise = False


    def set_train(self):
        """
            turns on data augmentation for training
        """
        if self.SPEC_AUGMENT_STATIC:
            self.spec_augment = True
        if self.INJECT_NOISE_STATIC:
            self.inject_noise = True


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
                string += name +": " + str(eval("self."+name))+"\n"
            return string
        except AttributeError:
            attribute_names = ["_input_dim", "start_and_end", "int_to_char", "char_to_int"]
            string="Showing limited attributes as not all new attributes are supported\n"
            for name in attribute_names:
                string += name +": " + str(eval("self."+name))+"\n"
            return string

def compute_mean_std(audio_files, preprocessor, window_size, step_size):
    samples = []
    if preprocessor == "log_spec":
        for audio_file in audio_files: 
            data, samp_rate = wave.array_from_wave(audio_file)
            samples.append(log_specgram_from_data(data, samp_rate, window_size, step_size))
                    
    elif preprocessor == "mfcc":
        for audio_file in audio_files: 
            data, samp_rate = wave.array_from_wave(audio_file)
            samples.append(mfcc_from_data(data, samp_rate, window_size, step_size))
    else: 
        raise ValueError("preprocessing config preprocessor value must be 'log_spec' or 'mfcc'")
     
    samples = np.vstack(samples)
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    return mean, std

class AudioDataset(tud.Dataset):

    def __init__(self, data_json, preproc, batch_size):

        data = read_data_json(data_json)        #loads the data_json into a list
        self.preproc = preproc                  # assign the preproc object

        bucket_diff = 4                         # number of different buckets
        max_len = max(len(x['text']) for x in data) # max number of phoneme labels in data
        num_buckets = max_len // bucket_diff        # the number of buckets
        buckets = [[] for _ in range(num_buckets)]  # creating an empy list for the buckets
        for d in data:                          
            bid = min(len(d['text']) // bucket_diff, num_buckets - 1)
            buckets[bid].append(d)

        # Sort by input length followed by output length
        sort_fn = lambda x : (round(x['duration'], 1),
                              len(x['text']))
        for b in buckets:
            b.sort(key=sort_fn)
        
        # unpack the data in the buckets into a list
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
    """Computes the Mel Frequency Cepstral Coefficients (MFCC) from an audio file path by calling the mfcc method

    Arguments
    ----------
    audio_file: str, the filename of the audio file

    Returns
    -------
        np.ndarray, the transposed log of the spectrogram as returned by mfcc
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


def log_specgram_from_file(audio_path:str, window_size=32, step_size=16):
    
    audio_data, samp_rate = wave.array_from_wave(audio_path)
    return log_specgram_from_data(audio_data, samp_rate, window_size=window_size, step_size=step_size)

def log_specgram_from_data(audio: np.ndarray, samp_rate:int, window_size=32, step_size=16, plot=False):
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
    return log_specgram(audio, samp_rate, window_size, step_size, plot=plot)

def log_specgram(audio, sample_rate, window_size=20,
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
    """This function takes in two audio paths and calculates the difference between the spectrograms 
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

def read_data_json(data_json):
    with open(data_json) as fid:
        return [json.loads(l) for l in fid]


def apply_spec_augment(inputs, logger):
    """calls the spec_augment function on the normalized log_spec. A policy defined 
        in the policy_dict will be chosen uniformly at random.
    Arguments:
        inputs (np.ndarray): normalized log_spec with dimensional order time x freq
    Returns:
        inputs (nd.ndarray): the modified log_spec array with order time x freq
    """

    use_log = (logger is not None)
    assert type(inputs) == np.ndarray, "input is not numpy array"

    policy_dict = {
        0: {'time_warping_para':0, 'frequency_masking_para':0,
            'time_masking_para':0, 'frequency_mask_num':0, 'time_mask_num':0}, 
        1: {"time_warping_para":20, "frequency_masking_para":60,
            "time_masking_para":60, "frequency_mask_num":1, "time_mask_num":1},
        2: {"time_warping_para":20, "frequency_masking_para":30,
            "time_masking_para":30, "frequency_mask_num":2, "time_mask_num":2},
        3: {"time_warping_para":20, "frequency_masking_para":20,
            "time_masking_para":20, "frequency_mask_num":3, "time_mask_num":3},
            }
    
    policy_choice = np.random.randint(low=0, high=4)
    if use_log: logger.info(f"app spec_aug: policy: {policy_choice}")

    policy = policy_dict.get(policy_choice)

    # the inputs need to be transposed and converted to torch tensor
    # as spec_augment method expects tensor with freq x time dimensions
    if use_log: logger.info(f"app s_a: input shape: {inputs.shape}")

    inputs = torch.from_numpy(inputs.T)

    inputs = spec_augment.spec_augment(inputs, 
                    time_warping_para=policy.get('time_warping_para'), 
                    frequency_masking_para=policy.get('frequency_masking_para'),
                    time_masking_para=policy.get('time_masking_para'),
                    frequency_mask_num=policy.get('frequency_mask_num'), 
                    time_mask_num=policy.get('time_mask_num'), logger=logger)
    
    # convert the torch tensor back to numpy array and transpose back to time x freq
    inputs = inputs.detach().cpu().numpy() if inputs.requires_grad else inputs.cpu().numpy()
    inputs = inputs.T
    assert type(inputs) == np.ndarray, "output is not numpy array"

    return inputs

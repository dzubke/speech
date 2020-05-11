# standard libraries
import time, logging
from datetime import datetime
import threading, collections, queue, os, os.path
# third-party libraries
import numpy as np
import pyaudio
import wave
import webrtcvad
from halo import Halo
from scipy import signal
import matplotlib.pyplot as plt
import torch
# project libraries
import speech
from speech.utils.convert import to_numpy
import speech.models as models
from speech.loader import log_specgram_from_data
from speech.models.ctc_decoder import decode as ctc_decode
from speech.utils import compat 


logging.basicConfig(level=20)

class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 62.5

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS, file=None):
        def proxy_callback(in_data, frame_count, time_info, status):
            #pylint: disable=unused-argument
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            callback(in_data)
            return (None, pyaudio.paContinue)
        if callback is None: callback = lambda in_data: self.buffer_queue.put(in_data)
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
        print(f"block_size input {self.block_size_input}")
        self.pa = pyaudio.PyAudio()

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }

        self.chunk = None
        # if not default device
        if self.device:
            kwargs['input_device_index'] = self.device
        elif file is not None:
            self.chunk = 256
            self.wf = wave.open(file, 'rb')

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def resample(self, data, input_rate):
        """
        Microphone may not support our native processing sampling rate, so
        resample from input_rate to RATE_PROCESS here for webrtcvad and
        deepspeech
        Args:
            data (binary): Input audio stream
            input_rate (int): Input audio rate to resample from
        """
        data16 = np.fromstring(string=data, dtype=np.int16)
        resample_size = int(len(data16) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        return resample16.tostring()

    def read_resampled(self):
        """Return a block of audio data resampled to 16000hz, blocking if necessary."""
        return self.resample(data=self.buffer_queue.get(),
                             input_rate=self.input_rate)

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)

    def write_wav(self, filename, data):
        logging.info("write wav %s", filename)
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
            assert self.FORMAT == pyaudio.paInt16
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(data)


class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, aggressiveness=3, device=None, input_rate=None, file=None):
        super().__init__(device=device, input_rate=input_rate, file=file)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            while True:
                yield self.read_resampled()

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        if frames is None: frames = self.frame_generator()
        for frame in frames:
            yield frame


def max_decode(output, blank=39):
    pred = np.argmax(output, 1)
    prev = pred[0]
    seq = [prev] if prev != blank else []
    for p in pred[1:]:
        if p != blank and p != prev:
            seq.append(p)
        prev = p
    return seq

def main(ARGS):

    print('Initializing model...')
    model, preproc = speech.load(ARGS.model, tag='best')
    model.eval()
    # Start audio with VAD
    vad_audio = VADAudio(aggressiveness=ARGS.vad_aggressiveness,
                         device=ARGS.device,
                         input_rate=ARGS.rate,
                         file=ARGS.file)
    print("Listening (ctrl-C to exit)...")
    frames = vad_audio.vad_collector()

    # Stream from microphone to DeepSpeech using VAD
    spinner = None
    if not ARGS.nospinner:
        spinner = Halo(spinner='line')

    wav_data           = bytearray()
    audio_buffer_size   = 2   # 2 steps in the log_spec window
    conv_buffer_size    = 31 # num of log_spec timesteps to feed into model
    audio_ring_buffer   = collections.deque(maxlen=audio_buffer_size)
    conv_ring_buffer    = collections.deque(maxlen=conv_buffer_size)
    predictions         = list()
    probs_list          = list()
    frames_per_block    = round(
        vad_audio.RATE_PROCESS / vad_audio.BLOCKS_PER_SECOND * 2
    ) 
    # initialize the hidden and cells states of the LSTM layers
    hidden_in           = torch.zeros((5, 1, 512), dtype=torch.float32)
    cell_in             = torch.zeros((5, 1, 512), dtype=torch.float32)

    # ------------ logging ---------------
    logging.info(ARGS)
    logging.info(model)
    logging.info(preproc)
    # ------------ logging ---------------

    try:
        for count, frame in enumerate(frames):
            if spinner: spinner.start()
            # exit the loop if there are no more full input frames
            if len(frame) <  frames_per_block:
                break

            # ------------ logging ---------------
            logging.debug(f"frame length: {len(frame)}")
            logging.debug(f"audio_buffer length: {len(audio_ring_buffer)}")
            # ------------ logging ---------------

            # fill up the audio_ring_buffer and then feed into the model
            if len(audio_ring_buffer) < audio_buffer_size-1:
                audio_ring_buffer.append(frame)
            else: 
                audio_ring_buffer.append(frame)
                buffer_list = list(audio_ring_buffer)
                numpy_buffer = np.concatenate(
                    (np.frombuffer(buffer_list[0], np.int16), 
                    np.frombuffer(buffer_list[1], np.int16)))
                log_spec_step = log_specgram_from_data(numpy_buffer, samp_rate=16000)
                # normalize the log_spec, older preproc objects do not have "normalize" method
                if hasattr(preproc, "normalize"):
                    norm_log_spec = preproc.normalize(log_spec_step)
                else: 
                    norm_log_spec = compat.normalize(preproc, log_spec_step)

                # ------------ logging ---------------
                logging.debug(f"numpy_buffer shape: {numpy_buffer.shape}")
                logging.debug(f"log_spec_step shape: {log_spec_step.shape}")
                # ------------ logging ---------------

                # fill up the conv_ring_buffer and then feed into the model
                logging.debug(f"conv_buffer length: {len(conv_ring_buffer)}")
                if len(conv_ring_buffer) < conv_buffer_size-1:
                    conv_ring_buffer.append(norm_log_spec)
                else: 
                    conv_ring_buffer.append(norm_log_spec)
                    # log_spec are freq (257) x time (11)
                    conv_context = np.concatenate(list(conv_ring_buffer), axis=0)
                    # addding batch dimension
                    conv_context = np.expand_dims(conv_context, axis=0)
                    model_out = model(torch.from_numpy(conv_context), (hidden_in, cell_in))
                    probs, (hidden_out, cell_out) = model_out
                    probs = to_numpy(probs)
                    probs_list.append(probs)
                    hidden_in, cell_in = hidden_out, cell_out
                    
                    # ------------ logging ---------------
                    logging.debug(f"conv_context shape: {conv_context.shape}")
                    logging.debug(f"probs shape: {probs.shape}")
                    logging.debug(f"probs_list len: {len(probs_list)}")
                    #logging.debug(f"probs value: {probs}")
                    # ------------ logging ---------------
                
                    # decoder_context = 1
                    # count_offset = 30 + decoder_context   # num of steps decoder is behind loop counnt
                    # if len(probs_list) >= decoder_context:
                    #     probs_start = count - count_offset
                    #     probs_end = count - count_offset + decoder_context
                    #     probs_steps = np.concatenate(probs_list[probs_start:probs_end], axis=1)
                    #     int_labels = max_decode(probs_steps[0], blank=39)
                    #     # int_labels, likelihood = ctc_decode(probs[0], beam_size=50, blank=39)
                    #     predictions = preproc.decode(int_labels)


                    if count%20 ==0 and count!=0:
                        probs_steps = np.concatenate(probs_list, axis=1)
                        int_labels = max_decode(probs_steps[0], blank=39)
                        # int_labels, likelihood = ctc_decode(probs[0], beam_size=50, blank=39)
                        predictions = preproc.decode(int_labels)

                        # ------------ logging ---------------
                        logging.info(f"predictions: {predictions}")
                        # ------------ logging ---------------

            if ARGS.savewav: wav_data.extend(frame)

    except KeyboardInterrupt:
        pass
    finally: 
        vad_audio.destroy()
        if ARGS.savewav:
            vad_audio.write_wav(os.path.join(ARGS.savewav, datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav")), wav_data)
            all_audio = np.frombuffer(wav_data, np.int16)
            plt.plot(all_audio)
            #plt.figure()
            plt.show()


if __name__ == '__main__':
    BEAM_WIDTH = 500
    DEFAULT_SAMPLE_RATE = 16000
    LM_ALPHA = 0.75
    LM_BETA = 1.85

    import argparse
    parser = argparse.ArgumentParser(description="Stream from microphone to DeepSpeech using VAD")

    parser.add_argument('-v', '--vad_aggressiveness', type=int, default=3,
                        help="Set aggressiveness of VAD: an integer between 0 and 3, 0 being the least aggressive about filtering out non-speech, 3 the most aggressive. Default: 3")
    parser.add_argument('--nospinner', action='store_true',
                        help="Disable spinner")
    parser.add_argument('-w', '--savewav',
                        help="Save .wav files of utterences to given directory")
    parser.add_argument('-f', '--file',
                        help="Read from .wav file instead of microphone")
    parser.add_argument('-m', '--model',
                        help="Path to the model (protocol buffer binary file, or entire directory containing all standard-named files for model)")
    parser.add_argument('-d', '--device', type=int, default=None,
                        help="Device input index (Int) as listed by pyaudio.PyAudio.get_device_info_by_index(). If not provided, falls back to PyAudio.get_default_device().")
    parser.add_argument('-r', '--rate', type=int, default=DEFAULT_SAMPLE_RATE,
                        help=f"Input device sample rate. Default: {DEFAULT_SAMPLE_RATE}. Your device may require 44100.")
    parser.add_argument('-bw', '--beam_width', type=int, default=BEAM_WIDTH,
                        help=f"Beam width used in the CTC decoder when building candidate transcriptions. Default: {BEAM_WIDTH}")

    ARGS = parser.parse_args()
    if ARGS.savewav: os.makedirs(ARGS.savewav, exist_ok=True)
    main(ARGS)
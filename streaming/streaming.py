# standard libraries
import time, logging
from datetime import datetime
import threading, collections, queue, os, os.path, json
# third-party libraries
import numpy as np
import pyaudio
import wave
from scipy import signal
import matplotlib.pyplot as plt
import torch
# project libraries
import speech
from speech.utils.convert import to_numpy
from speech.models.ctc_model_pyt14 import CTC_pyt14
from speech.loader import log_specgram_from_data, log_specgram_from_file
from speech.models.ctc_decoder import decode as ctc_decode
from speech.utils import compat

logging.basicConfig(level=20)


def main(ARGS):

    print('Initializing model...')
    state_dict_model, preproc = speech.load(ARGS.model, tag='best')
    
    with open(ARGS.config, 'r') as fid:
        config = json.load(fid)
        model_config = config["model"]

    model = CTC_pyt14(preproc.input_dim, preproc.vocab_size, model_config)
    state_dict = state_dict_model.state_dict()
    model.load_state_dict(state_dict)
    model.eval()

    #initial states for LSTM layers
    hidden_in = torch.zeros((5, 1, 512), dtype=torch.float32)
    cell_in   = torch.zeros((5, 1, 512), dtype=torch.float32)
    lstm_states = (hidden_in, cell_in)


    stream_infer(model, preproc, lstm_states, ARGS)

    list_chunk_infer(model, preproc, lstm_states, ARGS)

    fullaudio_infer(model, preproc, lstm_states, ARGS)



def stream_infer(model, preproc, lstm_states, ARGS):
    """
    Performs streaming inference of an input wav file (if provided in ARGS) or from
    the micropohone. Inference is performed my model and the preproc preprocessing
    object performs normalization.
    """
    begin_time = time.time()

    # Start audio with VAD
    audio = Audio(device=ARGS.device, input_rate=ARGS.rate, file=ARGS.file)
    frames = audio.frame_generator()

    print("Listening (ctrl-C to exit)...")
    logging.info(f"--- starting stream_infer  ---")

    hidden_in, cell_in = lstm_states
    wav_data = bytearray()
    audio_buffer_size = 2   # 2 16 ms steps in the log_spec window
    log_spec_buffer_size = 31
    audio_ring_buffer = collections.deque(maxlen=audio_buffer_size)
    log_spec_ring_buffer = collections.deque(maxlen=log_spec_buffer_size)
    predictions = list()
    probs_list  = list()
    frames_per_block = round( audio.RATE_PROCESS/ audio.BLOCKS_PER_SECOND * 2) 

    # -------time evaluation variables-----------
    audio_buffer_time, audio_buffer_count = 0.0, 0 
    numpy_buffer_time, numpy_buffer_count = 0.0, 0 
    log_spec_time, log_spec_count = 0.0, 0
    normalize_time, normalize_count = 0.0, 0 
    log_spec_buffer_time, log_spec_buffer_count = 0.0, 0
    numpy_conv_time, numpy_conv_count = 0.0, 0
    model_infer_time, model_infer_count = 0.0, 0 
    output_assign_time, output_assign_count = 0.0, 0
    decoder_time, decoder_count = 0.0, 0
    total_time, total_count = 0.0, 0 
    # -------------------------------------------

    # ------------ logging ----------------------
    logging.debug(ARGS)
    logging.debug(model)
    logging.debug(preproc)
    # -------------------------------------------

    try:
        total_time_start = time.time()
        for count, frame in enumerate(frames):
            # exit the loop if there are no more full input frames
            if len(frame) <  frames_per_block:
                break

            # ------------ logging ---------------
            logging.debug(f"frame length: {len(frame)}")
            logging.debug(f"audio_buffer length: {len(audio_ring_buffer)}")
            # ------------ logging ---------------

            # fill up the audio_ring_buffer and then feed into the model
            if len(audio_ring_buffer) < audio_buffer_size-1:
                # note: appending new frame to right of the buffer
                audio_buffer_time_start = time.time()
                audio_ring_buffer.append(frame)
                audio_buffer_time += time.time() - audio_buffer_time_start
                audio_buffer_count += 1
            else: 
                audio_buffer_time_start = time.time()
                audio_ring_buffer.append(frame)
                audio_buffer_time += time.time() - audio_buffer_time_start
                audio_buffer_count += 1
                
                numpy_buffer_time_start = time.time()
                buffer_list = list(audio_ring_buffer)
                # convert the buffer to numpy array
                # a single frame has dims: (512,) and numpy buffer (2 frames) is: (512,)
                # The dimension of numpy buffer is reduced by half because each 
                # integer in numpy buffer is encoded as 2 hexidecimal entries in frame
                numpy_buffer = np.concatenate(
                    (np.frombuffer(buffer_list[0], np.int16), 
                    np.frombuffer(buffer_list[1], np.int16)))
                # calculate the log_spec with dim: (1, 257)
                numpy_buffer_time += time.time() - numpy_buffer_time_start
                numpy_buffer_count += 1

                log_spec_time_start = time.time()
                log_spec_step = log_specgram_from_data(numpy_buffer, samp_rate=16000)
                log_spec_time += time.time() - log_spec_time_start
                log_spec_count += 1
                
                normalize_time_start = time.time()
                # normalize the log_spec_step, older preproc objects do not have "normalize" method
                if hasattr(preproc, "normalize"):
                    norm_log_spec = preproc.normalize(log_spec_step)
                else: 
                    norm_log_spec = compat.normalize(preproc, log_spec_step)
                normalize_time += time.time() - normalize_time_start
                normalize_count += 1

                # ------------ logging ---------------
                logging.debug(f"numpy_buffer shape: {numpy_buffer.shape}")
                logging.debug(f"log_spec_step shape: {log_spec_step.shape}")
                logging.debug(f"log_spec_buffer length: {len(log_spec_ring_buffer)}")
                # ------------ logging ---------------

                # fill up the log_spec_ring_buffer and then feed into the model
                if len(log_spec_ring_buffer) < log_spec_buffer_size-1:
                    log_spec_buffer_time_start = time.time()
                    log_spec_ring_buffer.append(norm_log_spec)
                    log_spec_buffer_time += time.time() - log_spec_buffer_time_start
                    log_spec_buffer_count += 1
                else: 
                    log_spec_buffer_time_start = time.time()
                    log_spec_ring_buffer.append(norm_log_spec)
                    log_spec_buffer_time += time.time() - log_spec_buffer_time_start
                    log_spec_buffer_count += 1

                    numpy_conv_time_start = time.time()
                    # conv_context dim: (31, 257)
                    conv_context = np.concatenate(list(log_spec_ring_buffer), axis=0)
                    # addding batch dimension: (1, 31, 257)
                    conv_context = np.expand_dims(conv_context, axis=0)
                    numpy_conv_time += time.time() - numpy_conv_time_start
                    numpy_conv_count += 1

                    model_infer_time_start = time.time()
                    model_out = model(torch.from_numpy(conv_context), (hidden_in, cell_in))
                    model_infer_time += time.time() - model_infer_time_start
                    model_infer_count += 1

                    output_assign_time_start = time.time()
                    probs, (hidden_out, cell_out) = model_out
                    # probs dim: (1, 1, 40)
                    probs = to_numpy(probs)
                    probs_list.append(probs)
                    hidden_in, cell_in = hidden_out, cell_out
                    output_assign_time += time.time() - output_assign_time_start
                    output_assign_count += 1

                    
                    # ------------ logging ---------------
                    logging.debug(f"conv_context shape: {conv_context.shape}")
                    logging.debug(f"probs shape: {probs.shape}")
                    logging.debug(f"probs_list len: {len(probs_list)}")
                    #logging.debug(f"probs value: {probs}")
                    # ------------ logging ---------------
            
                    # decoding every 20 time-steps
                    if count%20 ==0 and count!=0:
                        decoder_time_start = time.time()
                        probs_steps = np.concatenate(probs_list, axis=1)
                        int_labels = max_decode(probs_steps[0], blank=39)
                        # int_labels, likelihood = ctc_decode(probs[0], beam_size=50, blank=39)
                        predictions = preproc.decode(int_labels)
                        decoder_time += time.time() - decoder_time_start
                        decoder_count += 1
                        
                        # ------------ logging ---------------
                        logging.info(f"predictions: {predictions}")
                        # ------------ logging ---------------
                    
                    total_count += 1
   
            if ARGS.savewav: wav_data.extend(frame)
        
        logging.debug(f"final predictions: {predictions}")

    except KeyboardInterrupt:
        pass
    finally: 
        audio.destroy()
        total_time = time.time() - total_time_start
        acc = 3

        logging.info(f"audio_buffer        time (s), count: {round(audio_buffer_time, acc)}, {audio_buffer_count}")
        logging.info(f"numpy_buffer        time (s), count: {round(numpy_buffer_time, acc)}, {numpy_buffer_count}")
        logging.info(f"log_spec_operation  time (s), count: {round(log_spec_time, acc)}, {log_spec_count}")
        logging.info(f"normalize           time (s), count: {round(normalize_time, acc)}, {normalize_count}")
        logging.info(f"log_spec_buffer     time (s), count: {round(log_spec_buffer_time, acc)}, {log_spec_buffer_count}")
        logging.info(f"numpy_conv          time (s), count: {round(numpy_conv_time, acc)}, {numpy_conv_count}")
        logging.info(f"model_infer         time (s), count: {round(model_infer_time, acc)}, {model_infer_count}")
        logging.info(f"output_assign       time (s), count: {round(output_assign_time, acc)}, {output_assign_count}")
        logging.info(f"decoder             time (s), count: {round(decoder_time, acc)}, {decoder_count}")
        logging.info(f"total               time (s), count: {round(total_time, acc)}, {total_count}")
        #logging.info(f"total 2             time (s), count: {round((time.time()-begin_time), acc)}, ")

        if ARGS.savewav:
            audio.write_wav(os.path.join(ARGS.savewav, datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav")), wav_data)
            all_audio = np.frombuffer(wav_data, np.int16)
            plt.plot(all_audio)
            plt.show()


def list_chunk_infer(model, preproc, lstm_states, ARGS):
    

    
    if ARGS.file is None:
        logging.info(f"--- Skipping list_chunk_infer. No input file ---")
    else:
        
        #lc means listchunk
        lc_model_infer_time, lc_model_infer_count = 0.0, 0 
        lc_output_assign_time, lc_output_assign_count = 0.0, 0
        lc_decode_time, lc_decode_count = 0.0, 0
        lc_total_time, lc_total_count = 0.0, 0 

        lc_total_time = time.time()

        hidden_in, cell_in = lstm_states
        probs_list = list()

        log_spec = log_specgram_from_file(ARGS.file)
        if hasattr(preproc, "normalize"):
            norm_log_spec = preproc.normalize(log_spec)
        else: 
            norm_log_spec = compat.normalize(preproc, log_spec)
        norm_log_spec = np.expand_dims(norm_log_spec, axis=0)
        torch_input = torch.from_numpy(norm_log_spec)
        padding = (0, 0, 15, 15)
        padded_input = torch.nn.functional.pad(torch_input, padding, value=0)

        context_size = 31
        iterations = torch_input.shape[1] - (context_size - 1)

        # ------------ logging ---------------
        logging.info(f"--------- starting list_chunck_infer ----------")
        logging.debug(f"log_spec shape: {log_spec.shape}")
        logging.debug(f"norm_log_spec with batch shape: {norm_log_spec.shape}")
        logging.debug(f"torch_input shape: {torch_input.shape}")
        logging.debug(f"padded_input shape: {padded_input.shape}")
        # ------------ logging ---------------


        for i in range(iterations): 
            input_chunk = torch_input[:, i:i+context_size, :]
            
            lc_model_infer_time_start = time.time()
            model_output = model(input_chunk, (hidden_in, cell_in))
            lc_model_infer_time += time.time() - lc_model_infer_time_start
            lc_model_infer_count += 1

            lc_output_assign_time_start = time.time()
            probs, (hidden_out, cell_out) = model_output
            hidden_in, cell_in = hidden_out, cell_out
            probs = to_numpy(probs)
            probs_list.append(probs)
            lc_output_assign_time += time.time() - lc_output_assign_time_start
            lc_output_assign_count += 1
            
            # decoding every 20 time-steps
            if i%20 ==0 and i !=0:
                lc_decode_time_start = time.time()
                probs_steps = np.concatenate(probs_list, axis=1)
                int_labels = max_decode(probs_steps[0], blank=39)
                # int_labels, likelihood = ctc_decode(probs[0], beam_size=50, blank=39)
                predictions = preproc.decode(int_labels)
                lc_decode_time += time.time() - lc_decode_time_start
                lc_decode_count += 1
                #logging.debug(f"intermediate predictions: {predictions}")
            
            lc_total_count += 1

            # ------------ logging ---------------
            logging.debug(f"input_chunk shape: {input_chunk.shape}")
            logging.debug(f"probs shape: {probs.shape}")
            logging.debug(f"probs list len: {len(probs_list)}")
            # ------------ logging ---------------
        lc_total_time = time.time() - lc_total_time

        # ------------ logging ---------------
        logging.info(f"final predictions: {predictions}")
        acc = 3
        logging.info(f"model infer          time (s), count: {round(lc_model_infer_time, acc)}, {lc_model_infer_count}")
        logging.info(f"output assign        time (s), count: {round(lc_output_assign_time, acc)}, {lc_output_assign_count}")
        logging.info(f"decoder              time (s), count: {round(lc_decode_time, acc)}, {lc_decode_count}")
        logging.info(f"total                time (s), count: {round(lc_total_time, acc)}, {lc_total_count}")


def fullaudio_infer(model, preproc, lstm_states, ARGS):
    """
    conducts inference from an entire audio file. If no audio file
    is provided in ARGS when recording from mic, this function is exited.
    """

    if ARGS.file is None:
        logging.info(f"--- Skipping fullaudio_infer. No input file ---")
    else:
        # fa means fullaudio
        fa_total_time = 0.0
        fa_log_spec_time = 0.0
        fa_normalize_time = 0.0
        fa_convert_pad_time = 0.0
        fa_model_infer_time = 0.0
        fa_decode_time = 0.0

        hidden_in, cell_in = lstm_states

        fa_total_time = time.time()

        fa_log_spec_time = time.time()
        log_spec = log_specgram_from_file(ARGS.file)
        fa_log_spec_time = time.time() - fa_log_spec_time
        
        fa_normalize_time = time.time()
        if hasattr(preproc, "normalize"):
            norm_log_spec = preproc.normalize(log_spec)
        else: 
            norm_log_spec = compat.normalize(preproc, log_spec)
        fa_normalize_time = time.time() - fa_normalize_time

        fa_convert_pad_time = time.time()
        # adds the batch dimension (1, time, 257)
        norm_log_spec = np.expand_dims(norm_log_spec, axis=0)
        torch_input = torch.from_numpy(norm_log_spec)
        # paddings starts from the back, zero padding to freq, 15 paddding to time
        padding = (0, 0, 15, 15)
        padded_input = torch.nn.functional.pad(torch_input, padding, value=0)
        fa_convert_pad_time = time.time() - fa_convert_pad_time
        # !!! curently not padding !!!!
        fa_model_infer_time = time.time()
        model_output = model(torch_input, (hidden_in, cell_in))
        fa_model_infer_time = time.time() - fa_model_infer_time

        probs, (hidden_out, cell_out) = model_output
        probs = to_numpy(probs)
        fa_decode_time = time.time()
        int_labels = max_decode(probs[0], blank=39)
        fa_decode_time = time.time() - fa_decode_time
        # int_labels, likelihood = ctc_decode(probs[0], beam_size=50, blank=39)
        predictions = preproc.decode(int_labels)
        
        fa_total_time = time.time() - fa_total_time
        

        # ------------ logging ---------------
        logging.info(f"--------- starting fullaudio_infer ----------")
        logging.debug(f"log_spec shape: {log_spec.shape}")
        logging.debug(f"norm_log_spec with batch shape: {norm_log_spec.shape}")
        logging.debug(f"torch_input shape: {torch_input.shape}")
        logging.debug(f"padded_input shape: {padded_input.shape}")
        logging.debug(f"model probs shape: {probs.shape}")
        logging.info(f"predictions: {predictions}")
        acc = 3
        logging.info(f"log_spec             time (s): {round(fa_log_spec_time, acc)}")
        logging.info(f"normalization        time (s): {round(fa_normalize_time, acc)}")
        logging.info(f"convert & pad        time (s): {round(fa_convert_pad_time, acc)}")
        logging.info(f"model infer          time (s): {round(fa_model_infer_time, acc)}")
        logging.info(f"decoder              time (s): {round(fa_decode_time, acc)}")
        logging.info(f"total                time (s): {round(fa_total_time, acc)}")
        # ------------ logging ---------------




class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, 
    and stored in a buffer, to be read from.
    """

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
    
    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            while True:
                yield self.read_resampled()


def max_decode(output, blank=39):
    pred = np.argmax(output, 1)
    prev = pred[0]
    seq = [prev] if prev != blank else []
    for p in pred[1:]:
        if p != blank and p != prev:
            seq.append(p)
        prev = p
    return seq


if __name__ == '__main__':
    BEAM_WIDTH = 500
    DEFAULT_SAMPLE_RATE = 16000

    import argparse
    parser = argparse.ArgumentParser(description="Stream from microphone to DeepSpeech using VAD")
    parser.add_argument('-w', '--savewav',
                        help="Save .wav files of utterences to given directory")
    parser.add_argument('-f', '--file',
                        help="Read from .wav file instead of microphone")
    parser.add_argument('-m', '--model',
                        help="Path to the model (protocol buffer binary file, or entire directory containing all standard-named files for model)")
    parser.add_argument('-c', '--config', type = str,
                        help="Path to the config file for that model"),
    parser.add_argument('-d', '--device', type=int, default=None,
                        help="Device input index (Int) as listed by pyaudio.PyAudio.get_device_info_by_index(). If not provided, falls back to PyAudio.get_default_device().")
    parser.add_argument('-r', '--rate', type=int, default=DEFAULT_SAMPLE_RATE,
                        help=f"Input device sample rate. Default: {DEFAULT_SAMPLE_RATE}. Your device may require 44100.")
    # ctc decoder not currenlty used
    parser.add_argument('-bw', '--beam_width', type=int, default=BEAM_WIDTH,
                        help=f"Beam width used in the CTC decoder when building candidate transcriptions. Default: {BEAM_WIDTH}")

    ARGS = parser.parse_args()
    if ARGS.savewav: os.makedirs(ARGS.savewav, exist_ok=True)
    main(ARGS)
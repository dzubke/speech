# standard modules
import argparse
import shlex
import subprocess
from time import sleep, time
from queue import Queue
from threading import Thread

# third-party modules
import torch
import numpy as np
import soundfile

# project modules
import speech
from speech.loader import log_specgram
from speech.models.ctc_decoder import decode




"""
This implementation of model streaming is similar to Mozilla's implementation at: 
https://github.com/mozilla/DeepSpeech/blob/v0.5.1/native_client/deepspeech.cc#L62

   The streaming process uses three queues 
   - mic_queue, the mic_record method collects audio samples from the microphone
                and places them in the audio queue
   - preprocess_queue, the objects in the audio queue are collected by the audio_collection method
                into sets of 10 and put in the preprocess queue. Also, an overlap factor 
                can be set that will add an additional number of mic_buffers into a single 
                audio object to be put in the preprocess queue
   - model_queue, the objects in the preprocess queue are then fed into the model by the infer method
                 to calculate the probabilies of each phoneme label at each timestep. In the 
                 model.infer() call, those probabilities are also fed into the decoder to output
                 phoneme labels. 
    The phoneme label predictions are then added to the predictions list which is printed to stdout.

    This program is ended through a keyboard interrupt. The multi-threading relies on serveral infinite
    loops that need to be interrupted to shut the program down. This isn't ideal, but is how this code is
    written right now. 
"""

class StreamInfer():
    def __init__(self):
        # initializing the queues
        self.audio_q = Queue()
        self.preprocess_q = Queue()
        #self.model_q = Queue()
        self.predictions = []   # a list to contain the final predictions from the ctc_decoder
        self.audio_collection_count: int = 0    # to debug, a count of the number of audio collections created
        self.mic_put_time: float = 0.0       # to debug, sorts cummulative time to assign mic buffers
    

    def start_stream(self, audio_buffer_size: int = 10):
        """Creates a thread that will collect audio_buffers from the local microphone

        """

        mic_thread = Thread(target=self.mic_record, args=())
        mic_thread.start()

    def mic_record(self):
        """Function that does the audio collection within the thread called in start_stream
            Notes:
                Format of the the rec command: -q=quiet mode, -V0=volume factor of 0, -e signed=a signed integer encoding
                    -L=endian little, -c 1=one channel, -b 16=16 bit sample size, -r 16k=16kHZ sampele rate
                    -t raw=raw file type , - gain -2= 
        """
        
        # creates a subprocess object to record from the local mic
        subproc_args = 'rec -q -V0 -e signed -L -c 1 -b 16 -r 16k -t raw - gain -2'
        subproc = subprocess.Popen(shlex.split(subproc_args),
                            stdout=subprocess.PIPE,
                            bufsize=0)
        try:
            while True:
                mic_buffer = subproc.stdout.read(512)
                self.audio_q.put(mic_buffer)

        except KeyboardInterrupt:
            print("exiting mic_record")
            subproc.terminate()
            subproc.wait()
    

    def start_preprocess(self, preproc):

        preprocess_thread = Thread(target=self.preprocess, args=(preproc,))
        preprocess_thread.start()

    def preprocess(self, preproc):
        """this function gets

        """
        log_buffer = []
        # with window of 512 frames (32ms) you need two 256-f mic buffers
        log_window_size = 2 
        try:
            while True:
                if self.audio_q.empty():
                    continue
                else: 
                    if len(log_buffer) < log_window_size: 
                        log_buffer.append(
                            np.frombuffer(self.audio_q.get(), dtype=np.int16)
                            )
                        continue
                    else:
                        if not self.audio_q.empty():
                            self._process_log_buffer(log_buffer, preproc, self.preprocess_q)
                            log_buffer = self._progress_window(log_buffer,  
                                    np.frombuffer(self.audio_q.get(), dtype=np.int16)
                                    )
    
        except KeyboardInterrupt:
            print("existing preprocess")

    def _process_log_buffer(self, log_buffer:list, preproc, preprocess_q):
        """
        process the log_buffer though the log-spec and adds to the model queue
        """
        preprocess_input = np.concatenate(log_buffer, axis=0)
        log_spec = log_specgram(preprocess_input, sample_rate=16000, window_size=32, step_size=16)
        norm_log_spec = (log_spec - preproc.mean) / preproc.std
        preprocess_q.put(norm_log_spec)

    def _progress_window(self, buffer:list, data):
        """
        moves the buffer forward in time by droping the oldest value and adding
        a new value from the queue
            buffer (list): buffer to be filled and updated
            data(buffer or np.ndarray): object to fill the buffer
        """
        del buffer[0]
        buffer.append(data)
        return buffer


    def start_infer(self, model, preproc):
        infer_thread = Thread(target=self.infer, args=(model, preproc))
        infer_thread.start()

    def infer(self, model, preproc):
        # loading the model conducting inference and the preprocessing object preproc
        layer_count = 5
        rnn_args = (torch.randn(layer_count * 1, 1, 512), torch.randn(layer_count * 1, 1, 512))      
        conv_buffer = []
        conv_window_size = 31
        try:
            while True:
                if self.preprocess_q.empty():
                    continue
                else: 
                    if len(conv_buffer) < conv_window_size: 
                        conv_buffer.append(self.preprocess_q.get())
                        continue
                    else:
                        rnn_args = self._run_inference(model, preproc, rnn_args, conv_buffer, self.predictions)
                        conv_buffer = self._progress_window(conv_buffer,  self.preprocess_q.get())

        except KeyboardInterrupt:
            print("exiting infer")

    def _run_inference(self, model, preproc, rnn_args, conv_buffer:list, predictions:list):
        """
        conducts inference on the input conv_buffer. This function should call the existing
        model.infer() method but because that takes in a batch and not rnn_args and doesn't
        return rnn_args, it was easier (though less clean) just to copy the code over. 
            model : the pytorch model
            preproc (class): a Preprocessing class created in loader.py
            rnn_args (tuple(torch.tensor)): the hidden (and cell if LSTM) states of the model rnn
            conv_buffer (list): a list of np.arrays making up the convolutional buffer
            predictions (list): the list of phoneme label predictions
        """
        conv_input = np.concatenate(conv_buffer, axis=0)
        conv_input = torch.from_numpy(conv_input)
        conv_input = conv_input.unsqueeze(0)
        #fake_label = [27]
        #dummy_batch = ((conv_input,), (fake_label,))  # model.infer expects 2-element tuple
        probs, rnn_args = model.forward_impl(conv_input, rnn_args, softmax=True)
        # convert the torch tensor into a numpy array
        #probs = probs.data.cpu().numpy()
        #print(f"probs shape: {probs.shape}")
        #preds = [decode(p, beam_size=3, blank=39)[0] for p in probs] 
        #preds = [preproc.decode(pred) for pred in preds]
        predictions.extend([1])
        return rnn_args 

    
    def check_queue_size(self):
        """Checks the size of the preprocess_q and model_q
        """
        
        print(f"audio_q size: {self.audio_q.qsize()}, \
                preprocess_q size: {self.preprocess_q.qsize()}, \
                predictions length: {len(self.predictions)}")
    

def main(model_path: str):
    """This function takes in a path to a pytorch model and prints predictions of the model from live streaming
        audio from a computer microphone.

    """

    audio_buffer_size = 1
    stream_infer = StreamInfer()
    main_start_time = time()
    # collects audio from the microphone into the audio buffer audio_q
    stream_infer.start_stream()     
    model, preproc = speech.load(model_path, tag='best')
    # gets audio buffers from audio_q, preprocesses it, and puts it on the model_q
    stream_infer.start_preprocess(preproc)
    # getss preprocessed objects from model_q and updates the predictions list with the predictions 
    stream_infer.start_infer(model, preproc)     

    try:
        while True:
            sleep(0.2)
            stream_infer.check_queue_size() # output the final predictions
            print(stream_infer.predictions)

    except KeyboardInterrupt:
        #soundfile.write('new_file.wav', np_array, 16000)
        main_stop_time = time()
        time_duration = round(main_stop_time-main_start_time, 6)
        time_collected = stream_infer.audio_collection_count*audio_buffer_size*0.016
        print(f"time duration: {time_duration} sec")
        print(f"time collected: {time_collected} sec")
        print(f"time difference: {time_duration - time_collected} sec")
        print(f"mic_put_time: {stream_infer.mic_put_time} sec")
        print('All predictions:', stream_infer.predictions)


if __name__ == "__main__":
    ### format of script command
    # python streaming.py <path_to_model>
    parser = argparse.ArgumentParser(
            description="Will provide streaming predictions from model.")
    parser.add_argument("model",
        help="Path to the pytorch model.")

    args = parser.parse_args()

    main(args.model)

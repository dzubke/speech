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



"""
This implementation of model streaming is similar to Mozilla's implementation at: 
https://github.com/mozilla/DeepSpeech/blob/v0.5.1/native_client/deepspeech.cc#L62

   The streaming process uses three queues 
   - audio_buffer, collects audio samples until there's enough data to
     compute input features for a single window.
   - mfcc_buffer, used to buffer input features until there's enough data for
     a single timestep. Remember there's overlap in the features, each timestep
     contains n_context past feature frames, the current feature frame, and
     n_context future feature frames, for a total of 2*n_context + 1 feature
     frames per timestep.
   - batch_buffer, used to buffer timesteps until there's enough data to compute
     a batch of n_steps.
   Data flows through all three buffers as audio samples are fed via the public
   API. When audio_buffer is full, features are computed from it and pushed to
   mfcc_buffer. When mfcc_buffer is full, the timestep is copied to batch_buffer.
   When batch_buffer is full, we do a single step through the acoustic model
   and accumulate results in the DecoderState structure.
   When finishStream() is called, we decode the accumulated logits and return
   the corresponding transcription.
"""



class StreamInfer():
    def __init__(self):
        # initializing the queues
        self.audio_q = Queue()
        self.preprocess_q = Queue()
        self.model_q = Queue()
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
                put_start = time()
                self.audio_q.put(mic_buffer)
                put_stop = time()
                self.mic_put_time += (put_stop - put_start)
        
        except KeyboardInterrupt:
            print("exiting mic_record")
            subproc.terminate()
            subproc.wait()

    def start_collection(self, audio_buffer_size):

        mic_thread = Thread(target=self.audio_collection, args=(audio_buffer_size,))
        mic_thread.start()
        
    def audio_collection(self, audio_buffer_size):
        """This function collects the number of mic_buffers specified in audio_buffer_size into 
            a numpy array and puts it on the proprocess_q
        """
        overlap = 5  # the amount of overlapping mic_buffers between each audio_buffer
        tail_cache = np.array([], dtype=np.int16)
        head_cache = np.array([], dtype=np.int16)

        try:
            while True:
                # reinitialize the audio buffer to the tail_cache
                
                audio_buffer = tail_cache

                for _ in range(audio_buffer_size - 2*overlap):
                    audio_buffer = self.bufferq_to_numpy(audio_buffer)
                
                if  tail_cache.shape[0] != 0:       # the if statement avoids the first loop when the tail_cache is empty
                    assert audio_buffer.shape[0] == (audio_buffer_size)*256, \
                        "incorrect intermediate audio_buffer shape. expected: %i, actual %i" % (int(audio_buffer_size*256), audio_buffer.shape[0])
                
                for _ in range(2*overlap):
                    head_cache = self.bufferq_to_numpy(head_cache)

                assert head_cache.shape[0] == 2*overlap*256, \
                    "incorrect head cache shape, expected: %i, actual %i" % (int(2*overlap*256), head_cache.shape[0])

                audio_buffer = np.append(audio_buffer, head_cache)
            
                if  tail_cache.shape[0] != 0:       # the if statement avoids the first loop when the tail_cache is empty
                    assert audio_buffer.shape[0] == (audio_buffer_size + 2*overlap)*256, \
                        "incorrect final audio_buffer shape.  expected: %i, actual %i" % (int((audio_buffer_size + 2*overlap)*256), audio_buffer.shape[0])

                # add the audio_buffer to the preprocess_q
                self.preprocess_q.put(audio_buffer)
                self.audio_collection_count += 1
                tail_cache = head_cache
                head_cache = np.array([], dtype=np.int16)   # setting head_cache back to zero


        except KeyboardInterrupt:
            print("exiting audio_collection")

    def bufferq_to_numpy(self, audio_array: np.ndarray) -> np.ndarray:
        """gets a mic_buffer from the audio_q, converts it to a numpy array and appends
            it to the input audio_array
        """
        mic_buffer = self.audio_q.get()
        np_mic_buffer = np.frombuffer(mic_buffer, dtype=np.int16)
        audio_array = np.append(audio_array, np_mic_buffer)
        return audio_array
    

    def start_preprocess(self, preproc):

        preprocess_thread = Thread(target=self.preprocess, args=(preproc,))
        preprocess_thread.start()

    def preprocess(self, preproc):
        """this function gets

        """
        try:
            while True:
                np_array = self.preprocess_q.get()
                log_spec = log_specgram(np_array, sample_rate=16000)
                norm_log_spec = (log_spec - preproc.mean) / preproc.std
                self.model_q.put(norm_log_spec)
        except KeyboardInterrupt:
            print("existing preprocess")


    def start_infer(self, model, preproc):

        infer_thread = Thread(target=self.infer, args=(model, preproc))
        infer_thread.start()

    def infer(self, model, preproc):

        # loading the model conducting inference and the preprocessing object preproc
        fake_label = [27]
        try:
            while True:
                norm_log_spec = self.model_q.get()
                dummy_batch = ((norm_log_spec,), (fake_label,))  # model.infer expects 2-element tuple
                preds = model.infer(dummy_batch)
                preds = [preproc.decode(pred) for pred in preds]
                self.predictions.extend(*preds)
                pickel_once +=1
        except KeyboardInterrupt:
            print("existing infer")

    
    def check_queue_size(self):
        """Checks the size of the preprocess_q and model_q
        """
        
        print(f"audio_q size: {self.preprocess_q.qsize()}, \
                preprocess_q size: {self.preprocess_q.qsize()}, \
                model_q size: {self.model_q.qsize()}, \
                predictions length: {len(self.predictions)}")




def main(model_path: str):
    """This function takes in a path to a pytorch model and prints predictions of the model from live streaming
        audio from a computer microphone.

    """

    audio_buffer_size = 100

    assert audio_buffer_size > 9, "audio_buffer size must be greater than 9"

    stream_infer = StreamInfer()

    main_start_time = time()

    stream_infer.start_stream()     # collects audio from the microphone into audio buffer and puts it on the preprocess queue
    
    stream_infer.start_collection(audio_buffer_size)     # collects audio from the microphone into audio buffer and puts it on the preprocess queue
    
    model, preproc = speech.load(model_path, tag='')

    stream_infer.start_preprocess(preproc)      # continually gets audio buffers from preprocess_q, preprocesses it, and puts it on the model_q

    stream_infer.start_infer(model, preproc)     # continually getss preprocessed objects from model_q and updates the predictions list with the predictions 



    try:
        while True:
            sleep(0.1)
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

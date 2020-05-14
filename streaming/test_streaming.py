import argparse
import speech
import torch
import shlex
import subprocess
import numpy as np
import soundfile
from speech.loader import log_specgram
import pickle

import speech.models as models


freq_dim = 257 #freq dimension out of log_spectrogram 
time_dim = 186  #time dimension out of log_spectrogram
state_dict_fn = '/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/onnx_coreml/torch_models/20200211-0212_w32-s16_3sec_state_dict.pth'

def main():

    trained_model = models.CTC(freq_dim,40, model_cfg)
    state_dict = torch.load(state_dict_fn, map_location=torch.device('cpu'))
    trained_model.load_state_dict(state_dict)


def predict_from_stream(model_path: str):
    """This function takes in a path to a pytorch model and prints predictions of the model from live streaming
        audio from a computer microphone.
    """

    # the rec command: -q=quiet mode, -V0=volume factor of 0, -e signed=a signed integer encoding
    ## -L=endian little, -c 1=one channel, -b 16=16 bit sample size, -r 16k=16kHZ sampele rate
    ## -t raw=raw file type , - gain -2= 
    args = 'rec -q -V0 -e signed -L -c 1 -b 16 -r 16k -t raw - gain -2'
    subproc = subprocess.Popen(shlex.split(args),
                            stdout=subprocess.PIPE,
                            bufsize=0)

    model, preproc = speech.load(model_path, tag='')
    num_buffers = 250
    all_preds=[]
    try:
        while True:
            print('You can start speaking now. Press Control-C to stop recording.')
            data_list = []
            for _ in range(num_buffers):
                data = subproc.stdout.read(512)
                data_list.append(data)

            np_array = np.array([], dtype=np.int16)
            for data in data_list:
                np_data = np.frombuffer(data, dtype=np.int16)
                np_array = np.append(np_array, np_data)

            log_spec = log_specgram(np_array, sample_rate=16000)
            norm_log_spec = (log_spec - preproc.mean) / preproc.std
            fake_label = [27]

            dummy_batch = ((norm_log_spec,), (fake_label,))  # model.infer expects 2-element tuple
            preds = model.infer(dummy_batch)
            preds = [preproc.decode(pred) for pred in preds]
            print(preds)
            all_preds.extend(preds)

    except KeyboardInterrupt:
        #soundfile.write('new_file.wav', np_array, 16000)
        print('All predictions:', all_preds)
        subproc.terminate()
        subproc.wait()



if __name__ == "__main__":
    ### format of script command
    # python streaming.py <path_to_model>
    parser = argparse.ArgumentParser(
            description="Will provide streaming predictions from model.")
    parser.add_argument("model",
        help="Path to the pytorch model.")

    args = parser.parse_args()

    predict_from_stream(args.model) 


"""scratch from ipython

stream_log_spec = np.empty((0,257)).astype(np.float32) 
    ...: model.eval() 
    ...: with wave.open(audio_path, 'rb') as wf:  
    ...:         chunk_size = 256 
    ...:         audio_ring_buffer = deque(maxlen=2)   
    ...:         conv_ring_buffer    = collections.deque(maxlen=31) 
    ...:         probs_list          = list() 
    ...:         hidden_in           = torch.zeros((5, 1, 512), dtype=torch.float32) 
    ...:         cell_in             = torch.zeros((5, 1, 512), dtype=torch.float32) 
    ...:         num_samples = wf.getnframes()//chunk_size  
    ...:         print("num frames: ", wf.getnframes())  
    ...:         print("num_samples: ", num_samples)  
    ...:         for i in range(num_samples):  
    ...:             if len(audio_ring_buffer) < 1:  
    ...:                 audio_ring_buffer.append(wf.readframes(chunk_size))  
    ...:             else:  
    ...:                 audio_ring_buffer.append(wf.readframes(chunk_size))  
    ...:                 buffer_list = list(audio_ring_buffer)  
    ...:                 numpy_buffer = np.concatenate(  
    ...:                         (np.frombuffer(buffer_list[0], np.int16),   
    ...:                         np.frombuffer(buffer_list[1], np.int16)))  
    ...:                 log_spec_step = log_specgram_from_data( 
    ...:                     numpy_buffer, samp_rate=16000, window_size=32, step_size=16) 
    ...:                 norm_log_spec = normalize(preproc, log_spec_step) 
    ...:                 stream_log_spec = np.concatenate((stream_log_spec, norm_log_spec), axis=0) 
    ...:                 if len(conv_ring_buffer) < 30: 
    ...:                     conv_ring_buffer.append(norm_log_spec) 
    ...:                 else: 
    ...:                     conv_ring_buffer.append(norm_log_spec) 
    ...:                     conv_context = np.concatenate(list(conv_ring_buffer), axis=0) 
    ...:                     conv_context = np.expand_dims(conv_context, axis=0) 
    ...:                     model_out = model(torch.from_numpy(conv_context), (hidden_in, cell_in)) 
    ...:                     probs, (hidden_out, cell_out) = model_out 
    ...:                     probs = to_numpy(probs) 
    ...:                     probs_list.append(probs) 
    ...:                     hidden_in, cell_in = hidden_out, cell_out 

"""
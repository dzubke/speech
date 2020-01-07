import argparse
import speech
import torch
import shlex
import subprocess
import numpy as np
import soundfile
from speech.loader import log_specgram



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
            np_array = np.array([], dtype=np.int16)
            for _ in range(num_buffers):
                data = subproc.stdout.read(512)
                np_data = np.frombuffer(data, dtype=np.int16)
                np_array = np.append(np_array, np_data)

            log_spec = log_specgram(np_array, sample_rate=16000)
            norm_log_spec = (log_spec - preproc.mean) / preproc.std
            fake_label = [27]
            dummy_batch = ((norm_log_spec,), (fake_label,))  # model.infer expects 2-element tuple
            preds = model.infer(dummy_batch)
            preds = [preproc.decode(pred) for pred in preds]
            print(preds)
            all_preds.append(preds)

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
# standard library
import json
import argparse

# third-party libraries
import tqdm

# project libraries
from tests import model_debug
from speech.utils.io import load



def main(json_path, preproc_path):
    """
        this file iterates through all audio paths in json_path, processes the audio,
        and checks if there are any nan values in the processed output.
        Arguments:
            json_path (str): path to data json
            preproc_path (str): path to dir with preproc object, don't include preproc.pyc in path
    """
    
    print(f"checking: {json_path} with preproc: {preproc_path}")
    with open(json_path) as fid:
        data_json = [json.loads(l) for l in fid]

    _, preproc = load(preproc_path)
    

    json_has_nan = False
    tq = tqdm.tqdm(data_json)
    for sample in tq:
        samp_path = sample["audio"]
        inputs, sample_rate = model_debug.load_audio(samp_path, preproc)

        # checks for nan values
        if (inputs!=inputs).any():
            print(f"nan value in: {samp_path}")
            json_has_nan = True
    
    if not json_has_nan:
        print(f"data json doesn't have nan values")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Checks for nan values in processed audio specified in json_dir")
    parser.add_argument("json_path", help="path to the data json to be checked for nan values")
    parser.add_argument("preproc_path", help="path to directory with preproc object")
    args = parser.parse_args()

    main(args.json_path, args.preproc_path)

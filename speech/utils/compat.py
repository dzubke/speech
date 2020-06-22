# standard libraries
import platform
# third-party libraries
import numpy as np
# project libraries
from speech.loader import Preprocessor

def normalize(preproc, features:np.ndarray):
    """
    These methods are too allow for backwards compatbility with older objects that don't
    possess the most recent object methods.
    """

    if hasattr(preproc, "normalize"):
        norm_features = preproc.normalize(features)
    else: 
        norm_features= normalize_helper(preproc, features)
    return norm_features

def normalize_helper(preproc:Preprocessor, np_arr:np.ndarray):
    """
    takes in a preproc object from loader.py and returns
    the normalized output.
    """
    output = (np_arr - preproc.mean) / preproc.std
    return output.astype(np.float32)


def get_main_dir_path():
    """
    returns the path to the main speech directory for mac and linux depending on the OS
    """
    # checks if I am on my local (mac) or a VM (linux)
    system = platform.system()
    if system == 'Linux':
        path_prefix = '/home/dzubke/awni_speech/speech/'
    elif system == 'Darwin':
        path_prefix = '/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/'

    return path_prefix

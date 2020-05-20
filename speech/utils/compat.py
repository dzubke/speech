"""
These methods are too allow for backwards compatbility with older objects that don't
possess the most recent object methods.
"""

# third-party libraries
import numpy as np
# project libraries
from speech.loader import Preprocessor

def normalize(preproc:Preprocessor, np_arr:np.ndarray):
    """
    takes in a preproc object from loader.py and returns
    the normalized output.
    """
    output = (np_arr - preproc.mean) / preproc.std
    return output.astype(np.float32)
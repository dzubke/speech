# compatibility modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# standard libraries
import os
import subprocess
# third-party libaries
import numpy as np

FFMPEG = "ffmpeg"
AVCONV = "avconv"

def check_install(*args):
    try:
        subprocess.check_output(args,
                    stderr=subprocess.STDOUT)
        return True
    except OSError as e:
        return False

def check_avconv():
    """
    Check if avconv is installed.
    """
    return check_install(AVCONV, "-version")

def check_ffmpeg():
    """
    Check if ffmpeg is installed.
    """
    return check_install(FFMPEG, "-version")

USE_AVCONV = check_avconv()
USE_FFMPEG = check_ffmpeg()

if not (USE_AVCONV or USE_FFMPEG):
    raise OSError(("Must have avconv or ffmpeg "
                   "installed to use conversion functions."))
USE_AVCONV = not USE_FFMPEG

def to_wave(audio_file, wave_file, use_avconv=USE_AVCONV):
    """
    Convert audio file to wave format.
    """
    prog = AVCONV if use_avconv else FFMPEG
    args = [prog, "-y", "-i", audio_file, "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", "-f", "wav", wave_file]
    subprocess.check_output(args, stderr=subprocess.STDOUT)

def convert_2channels(audio_file:str, max_channels:int=1):
    """
    if the input audio file has more than the max_channels, the file will be converted
    to a version with a single channel.
    Set max_channels=0 to convert all files
    """
    cmd = subprocess.check_output(["soxi", audio_file])
    num_chan = parse_soxi_out(cmd)
    if num_chan>max_channels: 
        os.rename(audio_file, "/tmp/convert_2channels_audio.wav")
        to_wave("/tmp/convert_2channels_audio.wav", audio_file)

def parse_soxi_out(cmd:bytes):
    """
    this gross parser takes the bytes from the soxi output, decodes to utf-8, 
    splits by the newline "\n", takes the second element of the array which is
    the number of channels, splits by the semi-colon ':' takes the second element
    which is the string of the num channels and converts to int.
    """
    return int(cmd.decode("utf-8").strip().split("\n")[1].split(':')[1].strip())

def to_numpy(tensor):
    """
    converts a torch tensor to numpy array
    """
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# The two functions below pcm2float and float2pcm are taken from 
# https://gist.github.com/HudsonHuang/fbdf8e9af7993fe2a91620d3fb86a182

def pcm2float(sig, dtype='float64'):
    """Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float64' for double precision.
    Parameters
    ----------
    sig : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.
    Returns
    -------
    numpy.ndarray
        Normalized floating point data.
    See Also
    --------
    float2pcm, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max


def float2pcm(sig, dtype='int16'):
    """Convert floating point signal with a range from -1 to 1 to PCM.
    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.
    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.
    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.
    See Also
    --------
    pcm2float, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


if __name__ == "__main__":
    print("Use avconv", USE_AVCONV)




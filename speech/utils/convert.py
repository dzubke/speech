from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess
import wave
import os

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
    args = [prog, "-y", "-i", audio_file, "-ac", "1", "-ar", "16000","-f", "wav", wave_file]
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


if __name__ == "__main__":
    print("Use avconv", USE_AVCONV)




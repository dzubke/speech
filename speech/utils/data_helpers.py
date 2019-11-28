from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import tqdm

from speech.utils import convert

def convert_full_set(path, pattern, new_ext="wav", **kwargs):
    pattern = os.path.join(path, pattern)
    print(f"pattern:{pattern}")
    audio_files = glob.glob(pattern)
    print(f"audio_files:{audio_files}")
    for af in tqdm.tqdm(audio_files):
        base, ext = os.path.splitext(af)
        wav = base + os.path.extsep + new_ext
        convert.to_wave(af, wav, **kwargs)





# standard libraries
import os
import re
import argparse
import subprocess
import io
import json
import unicodedata
import tarfile
# third-party libraries
#import wget
from tqdm import tqdm
#project libraries
#from utils import create_manifest
from speech.utils import data_helpers
from speech.utils import wave 


def main(target_dir, tar_path, sample_rate, min_duration, max_duration, use_phonemes, no_segment):
    target_dl_dir = target_dir
    if not os.path.exists(target_dl_dir):
        os.makedirs(target_dl_dir)

    target_unpacked_dir = os.path.join(target_dl_dir, "TEDLIUM_release-3")
    if tar_path and os.path.exists(tar_path):
        target_file = tar_path
    else:
        print("Could not find downloaded TEDLIUM archive, Downloading corpus...")
        #wget.download(TED_LIUM_V2_DL_URL, target_dl_dir)
        target_file = os.path.join(target_dl_dir, "TEDLIUM_release-3.tgz")

    if not os.path.exists(target_unpacked_dir):
        print("Unpacking corpus...")
        tar = tarfile.open(target_file)
        tar.close()
        tar.extractall(target_dl_dir)
    else:
        print("Found TEDLIUM directory, skipping unpacking of tar files")

    # legacy means the data are in the format of previous version. 
    # legacy contains all the data in tedlium v3
    train_ted_dir = os.path.join(target_unpacked_dir, "legacy", "train")
    val_ted_dir = os.path.join(target_unpacked_dir, "legacy", "dev")
    test_ted_dir = os.path.join(target_unpacked_dir, "legacy", "test")

    prepare_dir(train_ted_dir, use_phonemes, no_segment)
    prepare_dir(val_ted_dir,  use_phonemes, no_segment)
    prepare_dir(test_ted_dir,  use_phonemes, no_segment)
    #print('Creating manifests...')

    #create_manifest(train_ted_dir, 'ted_train_manifest.csv', min_duration, max_duration)
    #create_manifest(val_ted_dir, 'ted_val_manifest.csv')
    #create_manifest(test_ted_dir, 'ted_test_manifest.csv')


def prepare_dir(ted_dir, use_phonemes, no_segment):
    """
        processed the audio and labels
        Arguments:
            ted_dir - str: path to the directory with the dataset
            use_phonemes (bool): if true, phoneme labels will be used
            segment_audio (bool): if true, the original audio will be segmented
    """

    converted_dir = os.path.join(ted_dir, "converted")
    # directories to store converted wav files and their transcriptions
    wav_dir = os.path.join(converted_dir, "wav")
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)
    txt_dir = os.path.join(converted_dir, "txt")
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    counter = 0
    entries = os.listdir(os.path.join(ted_dir, "sph"))
    
    unknown_set, unknown_dict = set(), dict()
    line_count, word_count= 0, 0
    with open(os.path.join(ted_dir, "data.json"), 'w') as fid:
        for sph_file in tqdm(entries, total=len(entries)):
            speaker_name = sph_file.split('.sph')[0]

            sph_file_full = os.path.join(ted_dir, "sph", sph_file)
            stm_file_full = os.path.join(ted_dir, "stm", "{}.stm".format(speaker_name))

            assert os.path.exists(sph_file_full) and os.path.exists(stm_file_full)
            if use_phonemes:
                LEXICON_PATH = "TEDLIUM.162k.dic"
                word_phoneme_dict = data_helpers.lexicon_to_dict(LEXICON_PATH, corpus_name="tedlium")
            all_utterances = get_utterances_from_stm(stm_file_full)

            all_utterances = filter(filter_short_utterances, all_utterances)
            for utterance_id, utterance in enumerate(all_utterances):
                target_fn = "{}_{}.wav".format(utterance["filename"], str(utterance_id))
                target_wav_file = os.path.join(wav_dir, target_fn+".wav")
                target_txt_file = os.path.join(txt_dir, target_fn+".txt")
                if not no_segment:
                    cut_utterance(sph_file_full, target_wav_file, 
                        utterance["start_time"], utterance["end_time"])
                
                if use_phonemes: 
                #checks for unknown characters, records information, and continues to next utterance
                    unk_words_list, unk_words_dict, counts = data_helpers.check_unknown_words(target_fn, utterance["transcript"], word_phoneme_dict)
                    line_count+=counts[0]
                    word_count+=counts[1]
                    if len(unk_words_list) > 0: 
                        unknown_set.update(unk_words_list)
                        unknown_dict.update(unk_words_dict)
                        continue
                dur = wave.wav_duration(target_wav_file)
                text = _preprocess_transcript(utterance["transcript"], use_phonemes, word_phoneme_dict)
                datum = {'text' : text,
                        'duration' : dur,
                        'audio' : target_wav_file}
                
                json.dump(datum, fid)
                fid.write("\n")
            counter += 1
    
    data_helpers.process_unknown_words(ted_dir, unknown_set, unknown_dict, line_count, word_count)


def get_utterances_from_stm(stm_file):
    """
    Return list of entries containing phrase and its start/end timings
    :param stm_file:
    :return:
    """
    res = []
    with io.open(stm_file, "r", encoding='utf-8') as f:
        for stm_line in f:
            tokens = stm_line.split()
            start_time = float(tokens[3])
            end_time = float(tokens[4])
            filename = tokens[0]
            transcript = unicodedata.normalize("NFKD",
                                               " ".join(t for t in tokens[6:]).strip()). \
                encode("utf-8", "ignore").decode("utf-8", "ignore")
            if transcript != "ignore_time_segment_in_scoring":
                res.append({
                    "start_time": start_time, "end_time": end_time,
                    "filename": filename, "transcript": transcript
                })
            
        return res


def filter_short_utterances(utterance_info, min_len_sec=1.0):
    return (utterance_info["end_time"] - utterance_info["start_time"]) > min_len_sec


def cut_utterance(src_sph_file, target_wav_file, start_time, end_time, sample_rate=16000):
    subprocess.call(["sox {}  -r {} -b 16 -c 1 {} trim {} ={}".format(src_sph_file, str(sample_rate),
                                                                      target_wav_file, start_time, end_time)],
                    shell=True)


def _preprocess_transcript(phrase, use_phonemes, word_phoneme_dict):
    """
        transform the input phrase. if use_phonemes is true, the words in phrase
        are transformed into phoneme labels specified in word_phoneme_dict
        Arguments:
            phrase (list): list of words
            use_phonemes (bool): if True, transform phrase into phonemes
            word_phoneme_dict (dict): dictionary mapping words to phonemes
    """
    phrase = phrase.strip()
    if use_phonemes:
        if type(phrase) == str:
            phrase = phrase.split() #converting string of space-separated words to list of words
        elif type(phrase) == list: 
            pass
        else: 
            raise(TypeError("input text is not string or list type"))
        phonemes = []
        for word in phrase:
            phonemes.extend(word_phoneme_dict[word])
        phrase = phonemes
    
    return phrase




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Processes and downloads TED-LIUMv2 dataset.')
    parser.add_argument("--target-dir", default='TEDLIUM_dataset/', type=str, help="Directory to store the dataset.")
    parser.add_argument("--tar-path", type=str, help="Path to the TEDLIUM_release tar if downloaded (Optional).")
    parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
    parser.add_argument('--min-duration', default=1, type=int,
                        help='Prunes training samples shorter than the min duration (given in seconds, default 1)')
    parser.add_argument('--max-duration', default=20, type=int,
                        help='Prunes training samples longer than the max duration (given in seconds, default 20)')
    parser.add_argument('--use_phonemes', default=False, type=bool,
                        help='Determines whether output phoneme labels.')
    parser.add_argument('--no_segment', default=False, type=bool,
                        help='if true, original audio files will not be segmented')
    args = parser.parse_args()

    TED_LIUM_V2_DL_URL = "http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz"

    main(args.target_dir, args.tar_path, args.sample_rate, args.min_duration, args.max_duration, args.use_phonemes, args.no_segment)
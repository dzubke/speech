# standard libary
import os
import json
from collections import defaultdict
import argparse
import glob
import subprocess
import csv
import re
# third party libraries
import tqdm
# project libraries
from speech.utils import data_helpers, wave, convert


class Preprocessor(object):
    
    def __init__(self, dataset_dir:str, dataset_name:str, lexicon_path:str,
                        force_convert:bool, min_duration:float, max_duration:float):

        self.dataset_dir = dataset_dir
        self.lex_dict = data_helpers.lexicon_to_dict(lexicon_path, dataset_name.lower())\
             if lexicon_path !='' else None
        # list of tuples of audio_path and transcripts
        self.audio_trans=list()
        self.force_convert = force_convert
        self.min_duration = min_duration
        self.max_duration = max_duration

    def process_datasets(self):
        """
        This function is usually written to iterate through the datasets in dataset_dict
        and call collect_audio_transcrtips which stores the audio_path and string transcripts
        ini the self.audio_trans object. Then, self.write_json writes the audio and transcripts
        to a file.
        """
        raise NotImplementedError

    def collect_audio_transcripts(self):
        raise NotImplementedError
    
    def write_json(self, save_path:str):
        """
        this method converts the audio files to wav format, filters out the 
        audio files based on the min and max duration and saves the audio_path, 
        transcript, and duration into a json file specified in the input save_path
        """
        # filter the entries by the duration bounds and write file
        unknown_words = UnknownWords()
        with open(save_path, 'w') as fid:
            print("Writing files to label json")
            for sample in tqdm.tqdm(self.audio_trans):
                audio_path, transcript = sample
                if not os.path.exists(audio_path):
                    print(f"file {audio_path} does not exists")
                else:
                    base, raw_ext = os.path.splitext(audio_path)
                    # using ".wv" extension so that original .wav files can be converteds
                    wav_path = base + os.path.extsep + "wv"
                    # if the wave file doesn't exist or it should be re-converted, convert to wave
                    if not os.path.exists(wav_path) or self.force_convert:
                        try:
                            convert.to_wave(audio_path, wav_path)
                        except subprocess.CalledProcessError:
                            # if the file can't be converted, skip the file by continuing
                            print(f"Error converting file: {audio_path}")
                            continue
                    dur = wave.wav_duration(wav_path)
                    if self.min_duration <= dur <= self.max_duration:
                        text = self.process_text(transcript, self.lex_dict, unknown_words, wav_path)
                        # if transcript has an unknown word, skip it
                        if unknown_words.has_unknown: 
                            continue
                        datum = {'text' : text,
                                'duration' : dur,
                                'audio' : wav_path}
                        json.dump(datum, fid)
                        fid.write("\n")
    
        unknown_words.process_save(save_path)
    
    def process_text(self, transcript:str, lex_dict:dict, unknown_words, audio_path:str)->list:
        """
        this method removed unwanted puncutation marks split the text into a list of words
        or list of phonemes if a lexicon_dict exists
        """
        # allows for alphanumeric characters, space, and apostrophe
        accepted_char = '[^A-Za-z0-9 \']+'
        # filters out unaccepted characters, lowers the case, & splits into list
        try:
            transcript = re.sub(accepted_char, '', transcript).lower()
        except TypeError:
            print(f"Type Error with: {transcript}")
        # check that all punctuation (minus apostrophe) has been removed 
        punct_noapost = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
        for p in punct_noapost:
            if p in transcript: raise ValueError(f"unwanted punctuation: {p} in transcript")
        #assert any([p in transcript for p in punct_noapost]), "unwanted punctuation in transcript"
        transcript = transcript.split()
        # if there is a pronunciation dict, convert to phonemes
        if self.lex_dict is not None:
            unknown_words.check_transcript(audio_path, transcript, self.lex_dict)
            phonemes = []
            for word in transcript:
                # TODO: I shouldn't need to include list() in get but dict is outputing None not []
                phonemes.extend(self.lex_dict.get(word, list()))
            transcript = phonemes

        return transcript

    
class CommonvoicePreprocessor(Preprocessor):
    def __init__(self, dataset_dir, dataset_name, lexicon_path,
                        force_convert, min_duration, max_duration):
        super(CommonvoicePreprocessor, self).__init__(dataset_dir, dataset_name, lexicon_path,
            force_convert, min_duration, max_duration)
        self.dataset_dict = {
                            "validated-25-max-repeat": "validated-25-maxrepeat.tsv"
        }

    def process_datasets(self):
        for set_name, label_name in self.dataset_dict.items():
            label_path = os.path.join(self.dataset_dir, label_name)
            print(f"label_path: {label_path}")
            self.collect_audio_transcripts(label_path)
            print(f"len of auddio_trans: {len(self.audio_trans)}")
            root, ext = os.path.splitext(label_path)
            json_path = root + os.path.extsep + "json"
            self.write_json(json_path)
        unique_unknown_words(self.dataset_dir)

    def collect_audio_transcripts(self, label_path:str):
        
        # open the file and select only entries with desired accents
        accents = ['us', 'canada']
        print(f"Filtering files by accents: {accents}")
        dir_path = os.path.dirname(label_path)
        with open(label_path) as fid: 
            reader = csv.reader(fid, delimiter='\t')
            # first line in reader is the header which equals:
            # ['client_id','path','sentence','up_votes','down_votes','age','gender','accent']
            header = next(reader)
            for line in reader:
                # filter by accent
                if line[7] in accents:
                    audio_path = os.path.join(dir_path, "clips", line[1])
                    transcript = line[2]
                    self.audio_trans.append((audio_path, transcript))


class VoxforgePreprocessor(Preprocessor):
    def __init__(self, dataset_dir, dataset_name, lexicon_path,
                        force_convert, min_duration, max_duration):
        super(VoxforgePreprocessor, self).__init__(dataset_dir, dataset_name, lexicon_path,
            force_convert, min_duration, max_duration)
        self.dataset_dict = {"all":"archive"}

    def process_datasets(self):
        for set_name, label_name in self.dataset_dict.items():
            data_path = os.path.join(self.dataset_dir, label_name)
            self.collect_audio_transcripts(data_path)
            json_path = os.path.join(self.dataset_dir, "all.json")
            self.write_json(json_path)
        unique_unknown_words(self.dataset_dir)

    def collect_audio_transcripts(self, data_path:str):
        """
        Voxforge audio in "archive/<sample_dir>/wav/<sample_name>.wav" and
        transcripts are in the file "archive/sample_dir/etc/prompts-original"
        """
        audio_pattern = "*"
        pattern_path = os.path.join(data_path, audio_pattern)
        list_sample_dirs = glob.glob(pattern_path)
        possible_text_fns = ["prompts-original", "PROMPTS", "Transcriptions.txt",  
                                "prompt.txt", "prompts.txt", "therainbowpassage.prompt", 
                                "cc.prompts", "a13.text"]
        print("Processing the dataset directories...")
        for sample_dir in tqdm.tqdm(list_sample_dirs):
            text_dir = os.path.join(sample_dir, "etc")
            # find the frist filename that exists in the directory
            for text_fn in possible_text_fns:
                text_path = os.path.join(text_dir, text_fn)
                if os.path.exists(text_path):
                    break
            with open(text_path, 'r') as fid:
                for line in fid:
                    line = line.strip().split()
                    # if an empty entry, skip it
                    if len(line)==0:
                        continue 
                    audio_name = self.parse_audio_name(line[0])
                    audio_path = self.find_audio_path(sample_dir, audio_name)
                    if audio_path is None:
                        continue
                    # audio_path is corrupted and is skipped
                    elif data_helpers.skip_file(audio_path):
                        continue
                    transcript = line[1:]
                    # transcript should be a string
                    transcript = " ".join(transcript)
                    self.audio_trans.append((audio_path, transcript))

    def parse_audio_name(self, raw_name:str)->str:
        """
        Extracts the audio_name from the raw_name in the PROMPTS file.
        The audio_name should be the last separate string.
        """
        split_chars = r'[/]'
        return re.split(split_chars, raw_name)[-1]


    def find_audio_path(self, sample_dir:str, audio_name:str)->str:
        """
        Most of the audio files are in a dir called "wav" but
        some are in the "flac" dir with the .flac extension
        """
        possible_exts =["wav", "flac"]
        found = False
        for ext in possible_exts:
            file_name = audio_name + os.path.extsep + ext
            audio_path = os.path.join(sample_dir, ext, file_name)
            if os.path.exists(audio_path):
                found =  True
                break
        if not found: 
            audio_path = None
            print(f"dir: {sample_dir} and name: {audio_name} not found")
        return audio_path
        
class TatoebaPreprocessor(Preprocessor):
    def __init__(self, dataset_dir, dataset_name, lexicon_path,
                        force_convert, min_duration, max_duration):
        super(TatoebaPreprocessor, self).__init__(dataset_dir, dataset_name, lexicon_path,
            force_convert, min_duration, max_duration)
        self.dataset_dict = {"all":"sentences_with_audio.csv"}

    def process_datasets(self):
        print("In Tatoeba process_datasets")
        for set_name, label_fn in self.dataset_dict.items():
            label_path = os.path.join(self.dataset_dir, label_fn)
            self.collect_audio_transcripts(label_path)
            root, ext = os.path.splitext(label_path)
            json_path = root + os.path.extsep + "json"
            self.write_json(json_path)
        unique_unknown_words(self.dataset_dir)
    

    def collect_audio_transcripts(self, label_path:str):
        # open the file and select only entries with desired accents
        speakers = ["CK", "Delian", "pencil", "Susan1430"]  # these speakers have north american accents
        print(f"Filtering files by speakers: {speakers}")
        error_files = {"CK": {"min":6122903, "max": 6123834}} # files in this range are often corrupted
        dir_path = os.path.dirname(label_path)
        with open(label_path) as fid: 
            reader = csv.reader(fid, delimiter='\t')
            # first line in reader is the header which equals:
            # ['id', 'username', 'text']
            is_header = True
            for line in reader:
                if is_header:
                    is_header=False
                    continue
                else: 
                    # filter by accent
                    if line[1] in speakers:
                        audio_path = os.path.join(dir_path, "audio", line[1], line[0]+".mp3")
                        transcript = " ".join(line[2:])
                        if data_helpers.skip_file(audio_path):
                            print(f"skipping {audio_path}")
                            continue
                        self.audio_trans.append((audio_path, transcript))

class UnknownWords():

    def __init__(self):
        self.word_set:set = set()
        self.filename_dict:dict = dict()
        self.line_count:int = 0
        self.word_count:int = 0
        self.has_unknown= False

    def check_transcript(self, filename:str, text, word_phoneme_dict:dict):

        if type(text) == str: text = text.split()
        elif type(text) == list: pass
        else: raise(TypeError("input text is not string or list type"))
        self.line_count += 1
        self.word_count += len(text) - 1
        line_unk = [word for word in text if word_phoneme_dict[word]==data_helpers.UNK_WORD_TOKEN]
        #if line_unk is empty, has_unk is False
        self.has_unknown = bool(line_unk)
        if self.has_unknown:
            self.word_set.update(line_unk)
            self.filename_dict.update({filename: len(line_unk)})

    def process_save(self, label_path:str):
        """
        saves a json object of the dictionary with relevant statistics on the unknown words in corpus
        """
        stats_dict=dict()
        stats_dict.update({"unique_unknown_words": len(self.word_set)})
        stats_dict.update({"count_unknown_words": sum(self.filename_dict.values())})
        stats_dict.update({"total_words": self.word_count})
        stats_dict.update({"lines_unknown_words": len(self.filename_dict)})
        stats_dict.update({"total_lines": self.line_count})
        stats_dict.update({"unknown_words_set": list(self.word_set)})
        stats_dict.update({"unknown_words_dict": self.filename_dict})
        
        dir_path, base_name = os.path.dirname(label_path),  os.path.basename(label_path)
        base, ext = os.path.splitext(base_name)
        stats_dir = os.path.join(dir_path, "unk_word_stats")
        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir)
        stats_dict_fn = os.path.join(stats_dir, base+"_unk-words-stats.json")
        with open(stats_dict_fn, 'w') as fid:
            json.dump(stats_dict, fid)
        

def unique_unknown_words(dataset_dir:str):
    """
    Creates a set of the total number of unknown words across all segments in a dataset assuming a
    unk-words-stats.json file from process_unknown_words() has been created for each part of the dataset. 
    Arguments:
        dataset_dir (str): pathname of dir continaing "unknown_word_stats" dir with unk-words-stats.json files
    """
    pattern = os.path.join(dataset_dir, "unk_word_stats", "*unk-words-stats.json")
    dataset_list = glob.glob(pattern)
    unknown_set = set()
    for data_fn in dataset_list: 
        with open(data_fn, 'r') as fid: 
            unk_words_dict = json.load(fid)
            unknown_set.update(unk_words_dict['unknown_words_set'])
            print(len(unk_words_dict['unknown_words_set']))

    unknown_set = filter_set(unknown_set)
    unknown_list = list(unknown_set)

    write_path = os.path.join(dataset_dir, "unk_word_stats","all_unk_words.txt")
    with open(write_path, 'w') as fid:
        fid.write('\n'.join(unknown_list))
    print(f"number of filtered unknown words: {len(unknown_list)}")


def filter_set(unknown_set:set):
    """
    filters the set based on the length and presence of digits.
    """
    unk_filter = filter(lambda x: len(x)<30, unknown_set)
    search_pattern = r'[0-9!#$%&()*+,\-./:;<=>?@\[\\\]^_{|}~]'
    unknown_set = set(filter(lambda x: not re.search(search_pattern, x), unk_filter))
    return unknown_set


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="creates a data json file")
    parser.add_argument("--dataset-dir", type=str,
        help="directory where the dataset is located.")
    parser.add_argument("--dataset-name", type=str,
        help="Name of dataset with a capitalized first letter.")
    parser.add_argument("--lexicon-path", type=str, default='',
        help="path to pronunciation lexicon, if desired.")
    parser.add_argument("--force-convert", action='store_true', default=False,
        help="Converts audio to wav file even if .wav file already exists.")
    parser.add_argument("--min-duration", type=float, default=1, 
        help="minimum audio duration in seconds")
    parser.add_argument("--max-duration", type=float, default=20,
        help="maximum audio duration in seconds")
    args = parser.parse_args()

    data_preprocessor = eval(args.dataset_name+"Preprocessor")
    data_preprocessor = data_preprocessor(args.dataset_dir, args.dataset_name, args.lexicon_path,
                        args.force_convert, args.min_duration, args.max_duration)
    data_preprocessor.process_datasets()

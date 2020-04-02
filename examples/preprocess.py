# standard libary
import os
import json
from collections import defaultdict
import argparse
import glob
import subprocess
# third party libraries
import tqdm
# project libraries
from speech.utils import data_helpers, wave, convert


class Preprocessor(object):
    
    def __init__(self, dataset_dir:str, dataset_name:str, lexicon_path:str,
                        min_duration:float, max_duration:float):

        self.dataset_dir = dataset_dir
        self.lex_dict = data_helpers.lexicon_to_dict(lexicon_path, dataset_name.lower()) if lexicon_path !='' else None
        # list of tuples of audio_path and transcripts
        self.audio_trans=list()
        self.min_duration = min_duration
        self.max_duration = max_duration

    def process_datasets(self):
        raise NotImplementedError

    def collect_audio_transcripts(self):
        raise NotImplementedError
    
    def write_json(self, label_path:str):
        # filter the entries by the duration bounds and write file
        root, ext = os.path.splitext(label_path)
        json_path = root + os.path.extsep + "json"
        unknown_words = data_helpers.UnknownWords()
        with open(json_path, 'w') as fid:
            print("Writing files to label json")
            for sample in tqdm.tqdm(self.audio_trans):
                audio_path, transcript = sample
                if not os.path.exists(audio_path):
                    print(f"file {audio_path} does not exists")
                else:
                    base, raw_ext = os.path.splitext(audio_path)
                    wav_path = base + os.path.extsep + "wav"
                    if not os.path.exists(wav_path):
                        try:
                            convert.to_wave(audio_path, wav_path)
                        except subprocess.CalledProcessError:
                            # if the file can't be converted, skip the file by continuing
                            print(f"Error converting file: {audio_path}")
                            continue
                    dur = wave.wav_duration(wav_path)
                    if self.min_duration <= dur <= self.max_duration:
                        text = process_text(transcript, self.lex_dict, unknown_words, wav_path)
                        # if transcript has an unknown word, skip it
                        if unknown_words.has_unknown: 
                            continue
                        datum = {'text' : text,
                                'duration' : dur,
                                'audio' : wav_file}
                        json.dump(datum, fid)
                        fid.write("\n")
    
        set_name = os.path.basename(os.path.splitext(label_path)[0])
        print(f"saving unk-word-stats here: {unk_path}")
        unknown_words.process_save(unk_path, set_name)
    
    def process_text(self, transcript:str, lex_dict:dict, unknown_words, audio_path:str)->list:
        # allows for alphanumeric characters, space, and apostrophe
        accepted_char = '[^A-Za-z0-9 \']+'
        # filters out unaccepted characters, lowers the case, & splits into list
        transcript = re.sub(accepted_char, '', transcript).lower()
        # check that all punctuation (minus apostrophe) has been removed 
        punct_noapost = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
        for p in punct_noapost:
            if p in transcript: raise ValueError(f"unwanted punctuation: {p} in transcript")
        #assert any([p in transcript for p in punct_noapost]), "unwanted punctuation in transcript"
        transcript = transcript.split()
        # if there is a pronunciation dict, convert to phonemes
        if self.lex_dict is  not None:
            unknown_words.check_transcript(audio_path, transcript, self.lex_dict)
            phonemes = []
            for word in transcript:
                # TODO: I shouldn't need to include list() in get but dict is outputing None not []
                phonemes.extend(self.lex_dict.get(word, list()))
            transcript = phonemes

        return transcript

    
class CommonVoicePreprocessor(Preprocessor):
    def __init__(self):
        super(VoxforgePreprocessor, self).__init__()
        self.dataset_dict = {"dev": "dev.tsv",
                            "test": "test.tsv",
                            "train":"train.tsv",
                            "validated": "validated.tsv"}
    
    def process_datasets(self):
        for set_name, label_name in self.dataset_dict.items():
            label_path = os.path.join(self.dataset_dir, label_name)
            self.collect_audio_transcripts(label_path)
            self.write_json(label_path)
        unk_words_dir = os.path.join(self.dataset_dir, 
        unique_unknown_words(self.dataset_dir)
        

    def collect_audio_transcripts(self, label_path:str):
        
        # open the file and select only entries with desired accents
        accents = ['us', 'canada']
        print(f"Filtering files by accents: {accents}")
        with open(label_fn) as fid: 
            reader = list(csv.reader(fid, delimiter='\t'))
            # first line in reader is the header which equals:
            # ['client_id','path','sentence','up_votes','down_votes','age','gender','accent']
            is_header = True
            for line in reader:
                if is_header:
                    is_header=False
                    continue
                else: 
                    # filter by accent
                    if line[7] in accents:
                        self.audio_trans.append((line[1], line[2]))


class UnknownWords():

    def __init__(self):
        self.word_set:set = set()
        self.filename_dict:dict = dict()
        self.line_count:int = 0
        self.word_count:int = 0
        self.has_unknown= False

    def check_transcript(self, filename:str, text, word_phoneme_dict:dict):

        if type(text) == str:
            text = text.split()
        elif type(text) == list: 
            pass
        else: 
            raise(TypeError("input text is not string or list type"))
   
        self.line_count += 1
        self.word_count += len(text) - 1
        line_unk = [word for word in text if word_phoneme_dict[word]==UNK_WORD_TOKEN]
        #if line_unk is empty, has_unk is False
        self.has_unknown = bool(line_unk)
        if self.has_unknown:
            self.word_set.update(line_unk)
            self.filename_dict.update({filename: len(line_unk)})

    def process_save(self, save_path:str, set_name:str):
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
        
        stats_dir = os.path.join(save_path, "unk_word_stats")
        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir)

        stats_dict_fn = os.path.join(stats_dir, set_name+"_unk-words-stats.json")
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="creates a data json file")
    parser.add_argument("--dataset-dir", type=str,
        help="directory where common voice .tsv files are located.")
    parser.add_argument("--dataset-name", type=str,
        help="Name of dataset with capitalized first letter.")
    parser.add_argument("--lexicon-path", type=str, default='',
        help="path to pronunciation lexicon, if desired.")
    parser.add_argument("--min-duration", type=float,
        help="minimum audio duration in seconds")
    parser.add_argument("--max-duration", type=float,
        help="maximum audio duration in seconds")


    args = parser.parse_args()

    data_preprocessor = eval(args.dataset_name+"Preprocessor")
    data_preprocessor(args.dataset_dir, args.dataset_name, args.lexicon_path,
                        args.min_duration, args.max_duration)
from collections import defaultdict
import string
import re

def lexicon_to_dict(corpus_name):
    """This function reads the librispeech-lexicon.txt file which is a mapping of words in the
        librispeech corpus to phoneme labels and represents the file as a dictionary.
        The digit accents are removed from the file name. 
        Note: the librispeech-lexicon.txt file needs to be in the same directory as this file.
    """
    
    if corpus_name == 'librispeech'
        lexicon_path = "../librispeech/librispeech-lexicon.txt"
    elif corpus_name == "tedlium":
        lexicon_path = "../tedlium/TEDLIUM.152k.dic"

    lex_dict = defaultdict(lambda: "unk")
    with open(lexicon_path, 'r') as fid:
        lexicon = (l.strip().lower().split() for l in fid)
        for line in lexicon: 
            word = line[0]
            phones = line[1:]
            # remove the accent digit from the phone, string.digits = '0123456789'
            phones = clean_phonemes(phones, corpus_name)
            # the if-statement will ignore the second pronunciation (phone list)
            if lex_dict[word] == "unk":
                lex_dict[word] = phones

    return clean_dict(lex_dict, corpus_name)

def clean_phonemes(phonemes, corpus_name)

    if corpus_name == 'librispeech':
        return list(map(lambda x: x.rstrip(string.digits), phones))
    else:
        return phonemes

def clean_dict(lex_dict, corpus_name):
    
    if corpus_name == "tedlium":
        return {key: value for key, value in lex_dict.items() if not re.search("\(\d\)$", key)}
    else: 
        return lex_dict
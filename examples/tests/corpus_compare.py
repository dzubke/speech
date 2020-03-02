from speech.utils import data_helpers


def export_phones(lex_dict):
    """exports all words and phones in lex_dict into two separate txt files
    """

    phone_set = set()
    word_set = set()
    phones_filename = "librispeech_lexicon_phone_set.txt"
    words_filename = "librispeech_lexicon_word_set.txt"

    for word, phones in lex_dict.items():
        if word not in word_set:
            word_set.add(word)
        
        for phone in phones:
            if phone not in phone_set:
                phone_set.add(phone)

    with open(phones_filename, 'w') as fid:
        for phone in phone_set:
            fid.write(phone+'\n')

    with open(words_filename, 'w') as fid:
        for word in word_set:
            fid.write(word+'\n')



def check_phones():
    """This function compares the phonemes in the librispeech corpus with the phoneme labels in the 39-phonemes
    in the timit dataset outlined here: 
    https://www.semanticscholar.org/paper/Speaker-independent-phone-recognition-using-hidden-Lee-Hon/3034afcd45fc190ed71982828b77f6e4154bdc5c
    
    Discrepencies in the CMU-39 and timit-39 phoneme sets and the librispeech phonemes: 
     - included in CMU-39 but not timit-39:  ao, zh, 
     - included timit-39 but not CMU-39: dx, sil
    """
    # standard 39 phones in the timit used by awni dictionary
    timit_phones39 = set(['ae', 'ah', 'aa', 'aw', 'er', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'l', 'm', 'n', 'ng', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'sil'])
    cmu_phones = set(['aa', 'ae', 'ah', 'ao', 'aw', 'ay',  'b', 'ch', 'd', 'dh', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh'])
    print(f"length of timit_dict: {len(timit_phones39)}")
    librispeech_phones = set()
    
    # greating a set of the librispeech phones by looping over every phone list in the word_to_phoneme mapping
    for phones in word_phoneme_dict.values():
        # looping over every phone in the word pronunciation
        for phone in phones:
            if phone not in librispeech_phones:
                librispeech_phones.add(phone)

    print(f"phones in librispeech but not cmu: {librispeech_phones.difference(cmu_phones)}")
    print(f"phones in cmu but not librispeech: {cmu_phones.difference(librispeech_phones)}")
    print(f"phones in timit but not cmu: {timit_phones39.difference(cmu_phones)}")
    print(f"phones in cmubut not timit: {cmu_phones.difference(timit_phones39)}")

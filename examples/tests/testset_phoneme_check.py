# standard libraries
import argparse
import json

# project libraries
from corpus_compare import process_name, export_phones
from speech.utils import data_helpers


def main(lexicon_path:str, dataset_path:str):
    """
        this script checks to see if all of the phoneme labels in the
        dataset are included in the lexicon
        Arguments:
            lexicon_path (str): path to pronunciation lexicon
            dataset_path (str): path to datases
    """

    corpus_name = process_name(lexicon_path)
    lex_dict = data_helpers.lexicon_to_dict(lexicon_path, corpus_name)
    phoneme_set, word_set = export_phones(lex_dict)
    print(f"phoneme set length: {len(phoneme_set)}")
    unknown_phonemes = set()

    with open(dataset_path, 'r') as fid:
        dataset = [json.loads(l) for l in fid]
        for sample in dataset:
            for phoneme in sample['text']:
                if phoneme not in phoneme_set:
                    unknown_phonemes.add(phoneme)

    print(f"unknown phonemes: number: {len(unknown_phonemes)}, phonemes: {unknown_phonemes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Preprocess librispeech dataset.")

    parser.add_argument("lexicon_path", type=str,
        help="The path to the lexicon pronuciation dictionary.")

    parser.add_argument("dataset_path", type=str,
        help="The path to the dataset json")
    args = parser.parse_args()

    main(args.lexicon_path, args.dataset_path)

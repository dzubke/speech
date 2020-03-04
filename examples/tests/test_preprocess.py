import json
import argparse


DATA_PATH = "./"

def main(test_json_fn, ref_json_fn):
    """
        Arguments:
            test_json_fn (str): path to json to be tested
            ref_json_fn (str): path to reference json
    """

    with open(test_json_fn, 'r') as test_id:
        test_json = json.load(test_id)
        with open(ref_json_fn, 'r') as ref_id:
            ref_json = json.load(ref_id)

            for test_sample in test_json:
                for ref_sample in ref_json:
                    if ref_sample["audio"] == test_sample["audio"]:
                        assert ref_sample["text"] == test_sample["text"], f"test_json text of not equal"




if __name__ == "__main__":
    ## format of command is >>python preprocess.py <path_to_dataset> --use_phonemes <True/False> 
    # where the optional --use_phonemes argument is whether the labels will be phonemes (True) or words (False)
    parser = argparse.ArgumentParser(
            description="Preprocess librispeech dataset.")

    parser.add_argument("test_json",
        help="Output json to be tested.")

    parser.add_argument("ref_json",
        help="The correct reference json.")
    args = parser.parse_args()

    main(args.test_json, args.ref_json)
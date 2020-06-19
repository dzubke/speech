# standard library
import argparse
# project libraries
from speech.utils import data_helpers


def main(cmu_lex_fn:str, ted_lex_fn:str, libsp_lex_fn:str, lex_out_fn:str)->None:
    """
    This function creates a pronunciation dictionary for common voice
    from the pronunciation dict of cmudict, tedlium and librispeech.
    Arguments:
        cmu_lex_fn - str: filename to cmudict lexicon
        ted_lex_fn - str: filename to tedlium lexicon
        libsp_lex_fn - str: filename to librispeech lexicon
        lex_out_fn - str: filename where lexicon dictionary will be saved
    Returns:
        None
    """
    cmu_dict = data_helpers.lexicon_to_dict(cmu_lex_fn, corpus_name="cmudict")
    ted_dict = data_helpers.lexicon_to_dict(ted_lex_fn, corpus_name="tedlium")
    libsp_dict = data_helpers.lexicon_to_dict(libsp_lex_fn, corpus_name="librispeech")

    master_dict = data_helpers.create_lexicon(cmu_dict, ted_dict, libsp_dict, out_path=lex_out_fn)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Preprocess librispeech dataset.")
    parser.add_argument("--cmu_lex_fn", type=str,
        help="The path to the cmudict pronuciation dictionary.")
    parser.add_argument("--ted_lex_fn", type=str,
        help="The path to the tedlium pronuciation dictionary.")
    parser.add_argument("--libsp_lex_fn", type=str,
        help="The path to the librispeech pronuciation dictionary.")
    parser.add_argument("--lex_out_fn", type=str,
        help="The path where the output lexicon will be saved.")
    args = parser.parse_args()

    main(args.cmu_lex_fn, args.ted_lex_fn, args.libsp_lex_fn, args.lex_out_fn)
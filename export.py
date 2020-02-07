import torch
import argparse
import pickle
import matplotlib

def preproc_pickle():
    with open('/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/examples/librispeech/models/ctc_models/20200121/20200127/best_preproc.pyc', 'rb') as fid:
        preproc = pickle.load(fid)
        print(f"self.mean, self.std: {preproc.mean}, {preproc.std}")
        preproc_dict = {'mean':preproc.mean, 
                        'std': preproc.std, 
                        "_input_dim": preproc._input_dim, 
                        "start_and_end": preproc.start_and_end, 
                        "int_to_char": preproc.int_to_char,
                        "char_to_int": preproc.char_to_int
                        }


        with open('./20200121-0127_preproc_dict_pickle', 'wb') as fid:
            pickle.dump(preproc_dict, fid)


    with open('./20200121-0127_preproc_dict_pickle', 'rb') as fid:
        preproc = pickle.load(fid)    
        print(preproc)


def export_state_dict(model_in_path, params_out_path):
    model = torch.load(model_in_path, map_location=torch.device('cpu'))
    torch.save(model.state_dict(), params_out_path)

def main(model_path, params_path):
    
    export_state_dict(model_path, params_path)
    preproc_pickle()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("params_path")
    args = parser.parse_args()

    main(args.model_path, args.params_path)


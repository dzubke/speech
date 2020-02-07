import torch
import pickle
import matplotlib

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

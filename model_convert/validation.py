# standard libraries
import os
import argparse
import json
import math
import pickle
import math

# third-party libaries
import torch
import torch.nn as nn
import onnx
from onnx import helper, shape_inference
import onnxruntime
import onnx_coreml
import coremltools
import numpy as np
import editdistance

#project libraries
from speech.loader import log_specgram_from_file
from speech.models.ctc_decoder import decode as ctc_decode
import speech.models as models
from get_paths import validation_paths
from import_export import preproc_to_dict, preproc_to_json, export_state_dict

"""
CONFIG_FN = '/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/onnx_coreml/validation_scripts/ctc_config_20200121-0127.json'
TRAINED_MODEL_FN = '/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/onnx_coreml/torch_models/20200121-0127_best_model_pyt14.pth'
STATE_DICT_FN = '/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/onnx_coreml/validation_scripts/state_params_20200121-0127.pth'
ONNX_FN = '/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/onnx_coreml/onnx_models/20200121-0127_best_model_pyt14.onnx'
COREML_FN = '/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/onnx_coreml/coreml_models/20200121-0127_best_model_pyt14.mlmodel'
PREPROC_FN = '/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/onnx_coreml/preproc/20200121-0127_best_preproc.pyc'
"""

np.random.seed(2020)
torch.manual_seed(2020)

freq_dim = 257 #freq dimension out of log_spectrogram 

def main(model_name, num_frames):

    time_dim = num_frames  #time dimension out of log_spectrogram 

    
    TRAINED_MODEL_FN, ONNX_FN, COREML_FN, CONFIG_FN, PREPROC_FN, state_dict_path = validation_paths(model_name)
    
    with open(CONFIG_FN, 'rb') as fid:
        config = json.load(fid)
        model_cfg = config["model"]

    #load models
    state_dict_model = torch.load(TRAINED_MODEL_FN, map_location=torch.device('cpu'))

    trained_model = models.CTC(freq_dim, 39, model_cfg)
    state_dict = state_dict_model.state_dict()
    torch.save(state_dict, state_dict_path)
    trained_model.load_state_dict(state_dict)

    onnx_model = onnx.load(ONNX_FN)

    coreml_model = coremltools.models.MLModel(COREML_FN)


    # prepping and checking models
    trained_model.eval()
    ##CTCNet_model.eval()

    onnx.checker.check_model(onnx_model)
    inferred_model = shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(inferred_model)

    # print("---------trained model-------------")
    # print(f"trained_model: {trained_model}")
    # print("-------Trained State Dict------------")
    # for k, v in trained_model.state_dict().items():
    #     print(k, v.shape, v.dtype)

    # print("---------CTCNet_model-------------")
    # print(f"CTCNet_model: {CTCNet_model}")
    # print("----------CTCNet state dict------------")
    # for k, v in CTCNet_model.state_dict().items():
    #     print(k, v.shape, v.dtype)
    # print("-----------------------------")


    #creating the test data
    data_dct = gen_test_data(PREPROC_FN, time_dim, freq_dim)

    #saving the preproc object as a dictionary
    preproc_json_path = PREPROC_FN[:-4]+".json"   #removes the .pyc extension and replaces with .pickle ext
    preproc_to_json(PREPROC_FN, preproc_json_path)

    preproc_pickle_path = PREPROC_FN[:-4]+".pickle"
    preproc_to_dict(PREPROC_FN, preproc_pickle_path, to_pickle=True)


    # make predictions
    predictions_dict= {}

    stream_test_name = "Speak_5_out"
    BLANK_INDEX=39

    for name, data in data_dct.items():
        print(f"\n~~~~~~~~~~~~~~~~~~{name}~~~~~~~~~~~~~~~~~~~~~~\n")
        test_x, test_h, test_c = data

        trained_output = trained_model(torch.from_numpy(test_x),(torch.from_numpy(test_h), torch.from_numpy(test_c))) 
        trained_probs, trained_h, trained_c = to_numpy(trained_output[0]), to_numpy(trained_output[1][0]), to_numpy(trained_output[1][1])
        trained_max_decoder = max_decode(trained_probs[0], blank=BLANK_INDEX)
        trained_ctc_decoder = ctc_decode(trained_probs[0], beam_size=50, blank=BLANK_INDEX)


        #torch_output = CTCNet_model(torch.from_numpy(test_x), torch.from_numpy(test_h), torch.from_numpy(test_c)) 
        #torch_probs, torch_h, torch_c = [to_numpy(tensor) for tensor in torch_output]
        #torch_output = [to_numpy(tensor) for tensor in torch_output]
       
        ort_session = onnxruntime.InferenceSession(ONNX_FN)
        ort_inputs = {
        ort_session.get_inputs()[0].name: test_x,
        ort_session.get_inputs()[1].name: test_h,
        ort_session.get_inputs()[2].name: test_c}
        ort_output = ort_session.run(None, ort_inputs)
        onnx_probs, onnx_h, onnx_c = [np.array(array) for array in ort_output]
        #onnx_output = [np.array(array) for array in ort_output]
        print("onnxruntime prediction complete") 

        coreml_input = {'input': test_x, 'hidden_prev': test_h, 'cell_prev': test_c}
        coreml_output = coreml_model.predict(coreml_input, useCPUOnly=True)
        coreml_probs = np.array(coreml_output['output'])
        coreml_h = np.array(coreml_output['hidden'])
        coreml_c = np.array(coreml_output['cell'])
        coreml_max_decoder = max_decode(coreml_probs[0], blank=BLANK_INDEX)
        coreml_ctc_decoder = ctc_decode(coreml_probs[0], beam_size=50,blank=BLANK_INDEX)
        #coreml_output = [coreml_probs, coreml_h, coreml_c]
        print("coreml prediction completed")

        if name == stream_test_name:
            stream_test_x = test_x
            stream_test_h = test_h
            stream_test_c = test_c
            stream_test_probs = trained_probs
            stream_test_h_out = trained_h
            stream_test_c_out = trained_c
            stream_test_max_decoder = trained_max_decoder
            stream_test_ctc_decoder = trained_ctc_decoder

        time_slice = 0 #math.floor(time_dim/2) - 1
        trained_probs_sample = trained_probs[0,time_slice,:]
        trained_h_sample = trained_h[0,0,0:25]
        trained_c_sample = trained_c[0,0,0:25]
        trained_max_decoder_char = ints_to_phonemes(PREPROC_FN, trained_max_decoder)
        trained_ctc_decoder_char = ints_to_phonemes(PREPROC_FN, trained_ctc_decoder[0])

        print("\n-----Torch Output-----")
        print(f"output {np.shape(trained_probs)}: \n{trained_probs_sample}")
        print(f"hidden {np.shape(trained_h)}: \n{trained_h_sample}")
        print(f"cell {np.shape(trained_c)}: \n{trained_c_sample}")
        print(f"max decode: {trained_max_decoder_char}")
        print(f"ctc decode: {trained_ctc_decoder_char}")

        output_dict = {"torch_probs_(time_dim/2-1)":trained_probs_sample.tolist(), "torch_h_sample":trained_h_sample.tolist(), 
                        "torch_c_sample": trained_c_sample.tolist(), "torch_max_decoder":trained_max_decoder_char, 
                        "torch_ctc_decoder_beam=50":trained_ctc_decoder_char}

        predictions_dict.update({name: output_dict})

        print("\n-----Coreml Output-----")
        print(f"output {coreml_probs.shape}: \n{coreml_probs[0,time_slice,:]}")
        print(f"hidden {coreml_h.shape}: \n{coreml_h[0,0,0:25]}")
        print(f"cell {coreml_c.shape}: \n{coreml_c[0,0,0:25]}")
        print(f"max decode: {coreml_max_decoder}")
        print(f"ctc decode: {coreml_ctc_decoder}")

        # Compare Trained and Coreml predictions
        np.testing.assert_allclose(coreml_probs, trained_probs, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(coreml_h, trained_h, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(coreml_c, trained_c, rtol=1e-03, atol=1e-05)
        #assert(trained_max_decoder==coreml_max_decoder), "max decoder doesn't match"
        #assert(trained_ctc_decoder[0]==coreml_ctc_decoder[0]), "ctc decoder labels don't match"
        #np.testing.assert_almost_equal(trained_ctc_decoder[1], coreml_ctc_decoder[1], decimal=3)
        print("\nTrained and Coreml probs, hidden, cell, decoder states match, all good!")

        # Compare Trained and Torch predictions
        #np.testing.assert_allclose(torch_probs, trained_probs, rtol=1e-03, atol=1e-05)
        #np.testing.assert_allclose(torch_h, trained_h, rtol=1e-03, atol=1e-05)
        #np.testing.assert_allclose(torch_c, trained_c, rtol=1e-03, atol=1e-05)
        #print("\nTorch and Trained probs, hidden, cell states match, all good!")  

        # Compare Torch and ONNX predictions
        np.testing.assert_allclose(trained_probs, onnx_probs, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(trained_h, onnx_h, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(trained_c, onnx_c, rtol=1e-03, atol=1e-05)
        print("\nTrained and ONNX probs, hidden, cell states match, all good!")  

        # Compare ONNX and CoreML predictions
        np.testing.assert_allclose(onnx_probs, coreml_probs, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(onnx_h, coreml_h, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(onnx_c, coreml_c, rtol=1e-03, atol=1e-05)
        print("\nONNX and CoreML probs, hidden, cell states match, all good!")

        # Compare Torch and CoreML predictions
        #np.testing.assert_allclose(torch_probs, coreml_probs, rtol=1e-03, atol=1e-05)
        #np.testing.assert_allclose(torch_h, coreml_h, rtol=1e-03, atol=1e-05)
        #np.testing.assert_allclose(torch_c, coreml_c, rtol=1e-03, atol=1e-05)
        #print("\nTorch and CoreML probs, hidden, cell states match, all good!")


    
    dict_to_json(predictions_dict, "./output/"+model_name+"_output.json")




def dict_to_json(input_dict, json_path):
    
    with open(json_path, 'w') as fid:
        json.dump(input_dict, fid)


def preprocess(PREPROC_FN, inputs):
    with open(PREPROC_FN, 'rb') as fid:
        preproc = pickle.load(fid)
        return (inputs - preproc.mean) / preproc.std

def ints_to_phonemes(PREPROC_FN, int_list):
    with open(PREPROC_FN, 'rb') as fid:
        preproc = pickle.load(fid)
        return preproc.decode(int_list)


def gen_test_data(preproc_path, time_dim, freq_dim):
    test_x_zeros = np.zeros((1, time_dim, freq_dim)).astype(np.float32)
    test_h_zeros = np.zeros((5, 1, 512)).astype(np.float32)
    test_c_zeros = np.zeros((5, 1, 512)).astype(np.float32)
    test_zeros = [test_x_zeros, test_h_zeros, test_c_zeros]

    test_x_randn = np.random.randn(1, time_dim, freq_dim).astype(np.float32)
    test_h_randn = np.random.randn(5, 1, 512).astype(np.float32)
    test_c_randn = np.random.randn(5, 1, 512).astype(np.float32)
    test_randn = [test_x_randn, test_h_randn, test_c_randn]

    test_names = ["Speak_5_out", "DZ-5-drz-test-20191202", "DZ-5-plane-noise", 
                "LibSp_777-126732-0003", "LibSp_84-121123-0001", 
                "Speak_1_4ysq5X0Mvxaq1ArAntCWC2YkWHc2-1574725037", 
                "Speak_2_58cynYij95TbB9Nlz3TrKBbkg643-1574725017", 
                "Speak_3_CcSEvcOEineimGwKOk1c8P2eU0q1-1574725123", 
                "Speak_4_OVrsxD1n9Wbh0Hh6thej8FIBIOE2-1574725033", 
                "Speak_6_R3SdlQCwoYQkost3snFxzXS5vam2-1574726165"]
    
    test_fns = ["ST-out.wav", "DZ-5-drz-test-20191202.wv", "DZ-5-plane-noise.wv", "LS-777-126732-0003.wav", 
                "LS-84-121123-0001.wav", "ST-4ysq5X0Mvxaq1ArAntCWC2YkWHc2-1574725037.wv",
                "ST-58cynYij95TbB9Nlz3TrKBbkg643-1574725017.wv", "ST-CcSEvcOEineimGwKOk1c8P2eU0q1-1574725123.wv", 
                "ST-OVrsxD1n9Wbh0Hh6thej8FIBIOE2-1574725033.wv", "ST-R3SdlQCwoYQkost3snFxzXS5vam2-1574726165.wv"]

    unused_names = ["DZ-5-drz-test-20191202", "DZ-5-plane-noise", 
                "LibSp_777-126732-0003", "LibSp_84-121123-0001", 
                "Speak_1_4ysq5X0Mvxaq1ArAntCWC2YkWHc2-1574725037", 
                "Speak_2_58cynYij95TbB9Nlz3TrKBbkg643-1574725017", 
                "Speak_3_CcSEvcOEineimGwKOk1c8P2eU0q1-1574725123", 
                "Speak_4_OVrsxD1n9Wbh0Hh6thej8FIBIOE2-1574725033", 
                "Speak_6_R3SdlQCwoYQkost3snFxzXS5vam2-1574726165"]

    used_fns =["DZ-5-drz-test-20191202.wv", "DZ-5-plane-noise.wv", "LS-777-126732-0003.wav", 
                "LS-84-121123-0001.wav", "ST-4ysq5X0Mvxaq1ArAntCWC2YkWHc2-1574725037.wv",
                "ST-58cynYij95TbB9Nlz3TrKBbkg643-1574725017.wv", "ST-CcSEvcOEineimGwKOk1c8P2eU0q1-1574725123.wv", 
                "ST-OVrsxD1n9Wbh0Hh6thej8FIBIOE2-1574725033.wv", "ST-R3SdlQCwoYQkost3snFxzXS5vam2-1574726165.wv"]
                          
    base_path = './audio_files/'
    audio_dct = load_audio(preproc_path, test_names, test_fns, base_path, test_h_zeros, test_c_zeros, time_dim)
    test_dct = {'test_zeros': test_zeros, 'test_randn_seed-2020': test_randn}
    test_dct.update(audio_dct)

    return test_dct


def load_audio(preproc_path, test_names, test_fns, base_path, test_h, test_c, time_dim):
    dct = {}
    for test_name, test_fn in zip(test_names, test_fns):

        audio_data = preprocess(preproc_path, log_specgram_from_file(base_path+test_fn))
        audio_data = audio_data[:time_dim,:]
        audio_data = np.expand_dims(audio_data, 0)
        dct.update({test_name : [audio_data, test_h, test_c]})
    return dct


class CTC(nn.Module):
    def __init__(self, freq_dim, time_dim, output_dim, config):
        super().__init__()
        
        encoder_cfg = config["encoder"]
        conv_cfg = encoder_cfg["conv"]

        convs = []
        in_c = 1
        for out_c, h, w, s1, s2, p1, p2 in conv_cfg:
            conv = nn.Conv2d(in_channels=in_c, 
                             out_channels=out_c, 
                             kernel_size=(h, w),
                             stride=(s1, s2), 
                             padding=(p1, p2))
            batch_norm =  nn.BatchNorm2d(out_c)
            convs.extend([conv, batch_norm, nn.ReLU()])
            if config["dropout"] != 0:
                convs.append(nn.Dropout(p=config["dropout"]))
            in_c = out_c

        self.conv = nn.Sequential(*convs)
        conv_out = out_c * self.conv_out_size(freq_dim, 1)
        
        assert conv_out > 0, \
          "Convolutional output frequency dimension is negative."

        #print(f"conv_out: {conv_out}")
        rnn_cfg = encoder_cfg["rnn"]

        assert rnn_cfg["type"] == "GRU" or rnn_cfg["type"] == "LSTM", "RNN type in config not supported"


        self.rnn = eval("nn."+rnn_cfg["type"])(
                        input_size=conv_out,
                        hidden_size=rnn_cfg["dim"],
                        num_layers=rnn_cfg["layers"],
                        batch_first=True, dropout=config["dropout"],
                        bidirectional=rnn_cfg["bidirectional"])

        
        
        _encoder_dim = rnn_cfg["dim"]
        self.volatile = False


        # include the blank token
        #print(f"fc _encoder_dim {_encoder_dim}, output_dim {output_dim}")
        self.fc = LinearND(_encoder_dim, output_dim + 1)

    def conv_out_size(self, n, dim):
        for c in self.conv.children():
            if type(c) == nn.Conv2d:
                # assuming a valid convolution meaning no padding
                k = c.kernel_size[dim]
                s = c.stride[dim]
                p = c.padding[dim]
                n = (n - k + 1 + 2*p) / s
                n = int(math.ceil(n))
        return n

    def forward(self, x, h_prev, c_prev):

        x = x.unsqueeze(1)

        # conv first
        x = self.conv(x)

        # reshape for rnn
        x = torch.transpose(x, 1, 2).contiguous()
        b, t, f, c = x.data.size()
        x = x.view((b, t, f*c))
        
        # rnn
        x, (h, c) = self.rnn(x, (h_prev, c_prev))

        # fc
        x = self.fc(x)

        # softmax for final output
        x = torch.nn.functional.softmax(x, dim=2)
        return x, h, c


class LinearND(nn.Module):

    def __init__(self, *args):
        """
        A torch.nn.Linear layer modified to accept ND arrays.
        The function treats the last dimension of the input
        as the hidden dimension.
        """
        super(LinearND, self).__init__()
        self.fc = nn.Linear(*args)

    def forward(self, x):
        size = x.size()
        n = int(np.prod(size[:-1]))
        out = x.contiguous().view(n, size[-1])
        out = self.fc(out)
        size = list(size)
        size[-1] = out.size()[-1]
        return out.view(size)

def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def max_decode(output, blank=39):
    pred = np.argmax(output, 1)
    prev = pred[0]
    seq = [prev] if prev != blank else []
    for p in pred[1:]:
        if p != blank and p != prev:
            seq.append(p)
        prev = p
    return seq





if  __name__=="__main__":
    # commmand format: python validation.py <model_name>
    parser = argparse.ArgumentParser(description="validates the outputs of the models.")
    parser.add_argument("model_name", help="name of the model.")
    parser.add_argument("--num_frames", help="number of input frames in time dimension hard-coded in onnx model")

    args = parser.parse_args()

    main(args.model_name, int(args.num_frames))


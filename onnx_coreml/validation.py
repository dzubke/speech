# standard libraries
import os
import argparse
import json
import math
import pickle

# third-party libaries
import torch
import torch.nn as nn
import onnx
from onnx import helper, shape_inference
import onnxruntime
import onnx_coreml
import coremltools
import numpy as np

#project libraries
from speech.loader import log_specgram_from_file
from speech.models.ctc_decoder import decode as ctc_decode

CONFIG_FN = '/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/onnx_coreml/validation_scripts/ctc_config_20200121-0127.json'
TRAINED_MODEL_FN = '/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/onnx_coreml/torch_models/20200121-0127_best_model.pth'
STATE_DICT_FN = '/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/onnx_coreml/validation_scripts/state_params_20200121-0127.pth'
ONNX_FN = '/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/onnx_coreml/validation_scripts/CTCNet_2020-02-05.onnx'
COREML_FN = '/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/onnx_coreml/validation_scripts/CTCNet_2020-02-05.mlmodel'
PREPROC_FN = '/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/onnx_coreml/preproc/20200121-0127_best_preproc.pyc'

np.random.seed(2020)
torch.manual_seed(2020)

def main():
    with open(CONFIG_FN, 'rb') as fid:
        config = json.load(fid)
        model_cfg = config["model"]

    #load models
    trained_model = torch.load(TRAINED_MODEL_FN, map_location=torch.device('cpu'))

    CTCNet_model = CTC(161, 100, 40, model_cfg)
    state_dict = torch.load(STATE_DICT_FN)
    CTCNet_model.load_state_dict(state_dict)

    onnx_model = onnx.load(ONNX_FN)

    coreml_model = coremltools.models.MLModel(COREML_FN)


    # prepping and checking models
    trained_model.eval()
    CTCNet_model.eval()

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
    data_dct = gen_test_data()


    # make predictions
    for name, data in data_dct.items():
        print(f"\n~~~~~~~~~~~~~~~~~~{name}~~~~~~~~~~~~~~~~~~~~~~\n")
        test_x, test_h, test_c = data

        trained_output = trained_model(torch.from_numpy(test_x), (torch.from_numpy(test_h), torch.from_numpy(test_c))) 
        trained_probs, trained_h, trained_c = to_numpy(trained_output[0]), to_numpy(trained_output[1][0]), to_numpy(trained_output[1][1])
        trained_max_decoder = max_decode(trained_probs[0], blank=40)
        trained_ctc_decoder = ctc_decode(trained_probs[0], beam_size=50, blank=40)
    
        torch_output = CTCNet_model(torch.from_numpy(test_x), torch.from_numpy(test_h), torch.from_numpy(test_c)) 
        torch_probs, torch_h, torch_c = [to_numpy(tensor) for tensor in torch_output]
        #torch_output = [to_numpy(tensor) for tensor in torch_output]
        
        ort_session = onnxruntime.InferenceSession(ONNX_FN)
        ort_inputs = {
        ort_session.get_inputs()[0].name: test_x,
        ort_session.get_inputs()[1].name: test_h,
        ort_session.get_inputs()[2].name: test_c}
        ort_output = ort_session.run(None, ort_inputs)
        onnx_probs, onnx_h, onnx_c = [np.array(array) for array in ort_output]
        #onnx_output = [np.array(array) for array in ort_output]

        coreml_input = {'input': test_x, 'h_prev': test_h, 'c_prev': test_c}
        coreml_output = coreml_model.predict(coreml_input, useCPUOnly=True)
        coreml_probs = np.array(coreml_output['output'])
        coreml_h = np.array(coreml_output['hidden'])
        coreml_c = np.array(coreml_output['c'])
        coreml_max_decoder = max_decode(coreml_probs[0], blank=40)
        coreml_ctc_decoder = ctc_decode(coreml_probs[0], beam_size=50,blank=40)
        #coreml_output = [coreml_probs, coreml_h, coreml_c]

        print("\n-----Coreml Output-----")
        print(f"output {np.shape(coreml_probs)}: \n{coreml_probs[0,100,:]}")
        print(f"hidden {np.shape(coreml_h)}: \n{coreml_h[0,0,0:20]}")
        print(f"cell {np.shape(coreml_c)}: \n{coreml_c[0,0,0:20]}")
        print(f"max decode: {ints_to_phonemes(coreml_max_decoder)}")
        print(f"ctc decode: {ints_to_phonemes(coreml_ctc_decoder[0])}")

        # print("\n-----Coreml Output-----")
        # print(f"output {coreml_probs.shape}: \n{coreml_probs[0,100,:]}")
        # print(f"hidden {coreml_h.shape}: \n{coreml_h[0,0,0:20]}")
        # print(f"cell {coreml_c.shape}: \n{coreml_c[0,0,0:20]}")
        # print(f"max decode: {coreml_max_decoder}")
        # print(f"ctc decode: {coreml_ctc_decoder}")

        # Compare Trained and Coreml predictions
        np.testing.assert_allclose(coreml_probs, trained_probs, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(coreml_h, trained_h, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(coreml_c, trained_c, rtol=1e-03, atol=1e-05)
        assert(trained_max_decoder==coreml_max_decoder), "max decoder doesn't match"
        assert(trained_ctc_decoder[0]==coreml_ctc_decoder[0]), "ctc decoder labels don't match"
        np.testing.assert_almost_equal(trained_ctc_decoder[1], coreml_ctc_decoder[1], decimal=3)
        print("\nTrained and Coreml probs, hidden, cell, decoder states match, all good!")

        # Compare Trained and Torch predictions
        np.testing.assert_allclose(torch_probs, trained_probs, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(torch_h, trained_h, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(torch_c, trained_c, rtol=1e-03, atol=1e-05)
        print("\nTorch and Trained probs, hidden, cell states match, all good!")  

        # Compare Torch and ONNX predictions
        np.testing.assert_allclose(torch_probs, onnx_probs, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(torch_h, onnx_h, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(torch_c, onnx_c, rtol=1e-03, atol=1e-05)
        print("\nTorch and ONNX probs, hidden, cell states match, all good!")  

        # Compare ONNX and CoreML predictions
        np.testing.assert_allclose(onnx_probs, coreml_probs, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(onnx_h, coreml_h, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(onnx_c, coreml_c, rtol=1e-03, atol=1e-05)
        print("\nONNX and CoreML probs, hidden, cell states match, all good!")

        # Compare Torch and CoreML predictions
        np.testing.assert_allclose(torch_probs, coreml_probs, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(torch_h, coreml_h, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(torch_c, coreml_c, rtol=1e-03, atol=1e-05)
        print("\nTorch and CoreML probs, hidden, cell states match, all good!")


def preprocess(inputs):
    with open(PREPROC_FN, 'rb') as fid:
        preproc = pickle.load(fid)
        return (inputs - preproc.mean) / preproc.std

def ints_to_phonemes(int_list):
    with open(PREPROC_FN, 'rb') as fid:
        preproc = pickle.load(fid)
        return preproc.decode(int_list)


def gen_test_data():
    test_x_zeros = np.zeros((1, 396, 161)).astype(np.float32)
    test_h_zeros = np.zeros((5, 1, 512)).astype(np.float32)
    test_c_zeros = np.zeros((5, 1, 512)).astype(np.float32)
    test_zeros = [test_x_zeros, test_h_zeros, test_c_zeros]

    test_x_randn = np.random.randn(1, 396, 161).astype(np.float32)
    test_h_randn = np.random.randn(5, 1, 512).astype(np.float32)
    test_c_randn = np.random.randn(5, 1, 512).astype(np.float32)
    test_randn = [test_x_randn, test_h_randn, test_c_randn]

    test_names = ["DZ_clean", "DZ_noise", "LibSp_1", "LibSp_2", "Speak_1_4ysq", "Speak_2_58cyn", 
                "Speak_3_CcSEv", "Speak_4_OVrsx", "Speak_5_out", "Speak_6_R3Sdl"]
    
    test_fns = ["DZ-5-drz-test-20191202.wv", "DZ-5-plane-noise.wv", "LS-777-126732-0003.wav", 
                "LS-84-121123-0001.wav", "ST-4ysq5X0Mvxaq1ArAntCWC2YkWHc2-1574725037.wv",
                "ST-58cynYij95TbB9Nlz3TrKBbkg643-1574725017.wv", "ST-CcSEvcOEineimGwKOk1c8P2eU0q1-1574725123.wv", 
                "ST-OVrsxD1n9Wbh0Hh6thej8FIBIOE2-1574725033.wv", "ST-out.wav", 
                "ST-R3SdlQCwoYQkost3snFxzXS5vam2-1574726165.wv"]
                          
    base_path = './audio_files/'
    audio_dct = load_audio(test_names, test_fns, base_path, test_h_randn, test_c_randn)
    test_dct = {'test_zeros': test_zeros, 'test_randn': test_randn}
    test_dct.update(audio_dct)

    return test_dct


def load_audio(test_names, test_fns, base_path, test_h, test_c):
    dct = {}
    for test_name, test_fn in zip(test_names, test_fns):

        audio_data = preprocess(log_specgram_from_file(base_path+test_fn))
        audio_data = audio_data[:396,:]
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


def max_decode(output, blank=40):
    pred = np.argmax(output, 1)
    prev = pred[0]
    seq = [prev] if prev != blank else []
    for p in pred[1:]:
        if p != blank and p != prev:
            seq.append(p)
        prev = p
    return seq





if  __name__=="__main__":

    main()
    """
    parser = argparse.ArgumentParser(
        description="Eval a speech model.")

    parser.add_argument("config_path",
        help="A path to a stored model.")
    parser.add_argument("trained_model_path",
        help="A path to a stored model.")
    parser.add_argument("stat_dict_path",
        help="A path to a stored model.")

    args = parser.parse_args()
    

    main(model_cfg, args.trained_model_path, args.stat_dict_path)
    """
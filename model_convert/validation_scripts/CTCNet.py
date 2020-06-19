import pickle
import soundfile
import scipy.signal

import math
from collections import OrderedDict
import numpy as np
import json
from datetime import date
import torch
import torch.nn as nn
import onnx
from onnx import onnx_pb
from onnx import helper, shape_inference

import onnxruntime
import onnx_coreml
import coremltools

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

        print(f"conv_out: {conv_out}")
        rnn_cfg = encoder_cfg["rnn"]

        assert rnn_cfg["type"] == "GRU" or rnn_cfg["type"] == "LSTM", "RNN type in config not supported"


        self.rnn = eval("nn."+rnn_cfg["type"])(
                        input_size=conv_out,
                        hidden_size=rnn_cfg["dim"],
                        num_layers=rnn_cfg["layers"],
                        batch_first=True, dropout=config["dropout"],
                        bidirectional=rnn_cfg["bidirectional"])

        
        
        _encoder_dim = rnn_cfg["dim"]

        # include the blank token
        print(f"fc _encoder_dim {_encoder_dim}, output_dim {output_dim}")
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

def array_from_wave(file_name):
    audio, samp_rate = soundfile.read(file_name, dtype='int16')
    
    return audio, samp_rate

def wav_duration(file_name):
    audio, samp_rate = soundfile.read(file_name, dtype='int16')
    nframes = audio.shape[0]
    duration = nframes / samp_rate
    return duration

def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10, plot=False):
    nperseg = int(window_size * sample_rate / 1e3)
    noverlap = int(step_size * sample_rate / 1e3)
    f, t, spec = scipy.signal.spectrogram(audio,
                    fs=sample_rate,
                    window='hann',
                    nperseg=nperseg,
                    noverlap=noverlap,
                    detrend=False)
    return np.log(spec.T.astype(np.float32) + eps)

def max_decode(output, blank=40):
    pred = np.argmax(output, 1)
    prev = pred[0]
    seq = [prev] if prev != blank else []
    for p in pred[1:]:
        if p != blank and p != prev:
            seq.append(p)
        prev = p
    return seq

def main():
    print(f"torch version: {torch.__version__}")
    print(f"onnx version: {onnx.__version__}")
    print(f"onnx_coreml version: {onnx_coreml.__version__}")
    print(f"onnxruntime version: {onnxruntime.__version__}")
    print(f"coremltools version: {coremltools.__version__}")
    
    #np.set_printoptions(formatter={'float': '{: 0.5f}'.format}, threshold=99999, linewidth=300)

    # Run Torch model once
    # state_dict_filename = 'state_dict_20200115-0120.pth'
    state_dict_filename = 'state_params_20200121-0127.pth'
    # config_json_filename = 'ctc_config_20200115.json'
    config_json_filename = 'ctc_config_20200121-0127.json'


    with open(config_json_filename, 'r') as fid, open('20200121-0127_preproc_dict_pickle', 'rb') as preproc:
        current_date = date.today()
        name = f"CTCNet_{current_date}"

        config = json.load(fid)
        model_cfg = config["model"]
        model = CTC(161, 100, 40, model_cfg)

        state_dict = torch.load(state_dict_filename)
        for k, v in state_dict.items():
            print(k, v.shape, v.dtype)
        model.load_state_dict(state_dict)
        print("loaded state dict successfully!")

        model.eval()

        # Set up test audio, out.wav
        test_filename = './audio_files/out.wav'
        audio, sr = array_from_wave(test_filename)
        duration = wav_duration(test_filename)
        print(f"wav array, shape {np.shape(audio)}, samplerate {sr}, duration {duration} sec")

        # Convert to log spectrogram
        inputs = log_specgram(audio, sr)
        print(f"log specgram output shape {np.shape(inputs)}, {inputs[100,:]}") # (396, 161)

        # Normalize with batch mean and std using training preprocessor file
        preproc_dict = pickle.load(preproc)
        preproc_mean, preproc_std = preproc_dict['mean'], preproc_dict['std']
        inputs = (inputs - preproc_mean) / preproc_std

        # Unsqueeze to add batch_size=1 value at dim 0
        test_input = torch.FloatTensor(inputs).unsqueeze(0) # (1, 396, 161) (batch, time, freq)
        process(model, name, test_input)
        

def process(torch_model, name, test_input):
    # Export Torch to ONNX with dummy input
    # dummy_x = torch.randn(1, 1000, 161) # dummy input to model, shape (batch, time, freq)
    # dummy_h0 = torch.zeros(4, 1, 256) # h_0 of shape (num_layers * num_directions, batch, hidden_size):
    
    torch.manual_seed(2020)
    #test_input = torch.randn(1, 396, 161)
    test_h0 = torch.zeros(5, 1, 512)
    test_c0 = torch.zeros(5, 1, 512)
    dummy_input = (test_input, test_h0, test_c0)
    
    onnx_filename = f"{name}.onnx"

    torch.onnx.export(torch_model, dummy_input, onnx_filename,
                    # export_params=True,
                    # verbose=True, 
                    input_names=['input', 'h_prev', 'c_prev'],
                    output_names=['output', 'hidden', 'c'],
                    opset_version=9,
                    # dynamic_axes={'input': {1: 'time_dim'}, 'output': {1: 'time_dim'}},
                    do_constant_folding=True
                    )
    
    # Load and check new ONNX model
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)
    
    # print("----- ONNX printable graph -----")
    # print(onnx_model.graph.value_info)
    # print(onnx_model.graph.input)
    # print(onnx_model.graph.output)
    # print(onnx.helper.printable_graph(onnx_model.graph))
    # print("-------------------")

    inferred_model = shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(inferred_model)
    # print(inferred_model.graph.value_info)
    
    # Export to CoreML
    print("\n\nTrying to convert to CoreML...")
    mlmodel_filename = f"{name}.mlmodel"
    mlmodel = onnx_coreml.convert(model=onnx_model, 
        # add_custom_layers=True,
        minimum_ios_deployment_target='13')
    mlmodel.save(mlmodel_filename)
    print(f"\nCoreML model saved to: {mlmodel_filename}\n")
    
    # Predict with CoreML
    spec = mlmodel.get_spec()

    from coremltools.models.neural_network.printer import print_network_spec
    # print_network_spec(spec, style='coding')
    # print_network_spec(spec)

    # print(f"CoreML Spec: {spec}\n\n")
    # coremltools.models.utils.save_spec(spec, 'saved_from_spec.mlmodel')
    # converted_mlmodel = coremltools.models.MLModel(spec)
    # print(f"Spec converted back to mlmodel and saved again")

    # print("Visualizing spec...")
    # mlmodel.visualize_spec()

    # Testing ONNX output against Torch model
    
    
    # Predict with Torch

    torch_output, torch_h, torch_c = torch_model(test_input, test_h0, test_c0)
    torch_np_out = to_numpy(torch_output) # shape (time, output_dims)
    torch_np_h = to_numpy(torch_h)
    torch_np_c = to_numpy(torch_c)
    

    print("\n----- Test Torch -----")
    print(f"output {np.shape(torch_np_out)}: \n{torch_np_out[0,100,:]}")
    print(f"hidden {np.shape(torch_np_h)}: \n{torch_np_h[0,0,0:20]}")
    print(f"c {np.shape(torch_np_c)}: \n{torch_np_c[0,0,0:20]}")
    print(np.argmax(torch_np_out, 2))
    print(f"max decode {max_decode(torch_np_out[0])}")
    print(f"output sum: {np.sum(torch_np_out)}")

    # Predict with ONNX using onnxruntime
    ort_session = onnxruntime.InferenceSession(onnx_filename)
    ort_inputs = {
        ort_session.get_inputs()[0].name: to_numpy(test_input),
        ort_session.get_inputs()[1].name: to_numpy(test_h0),
        ort_session.get_inputs()[2].name: to_numpy(test_c0)
    }
    ort_out, ort_h, ort_c = ort_session.run(None, ort_inputs)
    ort_np_out = np.array(ort_out)
    ort_np_h = np.array(ort_h)
    ort_np_c = np.array(ort_c)
    
    print("\n----- Test ONNX -----")
    print(f"output {np.shape(ort_np_out)}: \n{ort_np_out[0,100,:]}")
    print(f"hidden {np.shape(ort_np_h)}: \n{ort_np_h[0,0,0:20]}")
    print(f"c {np.shape(ort_np_c)}: \n{ort_np_c[0,0,0:20]}")
    print(f"max decode {max_decode(ort_np_out[0])}")

    # Predict CoreML
    predict_input = {'input': to_numpy(test_input), 'h_prev': to_numpy(test_h0), 'c_prev': to_numpy(test_c0)}
    predictions = mlmodel.predict(predict_input, useCPUOnly=True)
    coreml_output = np.array(predictions['output'])
    coreml_h = np.array(predictions['hidden'])
    coreml_c = np.array(predictions['c'])
    
    print("\n----- Test CoreML -----")
    print(f"output {np.shape(coreml_output)}: \n{coreml_output[0,100,:]}")
    print(f"hidden {np.shape(coreml_h)}: \n{coreml_h[0,0,0:20]}")
    print(f"c {np.shape(coreml_c)}: \n{coreml_c[0,0,0:20]}")
    print(f"max decode {max_decode(coreml_output[0])}")

    # TRAINED_MODEL_FN = '/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/examples/librispeech/models/ctc_models/20200121/20200127/best_model'
    # trained_model = torch.load(TRAINED_MODEL_FN, map_location=torch.device('cpu'))
    # trained_output = trained_model(test_input, (test_h0, test_c0)) 
    # trained_probs = to_numpy(trained_output[0])
    # np.testing.assert_allclose(torch_np_out, trained_probs, rtol=1e-03, atol=1e-05)
    # print("\nTorch and ONNX outputs match, all good!")   



    # Compare Torch and ONNX predictions
    np.testing.assert_allclose(torch_np_out, ort_np_out, rtol=1e-03, atol=1e-05)
    print("\nTorch and ONNX outputs match, all good!")    

    np.testing.assert_allclose(torch_np_h, ort_np_h, rtol=1e-03, atol=1e-05)
    print("Torch and ONNX hidden states match, all good!")   

    np.testing.assert_allclose(torch_np_c, ort_np_c, rtol=1e-04, atol=1e-05)
    print("Torch and ONNX cell states match, all good!")  

    # Compare ONNX and CoreML predictions
    np.testing.assert_allclose(ort_np_out, coreml_output, rtol=1e-03, atol=1e-05)
    print("ONNX and CoreML outputs match, all good!")

    np.testing.assert_allclose(ort_np_h, coreml_h, rtol=1e-03, atol=1e-05)
    print("ONNX and CoreML hidden states match, all good!")

    np.testing.assert_allclose(ort_np_c, coreml_c, rtol=1e-03, atol=1e-05)
    print("ONNX and CoreML cell match, all good!")

    # Compare Torch and CoreML predictions
    np.testing.assert_allclose(torch_np_out, coreml_output, rtol=1e-03, atol=1e-05)
    print("Torch and CoreML outputs match, all good!")

    np.testing.assert_allclose(torch_np_h, coreml_h, rtol=1e-03, atol=1e-05)
    print("Torch and CoreML hidden states match, all good!")

    np.testing.assert_allclose(torch_np_c, coreml_c, rtol=1e-03, atol=1e-05)
    print("Torch and CoreML cell states match, all good!")

     

    

if __name__ == "__main__":
    main()

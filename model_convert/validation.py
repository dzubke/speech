# standard libraries
import argparse
from datetime import date
import json
import logging
import os
import pickle
# third-party libaries
import coremltools
import editdistance
import numpy as np
import onnx
from onnx import helper, shape_inference
import onnxruntime
import onnx_coreml
import torch
import torch.nn as nn 
#project libraries
from get_paths import validation_paths
from import_export import preproc_to_dict, preproc_to_json, export_state_dict
from speech.loader import log_specgram_from_data, log_specgram_from_file
from speech.models.ctc_decoder import decode as ctc_decode
from speech.models import ctc_model
from speech.utils.compat import normalize
from speech.utils.convert import to_numpy
from speech.utils.stream_utils import make_full_window
from speech.utils.wave import array_from_wave

# ----- logging format/setup -----------
set_linewidth=160
np.set_printoptions(linewidth=set_linewidth)
torch.set_printoptions(linewidth=set_linewidth)

log_filename = "logs_probs-hiddencell_2020-05-20.log"
log_level = 10
logging.basicConfig(filename=None, filemode='w', level=log_level)
# -----------------------------

np.random.seed(2020)
torch.manual_seed(2020)

def main(model_name, num_frames):

    model_fn, onnx_fn, coreml_fn, config_fn, preproc_fn, state_dict_path = validation_paths(model_name)
    
    with open(config_fn, 'rb') as fid:
        config = json.load(fid)
        model_cfg = config["model"]
    with open(preproc_fn, 'rb') as fid:
        preproc = pickle.load(fid)

    freq_dim = preproc.input_dim

    PARAMS = {
        "sample_rate": 16000,
        "feature_win_len": 512,
        "feature_win_step":256,
        "feature_size":257,
        "chunk_size": 46,
        "n_context": 15
    }
    PARAMS['stride'] = PARAMS['chunk_size'] - 2*PARAMS['n_context']

    logging.warning(f"PARAMS dict: {PARAMS}")
    
    #load models
    state_dict_model = torch.load(model_fn, map_location=torch.device('cpu'))
    model = ctc_model.CTC(preproc.input_dim, preproc.vocab_size, model_cfg)
    state_dict = state_dict_model.state_dict()
    torch.save(state_dict, state_dict_path)
    model.load_state_dict(state_dict)

    onnx_model = onnx.load(onnx_fn)

    coreml_model = coremltools.models.MLModel(coreml_fn)

    # prepping and checking models
    model.eval()
    onnx.checker.check_model(onnx_model)
    inferred_model = shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(inferred_model)

    #creating the test data
    data_dct = gen_test_data(preproc, num_frames, freq_dim)

    #saving the preproc object as a dictionary
    # TODO change preproc methods to use the python object
    preproc_dict = preproc_to_dict(preproc_fn, export=False)
    preproc_json_path = preproc_fn[:-4]+".json"   
    preproc_to_json(preproc_fn, preproc_json_path)

    # make predictions 

    audio_dir = '/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/model_convert/audio_files/Validatio-audio_2020-05-21'

    validate_all_models(model, onnx_fn, coreml_model, preproc, audio_dir, model_name, num_frames)

    validation_tests = full_audio_infer(model, preproc, PARAMS, audio_dir)
    
    write_output_json(PARAMS, preproc_dict, validation_tests, model_name)

def write_output_json(PARAMS:dict, preproc_dict:dict, validation_tests:dict, model_name:str, output_path:str=None):
    output_json = {
        "metadata": PARAMS, 
        "preproc": preproc_dict,
        "validation_tests":validation_tests
    }
    logging.info(f"metadata to log: {PARAMS}")
    logging.info(f"preproc to log: {preproc_dict}")

    if output_path is None:
        json_filename = model_name+"_metadata_"+str(date.today())+".json"
        output_path = os.path.join("output", json_filename)
    with open(output_path, 'w') as fid:
        json.dump(output_json, fid)

 
def full_audio_infer(model, preproc, PARAMS:dict, audio_dir)->dict:
    """
    conducts inference on all audio files in audio_dir and returns a dictionary
    of the probabilities and phoneme predictions
    """
    validation_tests=dict()

    for audio_file in os.listdir(audio_dir):
        hidden_in = torch.zeros((5, 1, 512), dtype=torch.float32)
        cell_in = torch.zeros((5, 1, 512), dtype=torch.float32)

        audio_path = os.path.join(audio_dir, audio_file)

        audio_data, samp_rate = array_from_wave(audio_path)
        assert PARAMS['sample_rate'] == samp_rate, "audio sample rate is not equal to default sample rate"

        audio_data = make_full_window(audio_data, PARAMS['feature_win_len'], PARAMS['feature_win_step'])
        features = log_specgram_from_data(audio_data, samp_rate)
        norm_features = normalize(preproc, features)
        # adds the batch dimension (1, time, 257)
        norm_features = np.expand_dims(norm_features, axis=0) 
        torch_input = torch.from_numpy(norm_features)
        
        # padding time dim, pads from the back: zero padding (0,0) to freq, 15 paddding (15,0) to time
        padding = (0, 0, 15, 0)
        padded_input = torch.nn.functional.pad(torch_input, padding, value=0)

        fill_chunk_remainder = (padded_input.shape[1] - PARAMS['chunk_size']) % PARAMS['stride']

        if fill_chunk_remainder != 0:
            fill_chunk_padding = PARAMS['stride'] - fill_chunk_remainder
            fill_chunk_pad = torch.zeros(1, fill_chunk_padding, PARAMS['feature_size'], dtype=torch.float32, requires_grad=False)
            padded_input = torch.cat((padded_input, fill_chunk_pad),dim=1)
        
        # process last chunk with stride of zeros
        final_chunk_pad = torch.zeros(1, PARAMS['stride'], PARAMS['feature_size'], dtype=torch.float32, requires_grad=False)
        padded_input = torch.cat((padded_input, final_chunk_pad),dim=1)

        model_output = model(padded_input, (hidden_in, cell_in))

        probs, (hidden_out, cell_out) = model_output
        probs = to_numpy(probs)
        int_labels = max_decode(probs[0], blank=39)
        predictions = preproc.decode(int_labels)

        validation_tests.update({audio_file: {"logits": probs[0].tolist(), "maxDecodePhonemes": predictions}})
        logging.info(f"probs dimension: {probs.shape}")
        logging.info(f"prediction len: {len(predictions)}")

    return validation_tests


def validate_all_models(torch_model, onnx_fn, coreml_model, preproc, audio_dir, model_name, num_frames)->None:
    BLANK_INDEX=39
    stream_test_name = "Speak-out.wav"
    predictions_dict= {}

    for audio_file in os.listdir(audio_dir):
        test_h = np.zeros((5, 1, 512)).astype(np.float32)
        test_c = np.zeros((5, 1, 512)).astype(np.float32)

        audio_path = os.path.join(audio_dir, audio_file)
        log_spec = log_specgram_from_file(audio_path)
        features = normalize(preproc, log_spec)
        features = features[:num_frames,:]
        test_x = np.expand_dims(features, 0)
        logging.debug(f"\n~~~~~~~~~~~~~~~~~~{audio_file}~~~~~~~~~~~~~~~~~~~~~~\n")

        torch_output = torch_model(torch.from_numpy(test_x),(torch.from_numpy(test_h), torch.from_numpy(test_c))) 
        torch_probs, torch_h, torch_c = to_numpy(torch_output[0]), to_numpy(torch_output[1][0]), to_numpy(torch_output[1][1])
        torch_max_decoder = max_decode(torch_probs[0], blank=BLANK_INDEX)
        torch_ctc_decoder = ctc_decode(torch_probs[0], beam_size=50, blank=BLANK_INDEX)
       
        ort_session = onnxruntime.InferenceSession(onnx_fn)
        ort_inputs = {
            ort_session.get_inputs()[0].name: test_x,
            ort_session.get_inputs()[1].name: test_h,
            ort_session.get_inputs()[2].name: test_c
        }
        ort_output = ort_session.run(None, ort_inputs)
        onnx_probs, onnx_h, onnx_c = [np.array(array) for array in ort_output]
        logging.debug("onnxruntime prediction complete") 

        coreml_input = {'input': test_x, 'hidden_prev': test_h, 'cell_prev': test_c}
        coreml_output = coreml_model.predict(coreml_input, useCPUOnly=True)
        coreml_probs = np.array(coreml_output['output'])
        coreml_h = np.array(coreml_output['hidden'])
        coreml_c = np.array(coreml_output['cell'])
        coreml_max_decoder = max_decode(coreml_probs[0], blank=BLANK_INDEX)
        coreml_ctc_decoder = ctc_decode(coreml_probs[0], beam_size=50,blank=BLANK_INDEX)
        logging.debug("coreml prediction completed")

        if audio_file == stream_test_name:
            stream_test_x = test_x
            stream_test_h = test_h
            stream_test_c = test_c
            stream_test_probs = torch_probs
            stream_test_h_out = torch_h
            stream_test_c_out = torch_c
            stream_test_max_decoder = torch_max_decoder
            stream_test_ctc_decoder = torch_ctc_decoder

        time_slice = 0 #num_frames//2 - 1
        torch_probs_sample = torch_probs[0,time_slice,:]
        torch_h_sample = torch_h[0,0,0:25]
        torch_c_sample = torch_c[0,0,0:25]
        torch_max_decoder_char = preproc.decode(torch_max_decoder)
        torch_ctc_decoder_char = preproc.decode(torch_ctc_decoder[0])

        logging.debug("\n-----Torch Output-----")
        logging.debug(f"output {np.shape(torch_probs)}: \n{torch_probs_sample}")
        logging.debug(f"hidden {np.shape(torch_h)}: \n{torch_h_sample}")
        logging.debug(f"cell {np.shape(torch_c)}: \n{torch_c_sample}")
        logging.debug(f"max decode: {torch_max_decoder_char}")
        logging.debug(f"ctc decode: {torch_ctc_decoder_char}")

        output_dict = {"torch_probs_(num_frames/2-1)":torch_probs_sample.tolist(), "torch_h_sample":torch_h_sample.tolist(), 
                        "torch_c_sample": torch_c_sample.tolist(), "torch_max_decoder":torch_max_decoder_char, 
                        "torch_ctc_decoder_beam=50":torch_ctc_decoder_char}

        predictions_dict.update({audio_file: output_dict})

        logging.debug("\n-----Coreml Output-----")
        logging.debug(f"output {coreml_probs.shape}: \n{coreml_probs[0,time_slice,:]}")
        logging.debug(f"hidden {coreml_h.shape}: \n{coreml_h[0,0,0:25]}")
        logging.debug(f"cell {coreml_c.shape}: \n{coreml_c[0,0,0:25]}")
        logging.debug(f"max decode: {coreml_max_decoder}")
        logging.debug(f"ctc decode: {coreml_ctc_decoder}")

        # Compare torch and Coreml predictions
        np.testing.assert_allclose(coreml_probs, torch_probs, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(coreml_h, torch_h, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(coreml_c, torch_c, rtol=1e-03, atol=1e-05)
        #assert(torch_max_decoder==coreml_max_decoder), "max decoder doesn't match"
        #assert(torch_ctc_decoder[0]==coreml_ctc_decoder[0]), "ctc decoder labels don't match"
        #np.testing.assert_almost_equal(torch_ctc_decoder[1], coreml_ctc_decoder[1], decimal=3)
        logging.debug("\ntorch and Coreml probs, hidden, cell, decoder states match, all good!")

        # Compare Torch and ONNX predictions
        np.testing.assert_allclose(torch_probs, onnx_probs, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(torch_h, onnx_h, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(torch_c, onnx_c, rtol=1e-03, atol=1e-05)
        logging.debug("\ntorch and ONNX probs, hidden, cell states match, all good!")  

        # Compare ONNX and CoreML predictions
        np.testing.assert_allclose(onnx_probs, coreml_probs, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(onnx_h, coreml_h, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(onnx_c, coreml_c, rtol=1e-03, atol=1e-05)
        logging.debug("\nONNX and CoreML probs, hidden, cell states match, all good!")

    
    dict_to_json(predictions_dict, "./output/"+model_name+"_output.json")


def dict_to_json(input_dict, json_path):
    
    with open(json_path, 'w') as fid:
        json.dump(input_dict, fid)


def gen_test_data(preproc, num_frames, freq_dim):
    test_x_zeros = np.zeros((1, num_frames, freq_dim)).astype(np.float32)
    test_h_zeros = np.zeros((5, 1, 512)).astype(np.float32)
    test_c_zeros = np.zeros((5, 1, 512)).astype(np.float32)
    test_zeros = [test_x_zeros, test_h_zeros, test_c_zeros]

    test_x_randn = np.random.randn(1, num_frames, freq_dim).astype(np.float32)
    test_h_randn = np.random.randn(5, 1, 512).astype(np.float32)
    test_c_randn = np.random.randn(5, 1, 512).astype(np.float32)
    test_randn = [test_x_randn, test_h_randn, test_c_randn]

    test_names = ["Speak_5_out", "Dustin-5-drz-test-20191202", "Dustin-5-plane-noise", 
                "LibSp_777-126732-0003", "LibSp_84-121123-0001", 
                "Speak_1_4ysq5X0Mvxaq1ArAntCWC2YkWHc2-1574725037", 
                "Speak_2_58cynYij95TbB9Nlz3TrKBbkg643-1574725017", 
                "Speak_3_CcSEvcOEineimGwKOk1c8P2eU0q1-1574725123", 
                "Speak_4_OVrsxD1n9Wbh0Hh6thej8FIBIOE2-1574725033", 
                "Speak_6_R3SdlQCwoYQkost3snFxzXS5vam2-1574726165"]
    
    test_fns = ["Speak-out.wav", "Dustin-5-drz-test-20191202.wav", "Dustin-5-plane-noise.wav", "Librispeech-777-126732-0003.wav", 
                "Librispeech-84-121123-0001.wav", "Speak-4ysq5X0Mvxaq1ArAntCWC2YkWHc2-1574725037.wav",
                "Speak-58cynYij95TbB9Nlz3TrKBbkg643-1574725017.wav", "Speak-CcSEvcOEineimGwKOk1c8P2eU0q1-1574725123.wav", 
                "Speak-OVrsxD1n9Wbh0Hh6thej8FIBIOE2-1574725033.wav", "Speak-R3SdlQCwoYQkost3snFxzXS5vam2-1574726165.wav"]

    unused_names = ["Dustin-5-drz-test-20191202", "Dustin-5-plane-noise", 
                "LibSp_777-126732-0003", "LibSp_84-121123-0001", 
                "Speak_1_4ysq5X0Mvxaq1ArAntCWC2YkWHc2-1574725037", 
                "Speak_2_58cynYij95TbB9Nlz3TrKBbkg643-1574725017", 
                "Speak_3_CcSEvcOEineimGwKOk1c8P2eU0q1-1574725123", 
                "Speak_4_OVrsxD1n9Wbh0Hh6thej8FIBIOE2-1574725033", 
                "Speak_6_R3SdlQCwoYQkost3snFxzXS5vam2-1574726165"]

    used_fns =["Dustin-5-drz-test-20191202.wav", "Dustin-5-plane-noise.wav", "Librispeech-777-126732-0003.wav", 
                "Librispeech-84-121123-0001.wav", "Speak-4ysq5X0Mvxaq1ArAntCWC2YkWHc2-1574725037.wav",
                "Speak-58cynYij95TbB9Nlz3TrKBbkg643-1574725017.wav", "Speak-CcSEvcOEineimGwKOk1c8P2eU0q1-1574725123.wav", 
                "Speak-OVrsxD1n9Wbh0Hh6thej8FIBIOE2-1574725033.wav", "Speak-R3SdlQCwoYQkost3snFxzXS5vam2-1574726165.wav"]
                          
    base_path = './audio_files/'
    audio_dct = load_audio(preproc, test_names, test_fns, base_path, test_h_zeros, test_c_zeros, num_frames)
    test_dct = {'test_zeros': test_zeros, 'test_randn_seed-2020': test_randn}
    test_dct.update(audio_dct)

    return test_dct


def load_audio(preproc, test_names, test_fns, base_path, test_h, test_c, num_frames):
    dct = {}
    for test_name, test_fn in zip(test_names, test_fns):

        audio_data = normalize(preproc, log_specgram_from_file(base_path+test_fn))
        audio_data = audio_data[:num_frames,:]
        audio_data = np.expand_dims(audio_data, 0)
        dct.update({test_name : [audio_data, test_h, test_c]})
    return dct


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


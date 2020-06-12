# standard libraries
import time, logging
from datetime import datetime
import threading, collections, queue, os, os.path, json
# third-party libraries
import numpy as np
import pyaudio
import wave
from scipy import signal
import matplotlib.pyplot as plt
import torch
import editdistance as ed
# project libraries
import speech
from speech.utils.convert import to_numpy
from speech.models.ctc_model import CTC
from speech.loader import log_specgram_from_data, log_specgram_from_file
from speech.models.ctc_decoder import decode as ctc_decode
from speech.utils.compat import normalize
from speech.utils.wave import wav_duration, array_from_wave
from speech.utils.stream_utils import make_full_window

set_linewidth=160
np.set_printoptions(linewidth=set_linewidth)
torch.set_printoptions(linewidth=set_linewidth)

log_filename = "logs_probs-hiddencell_2020-05-20.log"
log_level = 30
logging.basicConfig(filename=None, filemode='w', level=log_level)
log_sample_len = 50     # number of data samples outputted to the log


def main(ARGS):

    print('Initializing model...')
    state_dict_model, preproc = speech.load(ARGS.model, tag='best')
    
    with open(ARGS.config, 'r') as fid:
        config = json.load(fid)
        model_config = config["model"]

    model = CTC(preproc.input_dim, preproc.vocab_size, model_config)
    state_dict = state_dict_model.state_dict()
    model.load_state_dict(state_dict)
    model.eval()

    #initial states for LSTM layers
    hidden_in = torch.zeros((5, 1, 512), dtype=torch.float32)
    cell_in   = torch.zeros((5, 1, 512), dtype=torch.float32)
    lstm_states = (hidden_in, cell_in)

    PARAMS = {
        "chunk_size": 46,
        "n_context": 15,
        "feature_window": 512,
        "feature_step":256,
        "feature_size":257,
        "initial_padding":15,
        "final_padding":0,
        'fill_chunk_padding':7  #TODO hard-coded value that is calculatedd as fill_chunk_padding
    }
    PARAMS['stride'] = PARAMS['chunk_size'] - 2*PARAMS['n_context']

    logging.warning(f"PARAMS dict: {PARAMS}")

    stream_probs, stream_preds = stream_infer(model, preproc, lstm_states, PARAMS, ARGS)

    lc_probs, lc_preds = list_chunk_infer_full_chunks(model, preproc, lstm_states, PARAMS, ARGS)

    fa_probs, fa_preds = full_audio_infer(model, preproc, lstm_states, PARAMS, ARGS)

    np.testing.assert_allclose(stream_probs, lc_probs, rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(stream_probs, fa_probs, rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(lc_probs, fa_probs, rtol=1e-03, atol=1e-05)

    assert ed.eval(stream_preds, lc_preds)==0, "stream and list-chunk predictions are not the same"
    assert ed.eval(stream_preds, fa_preds)==0, "stream and full-audio predictions are not the same"
    assert ed.eval(lc_preds, fa_preds)==0, "list-chunk and full-audio predictions are not the same"

    logging.warning(f"all probabilities and predictions are the same")


def stream_infer(model, preproc, lstm_states, PARAMS:dict, ARGS)->tuple:
    """
    Performs streaming inference of an input wav file (if provided in ARGS) or from
    the micropohone. Inference is performed my model and the preproc preprocessing
    object performs normalization.
    """
    begin_time = time.time()

    # Start audio with VAD
    audio = Audio(device=ARGS.device, input_rate=ARGS.rate, file=ARGS.file)
    frames = audio.frame_generator()

    print("Listening (ctrl-C to exit)...")
    logging.warning(f"--- starting stream_infer  ---")

    hidden_in, cell_in = lstm_states
    wav_data = bytearray()
    audio_buffer_size = 2   # two 16 ms steps in the features window
    stride_counter = 0      # used to stride the feature_buffer
    features_buffer_size = PARAMS['chunk_size']
    audio_ring_buffer = collections.deque(maxlen=audio_buffer_size)
    features_ring_buffer = collections.deque(maxlen=features_buffer_size)
    # add n_context zero frames as padding to the buffer
    zero_frame = np.zeros((1,PARAMS['feature_size']), dtype=np.float32)
    for _ in range(PARAMS['n_context']): features_ring_buffer.append(zero_frame)

    predictions = list()
    probs_list  = list()
    frames_per_block = round( audio.RATE_PROCESS/ audio.BLOCKS_PER_SECOND * 2) 

    # -------time evaluation variables-----------
    audio_buffer_time, audio_buffer_count = 0.0, 0 
    numpy_buffer_time, numpy_buffer_count = 0.0, 0 
    features_time, features_count = 0.0, 0
    normalize_time, normalize_count = 0.0, 0 
    features_buffer_time, features_buffer_count = 0.0, 0
    numpy_conv_time, numpy_conv_count = 0.0, 0
    model_infer_time, model_infer_count = 0.0, 0 
    output_assign_time, output_assign_count = 0.0, 0
    decoder_time, decoder_count = 0.0, 0
    total_time, total_count = 0.0, 0 
    # -------------------------------------------

    # ------------ logging ----------------------
    logging.warning(ARGS)
    logging.warning(model)
    logging.warning(preproc)
    logging.warning(f"audio_ring_buffer size: {audio_buffer_size}")
    logging.warning(f"feature_ring_buffer size: {features_buffer_size}")
    # -------------------------------------------

    try:
        total_time_start = time.time()
        for count, frame in enumerate(frames):
            logging.debug(f"----------iteration {count}------------")

            # exit the loop if there are no more full input frames
            if len(frame) <  frames_per_block:
                logging.warning(f"final sample length {len(frame)}")
                final_sample = frame
                break

            # ------------ logging ---------------
            logging.info(f"sample length: {len(frame)}")
            logging.info(f"audio_buffer length: {len(audio_ring_buffer)}")
            #logging.debug(f"iter {count}: first {log_sample_len} raw audio buffer values added to audio_ring_buffer: {frame[:log_sample_len]}")
            # ------------ logging ---------------

            # fill up the audio_ring_buffer and then feed into the model
            if len(audio_ring_buffer) < audio_buffer_size-1:
                # note: appending new frame to right of the buffer
                audio_buffer_time_start = time.time()
                audio_ring_buffer.append(frame)
                audio_buffer_time += time.time() - audio_buffer_time_start
                audio_buffer_count += 1
            else: 
                audio_buffer_time_start = time.time()
                audio_ring_buffer.append(frame)
                audio_buffer_time += time.time() - audio_buffer_time_start
                audio_buffer_count += 1
                
                numpy_buffer_time_start = time.time()
                buffer_list = list(audio_ring_buffer)
                # convert the buffer to numpy array
                # a single frame has dims: (512,) and numpy buffer (2 frames) is: (512,)
                # The dimension of numpy buffer is reduced by half because each 
                # integer in numpy buffer is encoded as 2 hexidecimal entries in frame
                numpy_buffer = np.concatenate(
                    (np.frombuffer(buffer_list[0], np.int16), 
                    np.frombuffer(buffer_list[1], np.int16)))
                # calculate the features with dim: (1, 257)
                numpy_buffer_time += time.time() - numpy_buffer_time_start
                numpy_buffer_count += 1

                features_time_start = time.time()
                features_step = log_specgram_from_data(numpy_buffer, samp_rate=16000)
                features_time += time.time() - features_time_start
                features_count += 1
                
                normalize_time_start = time.time()
                norm_features = normalize(preproc, features_step)
                normalize_time += time.time() - normalize_time_start
                normalize_count += 1

                # ------------ logging ---------------
                logging.info(f"audio integers shape: {numpy_buffer.shape}")  
                #logging.debug(f"iter {count}: first {log_sample_len} input audio samples {numpy_buffer.shape}: \n {numpy_buffer[:log_sample_len]}")
                logging.info(f"features_step shape: {features_step.shape}")
                #logging.debug(f"iter {count}: log_spec frame (all 257 values) {features_step.shape}:\n {features_step}")
                logging.info(f"features_buffer length: {len(features_ring_buffer)}")
                #logging.debug(f"iter {count}: normalized log_spec (all 257 values) {norm_features.shape}:\n {norm_features[0,:log_sample_len]}")
                logging.info(f"stride modulus: {stride_counter % PARAMS['stride']}")
                # ------------ logging ---------------

                # fill up the features_ring_buffer and then feed into the model
                if len(features_ring_buffer) < features_buffer_size-1:
                    features_buffer_time_start = time.time()
                    features_ring_buffer.append(norm_features)
                    features_buffer_time += time.time() - features_buffer_time_start
                    features_buffer_count += 1
                else:
                    if stride_counter % PARAMS['stride'] !=0:
                        features_ring_buffer.append(norm_features)
                        stride_counter += 1
                    else:
                        stride_counter += 1
                        features_buffer_time_start = time.time()
                        features_ring_buffer.append(norm_features)
                        features_buffer_time += time.time() - features_buffer_time_start
                        features_buffer_count += 1

                        numpy_conv_time_start = time.time()
                        # conv_context dim: (31, 257)
                        conv_context = np.concatenate(list(features_ring_buffer), axis=0)
                        # addding batch dimension: (1, 31, 257)
                        conv_context = np.expand_dims(conv_context, axis=0)
                        numpy_conv_time += time.time() - numpy_conv_time_start
                        numpy_conv_count += 1

                        model_infer_time_start = time.time()
                        logging.debug(f"iter {count}: first {log_sample_len} of input: {conv_context.shape}\n {conv_context[0, 0, :log_sample_len]}")                        
                        logging.debug(f"iter {count}: first {log_sample_len} of hidden_in first layer: {hidden_in.shape}\n {hidden_in[0, :, :log_sample_len]}")
                        logging.debug(f"iter {count}: first {log_sample_len} of cell_in first layer: {cell_in.shape}\n {cell_in[0, :, :log_sample_len]}")
                        model_out = model(torch.from_numpy(conv_context), (hidden_in, cell_in))
                        model_infer_time += time.time() - model_infer_time_start
                        model_infer_count += 1

                        output_assign_time_start = time.time()
                        probs, (hidden_out, cell_out) = model_out
                        logging.debug(f"iter {count}: first {log_sample_len} of prob output {probs.shape}:\n {probs[0, 0, :log_sample_len]}")                        
                        logging.debug(f"iter {count}: first {log_sample_len} of hidden_out first layer {hidden_out.shape}:\n {hidden_out[0, :, :log_sample_len]}")
                        logging.debug(f"iter {count}: first {log_sample_len} of cell_out first layer {cell_out.shape}:\n {cell_out[0, :, :log_sample_len]}")                        
                        # probs dim: (1, 1, 40)
                        probs = to_numpy(probs)
                        probs_list.append(probs)
                        hidden_in, cell_in = hidden_out, cell_out
                        output_assign_time += time.time() - output_assign_time_start
                        output_assign_count += 1

                        # ------------ logging ---------------
                        logging.info(f"conv_context shape: {conv_context.shape}")
                        logging.info(f"probs shape: {probs.shape}")
                        logging.info(f"probs_list len: {len(probs_list)}")
                        #logging.info(f"probs value: {probs}")
                        # ------------ logging ---------------
                
                        # decoding every 20 time-steps
                        #if count%20 ==0 and count!=0:
                        decoder_time_start = time.time()
                        probs_steps = np.concatenate(probs_list, axis=1)
                        int_labels = max_decode(probs_steps[0], blank=39)
                        # int_labels, likelihood = ctc_decode(probs[0], beam_size=50, blank=39)
                        predictions = preproc.decode(int_labels)
                        decoder_time += time.time() - decoder_time_start
                        decoder_count += 1
                        
                        # ------------ logging ---------------
                        logging.warning(f"predictions: {predictions}")
                        # ------------ logging ---------------
                        
                    total_count += 1
   
            if ARGS.savewav: wav_data.extend(frame)
        

    except KeyboardInterrupt:
        pass
    finally:
        # IN THE FINALLY BLOCK
        # if frames is empty
        if not next(frames):
            logging.info(f"---------- procerssing final sample in audio buffer ------------")
            zero_byte = b'\x00'
            num_missing_bytes = PARAMS['feature_step']*2 - len(final_sample)
            final_sample += zero_byte * num_missing_bytes
            audio_ring_buffer.append(final_sample)
            buffer_list = list(audio_ring_buffer)
            numpy_buffer = np.concatenate(
                (np.frombuffer(buffer_list[0], np.int16), 
                np.frombuffer(buffer_list[1], np.int16)))

            features_step = log_specgram_from_data(numpy_buffer, samp_rate=16000)
            norm_features = normalize(preproc, features_step)
            

            # --------logging ------------
            # logging.warning(f"final sample length 2: {len(final_sample)}")     
            logging.warning(f"numpy_buffer shape: {len(numpy_buffer)}")
            # logging.warning(f"audio_buffer 1 length: {len(buffer_list[0])}")
            # logging.warning(f"audio_buffer 2 length: {len(buffer_list[1])}")
            #logging.debug(f"iter {count}: first {log_sample_len} input audio samples {numpy_buffer.shape}: \n {numpy_buffer[:log_sample_len]}")
            logging.warning(f"features_step shape: {features_step.shape}")
            #logging.debug(f"iter {count}: log_spec frame (all 257 values) {features_step.shape}:\n {features_step}")
            #logging.debug(f"iter {count}: normalized log_spec (all 257 values) {norm_features.shape}:\n {norm_features[0,:log_sample_len]}")
            logging.warning(f"features_buffer length: {len(features_ring_buffer)}")
            logging.warning(f"stride modulus: {stride_counter % PARAMS['stride']}")
            # --------logging ------------

            if stride_counter % PARAMS['stride'] !=0:
                features_ring_buffer.append(norm_features)
                stride_counter += 1
            else:
                features_ring_buffer.append(norm_features)
                stride_counter += 1
                model_out = model(torch.from_numpy(conv_context), (hidden_in, cell_in))
                probs, (hidden_out, cell_out) = model_out
                logging.debug(f"iter {count}: first {log_sample_len} of prob output {probs.shape}:\n {probs[0, 0, :log_sample_len]}")                        
                logging.debug(f"iter {count}: first {log_sample_len} of hidden_out first layer {hidden_out.shape}:\n {hidden_out[0, :, :log_sample_len]}")
                logging.debug(f"iter {count}: first {log_sample_len} of cell_out first layer {cell_out.shape}:\n {cell_out[0, :, :log_sample_len]}")                 
                probs = to_numpy(probs)
                probs_list.append(probs)
            
            padding_iterations = PARAMS["final_padding"]+PARAMS['fill_chunk_padding']+PARAMS['stride']
            for count, frame in enumerate(range(padding_iterations)):
                logging.debug(f"---------- adding zeros at the end of audio sample ------------")

                # -------------logging ----------------
                logging.info(f"stride modulus: {stride_counter % PARAMS['stride']}")
                # -------------logging ----------------

                if stride_counter % PARAMS['stride'] !=0:
                    # zero_frame is (1, 257) numpy array of zeros
                    features_ring_buffer.append(zero_frame)
                    stride_counter += 1
                else:
                    stride_counter += 1
                    features_buffer_time_start = time.time()
                    features_ring_buffer.append(zero_frame)
                    features_buffer_time += time.time() - features_buffer_time_start
                    features_buffer_count += 1

                    numpy_conv_time_start = time.time()
                    # conv_context dim: (31, 257)
                    conv_context = np.concatenate(list(features_ring_buffer), axis=0)
                    # addding batch dimension: (1, 31, 257)
                    conv_context = np.expand_dims(conv_context, axis=0)
                    numpy_conv_time += time.time() - numpy_conv_time_start
                    numpy_conv_count += 1

                    model_infer_time_start = time.time()
                    model_out = model(torch.from_numpy(conv_context), (hidden_in, cell_in))
                    model_infer_time += time.time() - model_infer_time_start
                    model_infer_count += 1

                    output_assign_time_start = time.time()
                    probs, (hidden_out, cell_out) = model_out
                    
                    # probs dim: (1, 1, 40)
                    probs = to_numpy(probs)
                    probs_list.append(probs)
                    hidden_in, cell_in = hidden_out, cell_out
                    output_assign_time += time.time() - output_assign_time_start
                    output_assign_count += 1

                    
                    # ------------ logging ---------------
                    logging.info(f"conv_context shape: {conv_context.shape}")
                    logging.info(f"probs shape: {probs.shape}")
                    logging.info(f"probs_list len: {len(probs_list)}")
                    #logging.info(f"probs value: {probs}")
                    # ------------ logging ---------------
            
                    # decoding every 20 time-steps
                    if count%20 ==0:
                        decoder_time_start = time.time()
                        probs_steps = np.concatenate(probs_list, axis=1)
                        int_labels = max_decode(probs_steps[0], blank=39)
                        # int_labels, likelihood = ctc_decode(probs[0], beam_size=50, blank=39)
                        predictions = preproc.decode(int_labels)
                        decoder_time += time.time() - decoder_time_start
                        decoder_count += 1
                        
                        # ------------ logging ---------------
                        logging.warning(f"predictions: {predictions}")
                        # ------------ logging ---------------
        
                total_count += 1

        if ARGS.savewav: wav_data.extend(frame)

        # process the final frames
        logging.warning(f"length of final_frames: {len(final_sample)}")


        decoder_time_start = time.time()
        probs_steps = np.concatenate(probs_list, axis=1)
        int_labels = max_decode(probs_steps[0], blank=39)
        # int_labels, likelihood = ctc_decode(probs[0], beam_size=50, blank=39)
        predictions = preproc.decode(int_labels)
        decoder_time += time.time() - decoder_time_start
        decoder_count += 1
        logging.info(f"final predictions: {predictions}")

        
        audio.destroy()
        total_time = time.time() - total_time_start
        acc = 3
        duration = wav_duration(ARGS.file)

        logging.warning(f"-------------- streaming_infer --------------")
        logging.warning(f"audio_buffer        time (s), count: {round(audio_buffer_time, acc)}, {audio_buffer_count}")
        logging.warning(f"numpy_buffer        time (s), count: {round(numpy_buffer_time, acc)}, {numpy_buffer_count}")
        logging.warning(f"features_operation  time (s), count: {round(features_time, acc)}, {features_count}")
        logging.warning(f"normalize           time (s), count: {round(normalize_time, acc)}, {normalize_count}")
        logging.warning(f"features_buffer     time (s), count: {round(features_buffer_time, acc)}, {features_buffer_count}")
        logging.warning(f"numpy_conv          time (s), count: {round(numpy_conv_time, acc)}, {numpy_conv_count}")
        logging.warning(f"model_infer         time (s), count: {round(model_infer_time, acc)}, {model_infer_count}")
        logging.warning(f"output_assign       time (s), count: {round(output_assign_time, acc)}, {output_assign_count}")
        logging.warning(f"decoder             time (s), count: {round(decoder_time, acc)}, {decoder_count}")
        logging.warning(f"total               time (s), count: {round(total_time, acc)}, {total_count}")
        logging.warning(f"Multiples faster than realtime      : {round(duration/total_time, acc)}x")

        if ARGS.savewav:
            audio.write_wav(os.path.join(ARGS.savewav, datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav")), wav_data)
            all_audio = np.frombuffer(wav_data, np.int16)
            plt.plot(all_audio)
            plt.show()
        
        probs = np.concatenate(probs_list, axis=1)
        return probs, predictions


def list_chunk_infer_full_chunks(model, preproc, lstm_states, PARAMS:dict, ARGS)->tuple:
    
    if ARGS.file is None:
        logging.warning(f"--- Skipping list_chunk_infer. No input file ---")
    else:
        
        #lc means listchunk
        lc_model_infer_time, lc_model_infer_count = 0.0, 0 
        lc_output_assign_time, lc_output_assign_count = 0.0, 0
        lc_decode_time, lc_decode_count = 0.0, 0
        lc_total_time, lc_total_count = 0.0, 0 


        lc_total_time = time.time()

        hidden_in, cell_in = lstm_states
        probs_list = list()

        audio_data, samp_rate = array_from_wave(ARGS.file)

        audio_data = make_full_window(audio_data, PARAMS['feature_window'], PARAMS['feature_step'])

        features = log_specgram_from_data(audio_data, samp_rate)
        norm_features = normalize(preproc, features)
        norm_features = np.expand_dims(norm_features, axis=0)
        torch_input = torch.from_numpy(norm_features)
        padding = (0, 0, PARAMS["initial_padding"], PARAMS["final_padding"])
        padded_input = torch.nn.functional.pad(torch_input, padding, value=0)

        full_chunks = (padded_input.shape[1] - PARAMS['chunk_size']) // PARAMS['stride']
        full_chunks += 1   

        fill_chunk_remainder = (padded_input.shape[1] - PARAMS['chunk_size']) % PARAMS['stride']

        if fill_chunk_remainder != 0:
            full_chunks += 1 # to include the filled chunk
            fill_chunk_padding = PARAMS['stride'] - fill_chunk_remainder
            fill_chunk_pad = torch.zeros(1, fill_chunk_padding, PARAMS['feature_size'], dtype=torch.float32, requires_grad=False)
            padded_input = torch.cat((padded_input, fill_chunk_pad),dim=1)

        # process last chunk with stride of zeros
        final_chunk_pad = torch.zeros(1, PARAMS['stride'], PARAMS['feature_size'], dtype=torch.float32, requires_grad=False)
        padded_input = torch.cat((padded_input, final_chunk_pad),dim=1)
        full_chunks += 1 # to include the last chunk

    
        # ------------ logging ---------------
        logging.warning(f"-------------- list_chunck_infer --------------")
        logging.warning(f"chunk_size: {PARAMS['chunk_size']}")
        logging.warning(f"full_chunks: {full_chunks}")
        logging.warning(f"features shape: {features.shape}")
        logging.warning(f"final_padding: {fill_chunk_padding}")
        logging.info(f"norm_features with batch shape: {norm_features.shape}")
        logging.info(f"torch_input shape: {torch_input.shape}")
        logging.info(f"padded_input shape: {padded_input.shape}")
        #logging.warning(f"stride: {PARAMS['stride']}")
        # torch_input.shape[1] is time dimension
        #logging.warning(f"time dim: {torch_input.shape[1]}")
        #logging.warning(f"iterations: {iterations}")
        # ------------ logging ---------------


        for i in range(full_chunks):
            
            input_chunk = padded_input[:, i*PARAMS['stride']:i*PARAMS['stride']+PARAMS['chunk_size'], :]
            
            lc_model_infer_time_start = time.time()
            model_output = model(input_chunk, (hidden_in, cell_in))
            lc_model_infer_time += time.time() - lc_model_infer_time_start
            lc_model_infer_count += 1

            lc_output_assign_time_start = time.time()
            probs, (hidden_out, cell_out) = model_output
            hidden_in, cell_in = hidden_out, cell_out
            probs = to_numpy(probs)
            probs_list.append(probs)
            lc_output_assign_time += time.time() - lc_output_assign_time_start
            lc_output_assign_count += 1
            
            # decoding every 20 time-steps
            if i%10 ==0 and i !=0:
                lc_decode_time_start = time.time()
                probs_steps = np.concatenate(probs_list, axis=1)
                int_labels = max_decode(probs_steps[0], blank=39)
                # int_labels, likelihood = ctc_decode(probs[0], beam_size=50, blank=39)
                predictions = preproc.decode(int_labels)
                lc_decode_time += time.time() - lc_decode_time_start
                lc_decode_count += 1
                #logging.info(f"intermediate predictions: {predictions}")
            
            lc_total_count += 1


            # decoding the last section
            lc_decode_time_start = time.time()
            probs_steps = np.concatenate(probs_list, axis=1)
            int_labels = max_decode(probs_steps[0], blank=39)
            # int_labels, likelihood = ctc_decode(probs[0], beam_size=50, blank=39)
            predictions = preproc.decode(int_labels)
            lc_decode_time += time.time() - lc_decode_time_start
            lc_decode_count += 1

            # ------------ logging ---------------
            logging.info(f"input_chunk shape: {input_chunk.shape}")
            logging.info(f"probs shape: {probs.shape}")
            logging.info(f"probs list len: {len(probs_list)}")
            # ------------ logging ---------------
        lc_total_time = time.time() - lc_total_time

        duration = wav_duration(ARGS.file)
        # ------------ logging ---------------
        logging.warning(f"predictions: {predictions}")
        acc = 3
        logging.warning(f"model infer          time (s), count: {round(lc_model_infer_time, acc)}, {lc_model_infer_count}")
        logging.warning(f"output assign        time (s), count: {round(lc_output_assign_time, acc)}, {lc_output_assign_count}")
        logging.warning(f"decoder              time (s), count: {round(lc_decode_time, acc)}, {lc_decode_count}")
        logging.warning(f"total                time (s), count: {round(lc_total_time, acc)}, {lc_total_count}")
        logging.warning(f"Multiples faster than realtime      : {round(duration/lc_total_time, acc)}x")


        probs = np.concatenate(probs_list, axis=1)
        return probs, predictions


def full_audio_infer(model, preproc, lstm_states, PARAMS:dict, ARGS)->tuple:
    """
    conducts inference from an entire audio file. If no audio file
    is provided in ARGS when recording from mic, this function is exited.
    """

    if ARGS.file is None:
        logging.warning(f"--- Skipping fullaudio_infer. No input file ---")
    else:
        # fa means fullaudio
        fa_total_time = 0.0
        fa_features_time = 0.0
        fa_normalize_time = 0.0
        fa_convert_pad_time = 0.0
        fa_model_infer_time = 0.0
        fa_decode_time = 0.0

        hidden_in, cell_in = lstm_states

        fa_total_time = time.time()

        audio_data, samp_rate = array_from_wave(ARGS.file)
        audio_data = make_full_window(audio_data, PARAMS['feature_window'], PARAMS['feature_step'])

        fa_features_time = time.time()
        features = log_specgram_from_data(audio_data, samp_rate)
        fa_features_time = time.time() - fa_features_time
        
        fa_normalize_time = time.time()
        norm_features = normalize(preproc, features)

        fa_normalize_time = time.time() - fa_normalize_time

        fa_convert_pad_time = time.time()
        # adds the batch dimension (1, time, 257)
        norm_features = np.expand_dims(norm_features, axis=0)
        torch_input = torch.from_numpy(norm_features)
        # paddings starts from the back, zero padding to freq, 15 padding to time
        padding = (0, 0, PARAMS["initial_padding"],PARAMS["final_padding"])
        padded_input = torch.nn.functional.pad(torch_input, padding, value=0)

        fill_chunk_remainder = (padded_input.shape[1] - PARAMS['chunk_size']) % PARAMS['stride']

        if fill_chunk_remainder != 0:
            fill_chunk_padding = PARAMS['stride'] - fill_chunk_remainder
            fill_chunk_pad = torch.zeros(1, fill_chunk_padding, PARAMS['feature_size'], dtype=torch.float32, requires_grad=False)
            padded_input = torch.cat((padded_input, fill_chunk_pad),dim=1)

        # process last chunk with stride of zeros
        final_chunk_pad = torch.zeros(1, PARAMS['stride'], PARAMS['feature_size'], dtype=torch.float32, requires_grad=False)
        padded_input = torch.cat((padded_input, final_chunk_pad),dim=1)

        fa_model_infer_time = time.time()
        model_output = model(padded_input, (hidden_in, cell_in))
        fa_model_infer_time = time.time() - fa_model_infer_time

        probs, (hidden_out, cell_out) = model_output
        probs = to_numpy(probs)
        fa_decode_time = time.time()
        int_labels = max_decode(probs[0], blank=39)
        fa_decode_time = time.time() - fa_decode_time
        # int_labels, likelihood = ctc_decode(probs[0], beam_size=50, blank=39)
        predictions = preproc.decode(int_labels)
        
        fa_total_time = time.time() - fa_total_time
        

        duration = wav_duration(ARGS.file)

        # ------------ logging ---------------
        logging.warning(f"------------ fullaudio_infer -------------")
        logging.info(f"features shape: {features.shape}")
        logging.info(f"norm_features with batch shape: {norm_features.shape}")
        logging.info(f"torch_input shape: {torch_input.shape}")
        logging.warning(f"chunk_size: {PARAMS['chunk_size']}")
        logging.warning(f"final_padding: {PARAMS['final_padding']}")
        logging.info(f"padded_input shape: {padded_input.shape}")
        logging.info(f"model probs shape: {probs.shape}")
        logging.warning(f"predictions: {predictions}")
        acc = 3
        logging.warning(f"features             time (s): {round(fa_features_time, acc)}")
        logging.warning(f"normalization        time (s): {round(fa_normalize_time, acc)}")
        logging.warning(f"convert & pad        time (s): {round(fa_convert_pad_time, acc)}")
        logging.warning(f"model infer          time (s): {round(fa_model_infer_time, acc)}")
        logging.warning(f"decoder              time (s): {round(fa_decode_time, acc)}")
        logging.warning(f"total                time (s): {round(fa_total_time, acc)}")
        logging.warning(f"Multiples faster than realtime      : {round(duration/fa_total_time, acc)}x")

        # ------------ logging ---------------


        return probs, predictions


def list_chunk_infer_fractional_chunks(model, preproc, lstm_states, PARAMS:dict, ARGS)->tuple:
    
    if ARGS.file is None:
        logging.warning(f"--- Skipping list_chunk_infer. No input file ---")
    else:
        
        #lc means listchunk
        lc_model_infer_time, lc_model_infer_count = 0.0, 0 
        lc_output_assign_time, lc_output_assign_count = 0.0, 0
        lc_decode_time, lc_decode_count = 0.0, 0
        lc_total_time, lc_total_count = 0.0, 0 


        lc_total_time = time.time()

        hidden_in, cell_in = lstm_states
        probs_list = list()

        features = log_specgram_from_file(ARGS.file)
        norm_features = normalize(preproc, features)
        norm_features = np.expand_dims(norm_features, axis=0)
        torch_input = torch.from_numpy(norm_features)
        padding = (0, 0, PARAMS["initial_padding"], PARAMS["final_padding"])
        padded_input = torch.nn.functional.pad(torch_input, padding, value=0)

        full_chunks = (padded_input.shape[1] - PARAMS['chunk_size']) // PARAMS['stride']
        full_chunks += 1

        # ------------ logging ---------------
        logging.warning(f"-------------- list_chunck_infer --------------")
        logging.warning(f"======= chunk_size: {PARAMS['chunk_size']}===========")
        logging.warning(f"======= full_chunks: {full_chunks}===========")
        logging.warning(f"======= fraction_chunks: {PARAMS['remainder']}===========")
        logging.info(f"features shape: {features.shape}")
        logging.info(f"norm_features with batch shape: {norm_features.shape}")
        logging.info(f"torch_input shape: {torch_input.shape}")
        logging.info(f"padded_input shape: {padded_input.shape}")
        #logging.warning(f"stride: {PARAMS['stride']}")
        # torch_input.shape[1] is time dimension
        #logging.warning(f"time dim: {torch_input.shape[1]}")
        #logging.warning(f"iterations: {iterations}")
        # ------------ logging ---------------


        for i in range(full_chunks+PARAMS['remainder']):
            
            # if and elif handle fractional chunks, else handles full chunks
            if i == full_chunks:  
                inner_bound = i*PARAMS['stride']
                outer_bound = inner_bound+(2*PARAMS['context']+1)
                input_chunk = padded_input[:, inner_bound:outer_bound, :]
            elif i > full_chunks:
                # stride of 1
                inner_bound += 1
                outer_bound = inner_bound+(2*PARAMS['context']+1)
                input_chunk = padded_input[:, inner_bound:outer_bound, :]
            else: 
                input_chunk = padded_input[:, i*PARAMS['stride']:i*PARAMS['stride']+PARAMS['chunk_size'], :]
            
            lc_model_infer_time_start = time.time()
            model_output = model(input_chunk, (hidden_in, cell_in))
            lc_model_infer_time += time.time() - lc_model_infer_time_start
            lc_model_infer_count += 1

            lc_output_assign_time_start = time.time()
            probs, (hidden_out, cell_out) = model_output
            hidden_in, cell_in = hidden_out, cell_out
            probs = to_numpy(probs)
            probs_list.append(probs)
            lc_output_assign_time += time.time() - lc_output_assign_time_start
            lc_output_assign_count += 1
            
            # decoding every 20 time-steps
            if i%10 ==0 and i !=0:
                lc_decode_time_start = time.time()
                probs_steps = np.concatenate(probs_list, axis=1)
                int_labels = max_decode(probs_steps[0], blank=39)
                # int_labels, likelihood = ctc_decode(probs[0], beam_size=50, blank=39)
                predictions = preproc.decode(int_labels)
                lc_decode_time += time.time() - lc_decode_time_start
                lc_decode_count += 1
                #logging.info(f"intermediate predictions: {predictions}")
            
            lc_total_count += 1


            # decoding the last section
            lc_decode_time_start = time.time()
            probs_steps = np.concatenate(probs_list, axis=1)
            int_labels = max_decode(probs_steps[0], blank=39)
            # int_labels, likelihood = ctc_decode(probs[0], beam_size=50, blank=39)
            predictions = preproc.decode(int_labels)
            lc_decode_time += time.time() - lc_decode_time_start
            lc_decode_count += 1

            # ------------ logging ---------------
            logging.info(f"input_chunk shape: {input_chunk.shape}")
            logging.info(f"probs shape: {probs.shape}")
            logging.info(f"probs list len: {len(probs_list)}")
            # ------------ logging ---------------
        lc_total_time = time.time() - lc_total_time

        duration = wav_duration(ARGS.file)
        # ------------ logging ---------------
        logging.warning(f"predictions: {predictions}")
        acc = 3
        logging.warning(f"model infer          time (s), count: {round(lc_model_infer_time, acc)}, {lc_model_infer_count}")
        logging.warning(f"output assign        time (s), count: {round(lc_output_assign_time, acc)}, {lc_output_assign_count}")
        logging.warning(f"decoder              time (s), count: {round(lc_decode_time, acc)}, {lc_decode_count}")
        logging.warning(f"total                time (s), count: {round(lc_total_time, acc)}, {lc_total_count}")
        logging.warning(f"Multiples faster than realtime      : {round(duration/lc_total_time, acc)}x")


        probs = np.concatenate(probs_list, axis=1)
        return probs, predictions






class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, 
    and stored in a buffer, to be read from.
    """

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 62.5

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS, file=None):
        def proxy_callback(in_data, frame_count, time_info, status):
            #pylint: disable=unused-argument
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            callback(in_data)
            return (None, pyaudio.paContinue)
        if callback is None: callback = lambda in_data: self.buffer_queue.put(in_data)
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
        print(f"block_size input {self.block_size_input}")
        self.pa = pyaudio.PyAudio()

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }

        self.chunk = None
        # if not default device
        if self.device:
            kwargs['input_device_index'] = self.device
        elif file is not None:
            self.chunk = 256
            self.wf = wave.open(file, 'rb')

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def resample(self, data, input_rate):
        """
        Microphone may not support our native processing sampling rate, so
        resample from input_rate to RATE_PROCESS here for webrtcvad and
        deepspeech
        Args:
            data (binary): Input audio stream
            input_rate (int): Input audio rate to resample from
        """
        data16 = np.fromstring(string=data, dtype=np.int16)
        resample_size = int(len(data16) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        return resample16.tostring()

    def read_resampled(self):
        """Return a block of audio data resampled to 16000hz, blocking if necessary."""
        return self.resample(data=self.buffer_queue.get(),
                             input_rate=self.input_rate)

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)

    def write_wav(self, filename, data):
        logging.warning("write wav %s", filename)
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
            assert self.FORMAT == pyaudio.paInt16
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(data)
    
    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            while True:
                yield self.read_resampled()


def max_decode(output, blank=39):
    pred = np.argmax(output, 1)
    prev = pred[0]
    seq = [prev] if prev != blank else []
    for p in pred[1:]:
        if p != blank and p != prev:
            seq.append(p)
        prev = p
    return seq


if __name__ == '__main__':
    BEAM_WIDTH = 500
    DEFAULT_SAMPLE_RATE = 16000

    import argparse
    parser = argparse.ArgumentParser(description="Stream from microphone to DeepSpeech using VAD")
    parser.add_argument('-w', '--savewav',
                        help="Save .wav files of utterences to given directory")
    parser.add_argument('-f', '--file',
                        help="Read from .wav file instead of microphone")
    parser.add_argument('-m', '--model',
                        help="Path to the model (protocol buffer binary file, or entire directory containing all standard-named files for model)")
    parser.add_argument('-c', '--config', type = str,
                        help="Path to the config file for that model"),
    parser.add_argument('-d', '--device', type=int, default=None,
                        help="Device input index (Int) as listed by pyaudio.PyAudio.get_device_info_by_index(). If not provided, falls back to PyAudio.get_default_device().")
    parser.add_argument('-r', '--rate', type=int, default=DEFAULT_SAMPLE_RATE,
                        help=f"Input device sample rate. Default: {DEFAULT_SAMPLE_RATE}. Your device may require 44100.")
    # ctc decoder not currenlty used
    parser.add_argument('-bw', '--beam_width', type=int, default=BEAM_WIDTH,
                        help=f"Beam width used in the CTC decoder when building candidate transcriptions. Default: {BEAM_WIDTH}")

    ARGS = parser.parse_args()
    if ARGS.savewav: os.makedirs(ARGS.savewav, exist_ok=True)
    main(ARGS)

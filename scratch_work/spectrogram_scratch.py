# standard libraries

# 3rd party libraries
import torch
import scipy.signal
import matplotlib.pyplot as plt
import librosa
import numpy as np

# project libraries
from speech.utils import spec_augment
from speech.loader import log_specgram_from_file, apply_spec_augment
from speech.utils import wave

  

def main():
    
    # convert the audio
    audio_path = './onnx_coreml/audio_files/ST-out.wav'
    audio, sample_rate = wave.array_from_wave(audio_path)
    print(f"sample_rate: {sample_rate}")

    # compute the spectrogram
    window_size = 32
    step_size = 16
    nperseg = int(window_size * sample_rate / 1e3)
    noverlap = int(step_size * sample_rate / 1e3)
    f, t, spec = scipy.signal.spectrogram(audio,
                fs=sample_rate,
                window='hann',
                nperseg=nperseg,
                noverlap=noverlap,
                detrend=False)

    print(f"fshape: {f.shape}, spec shape: {spec.shape}")

    log_spec = np.log(spec.astype(np.float32))
    log_f = np.log(f.astype(np.float32))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(10, 20))

    # plot the spectrogram

    ax1.pcolormesh(t, f, log_spec)
    ax1.set_ylabel('Hz')
    #ax1.set_yscale('symlog')
    ax1.set_yscale('symlog', basey=2, linthreshy=64)
    #ax1.set_ylim(0,8000)


    print(f"log_spec shape: {log_spec.shape}")

    librosa.display.specshow(
            log_spec,
            y_axis='log', sr=32000, ax=ax2
            )
    ax2.set_title('original')

    # spec_1, f_1, t_1, im= plt.specgram(audio, Fs= sample_rate, NFFT=nperseg, noverlap=noverlap)
    # log_spec_1 = np.log(spec_1.astype(np.float32))
    # ax2.pcolormesh(t_1, f_1, log_spec_1)

    log_spec = log_specgram_from_file(audio_path, window_size=32, step_size=16)  
    log_spec_T = torch.from_numpy(log_spec.T)
    policies = {0: {'time_warping_para':0, 'frequency_masking_para':50,
                 'time_masking_para':50, 'frequency_mask_num':0, 'time_mask_num':0}, 
                1: {"time_warping_para":5, "frequency_masking_para":60,
                 "time_masking_para":60, "frequency_mask_num":1, "time_mask_num":1},
                2: {"time_warping_para":5, "frequency_masking_para":30,
                 "time_masking_para":30, "frequency_mask_num":2, "time_mask_num":2},
                3: {"time_warping_para":5, "frequency_masking_para":20,
                 "time_masking_para":20, "frequency_mask_num":3, "time_mask_num":3},
                 }
    policy = policies.get(2)
    aug_log_spec_T = spec_augment.spec_augment(log_spec_T, 
                        time_warping_para=policy.get('time_warping_para'), 
                        frequency_masking_para=policy.get('frequency_masking_para'),
                        time_masking_para=policy.get('time_masking_para'),
                        frequency_mask_num=policy.get('frequency_mask_num'), 
                        time_mask_num=policy.get('time_mask_num'))
    spec_augment.visualization_spectrogram(aug_log_spec_T.numpy(), f"double mask augment", ax=ax3) 

    output = apply_spec_augment(log_spec)
    spec_augment.visualization_spectrogram(output, f"apply_spec_augment", ax=ax4) 
    plt.show()

if __name__ == "__main__":
    main()
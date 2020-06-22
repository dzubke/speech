# third-party libraries
import librosa
import librosa.display
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import torch
# project libraries
from .sparse_image_warp import sparse_image_warp



def feature_gaussian_noise_inject(inputs:np.ndarray, rand_noise_multi_std:float, rand_noise_add_std:float):
  inputs = inputs * np.random.normal(loc=1, scale=rand_noise_multi_std, size=inputs.shape)
  inputs = inputs + np.random.normal(loc=0, scale=rand_noise_add_std, size=inputs.shape)
  return inputs

def apply_spec_augment(inputs, logger):
    """calls the spec_augment function on the normalized log_spec. A policy defined 
        in the policy_dict will be chosen uniformly at random.
    Arguments:
        inputs (np.ndarray): normalized log_spec with dimensional order time x freq
    Returns:
        inputs (nd.ndarray): the modified log_spec array with order time x freq
    """

    use_log = (logger is not None)
    assert type(inputs) == np.ndarray, "input is not numpy array"

    policy_dict = {
        0: {'time_warping_para':0, 'frequency_masking_para':0,
            'time_masking_para':0, 'frequency_mask_num':0, 'time_mask_num':0}, 
        1: {"time_warping_para":20, "frequency_masking_para":60,
            "time_masking_para":60, "frequency_mask_num":1, "time_mask_num":1},
        2: {"time_warping_para":20, "frequency_masking_para":30,
            "time_masking_para":30, "frequency_mask_num":2, "time_mask_num":2},
        3: {"time_warping_para":20, "frequency_masking_para":20,
            "time_masking_para":20, "frequency_mask_num":3, "time_mask_num":3},
            }
    
    policy_choice = np.random.randint(low=0, high=4)
    if use_log: logger.info(f"spec_aug: policy: {policy_choice}")

    policy = policy_dict.get(policy_choice)

    # the inputs need to be transposed and converted to torch tensor
    # as spec_augment method expects tensor with freq x time dimensions
    if use_log: logger.info(f"spec_aug: input shape: {inputs.shape}")

    inputs = torch.from_numpy(inputs.T)

    inputs = spec_augment(inputs, 
                    time_warping_para=policy.get('time_warping_para'), 
                    frequency_masking_para=policy.get('frequency_masking_para'),
                    time_masking_para=policy.get('time_masking_para'),
                    frequency_mask_num=policy.get('frequency_mask_num'), 
                    time_mask_num=policy.get('time_mask_num'), logger=logger)
    
    # convert the torch tensor back to numpy array and transpose back to time x freq
    inputs = inputs.detach().cpu().numpy() if inputs.requires_grad else inputs.cpu().numpy()
    inputs = inputs.T
    assert type(inputs) == np.ndarray, "output is not numpy array"

    return inputs



def time_warp(spec, W, logger):
    use_log = (logger is not None)
    if W==0:
        return spec

    num_rows = spec.shape[1]
    spec_len = spec.shape[2]

    assert spec_len>2*W, "frequency dimension is not large enough for W parameter"
    assert num_rows>0, "time dimension must be greater than zero"

    y = num_rows // 2
    horizontal_line_at_ctr = spec[0][y]
    # assert len(horizontal_line_at_ctr) == spec_len

    point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len-W)]
    # assert isinstance(point_to_warp, torch.Tensor)

    # Uniform distribution from (0,W) with chance to be up to W negative
    dist_to_warp = random.randrange(-W, W)
    
    if use_log: logger.info(f"spec_aug: W is: {W}")
    if use_log: logger.info(f"spec_aug: point_to_warp: {point_to_warp}")
    if use_log: logger.info(f"spec_aug: dist_to_warp: {dist_to_warp}")

    src_pts = torch.tensor([[[y, point_to_warp]]])
    dest_pts = torch.tensor([[[y, point_to_warp + dist_to_warp]]])
    warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)

    return warped_spectro.squeeze(3)


def spec_augment(mel_spectrogram, time_warping_para=5, frequency_masking_para=50,
                 time_masking_para=50, frequency_mask_num=1, time_mask_num=1, logger=None):
    """Spec augmentation Calculation Function.
    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    # Arguments:
      spectrogram(torch tensor): audio file path of you want to warping and masking.
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
        If none, default = 4.
      frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 50.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 50.
      frequency_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1.
    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    use_log = (logger is not None)
    
    mel_spectrogram = mel_spectrogram.unsqueeze(0)

    v = mel_spectrogram.shape[1]
    tau = mel_spectrogram.shape[2]
    if use_log: logger.info(f"spec_aug: nu is: {v}")
    if use_log: logger.info(f"spec_aug: tau is: {tau}")

    # Step 1 : Time warping
    warped_mel_spectrogram = time_warp(mel_spectrogram, W=time_warping_para, logger=logger)
    if use_log: logger.info(f"spec_aug: finished time_warp")
    #warped_mel_spectrogram = mel_spectrogram

    # Step 2 : Frequency masking
    for i in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        if v - f < 0:
            continue
        f0 = random.randint(0, v-f)
        if use_log: logger.info(f"spec_aug: f is: {f} at: {f0}")

        warped_mel_spectrogram[:, f0:f0+f, :] = 0
    # Step 3 : Time masking
    for i in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)

        if tau - t < 0:
            continue
        t0 = random.randint(0, tau-t)
        if use_log: logger.info(f"spec_aug: t is: {t} at: {t0}")

        warped_mel_spectrogram[:, :, t0:t0+t] = 0

    return warped_mel_spectrogram.squeeze()


def visualization_spectrogram(mel_spectrogram, title, ax=None):
    """visualizing result of SpecAugment
    # Arguments:
      spectrogram(ndarray): mel_spectrogram to visualize.
      title(String): plot figure's title
    """
    #mel_spectrogram = mel_spectrogram.unsqueeze(0)
    # Show mel-spectrogram using librosa's specshow.

    #plt.figure(figsize=(10, 4))
    librosa.display.specshow(
            mel_spectrogram,
            y_axis='log',x_axis='time', sr=32000, ax=ax
            )
    # plt.colorbar(format='%+2.0f dB')
    
    plt.title(title) if ax==None else ax.set_title(title) 
    #plt.tight_layout()
    #plt.show()

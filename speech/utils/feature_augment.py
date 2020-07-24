# standard libraries
from logging import Logger
import random
# third-party libraries
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
# project libraries
from .sparse_image_warp import sparse_image_warp
from speech.utils.convert import to_numpy



def feature_gaussian_noise_inject(inputs:np.ndarray, 
                                  rand_noise_multi_std:float, 
                                  rand_noise_add_std:float)->np.ndarray:
    """
    adds gaussian noise to the 2d feature from a standard distribution

    """
    inputs = inputs * np.random.normal(loc=1, scale=rand_noise_multi_std, size=inputs.shape)
    inputs = inputs + np.random.normal(loc=0, scale=rand_noise_add_std, size=inputs.shape)
    return inputs


# spec-augment functions: apply_spec_augment, time_warp, spec_augment, visualize_spectrogram
def apply_spec_augment(features:np.ndarray, policy:dict, logger:Logger=None)->np.ndarray:
    """
    Calls the spec_augment function on the normalized features. A policy defined 
    in the policy_dict will be chosen uniformly at random.
    Arguments:
        features - np.ndarray: normalized features with dimensional order time x freq
        policy - dict: set of augmentation policies
        logger - Logger
    Returns:
        features - nd.ndarray: the modified features array with order time x freq
    """
    
    use_log = (logger is not None)
    assert type(features) == np.ndarray, "input is not numpy array"
        
    policy_choice = np.random.randint(low=0, high=len(policy.keys()))
    if use_log: logger.info(f"spec_aug: policy: {policy_choice}")
    
    # the index of the original policy_dict is a integer in yaml 0.1.7 and a str in 0.2.5
    # yaml 0.1.7 is being used currently. will throw 'None has no get attribute error' if yaml install is wrong
    policy = policy[policy_choice]

    # the inputs need to be transposed and converted to torch tensor
    # as spec_augment method expects tensor with freq x time dimensions
    if use_log: logger.info(f"spec_aug: features shape: {features.shape}")

    features = torch.from_numpy(features.T)

    features = spec_augment(features, 
                            time_warping_para=policy.get('W', 0.0), 
                            frequency_masking_para=policy.get('F', 0.0),
                            time_masking_para=policy.get('T', 0.0),
                            frequency_mask_num=policy.get('m_F', 0.0), 
                            time_mask_num=policy.get('m_T', 0.0), 
                            logger=logger)
    
    # convert the torch tensor back to numpy array and transpose back to time x freq
    features = to_numpy(features)
    features = features.T
    assert type(features) == np.ndarray, "output is not numpy array"

    return features



def time_warp(spec:torch.Tensor, W:float, logger:Logger=None, fixed_params:dict=None):
    """
    Given a log mel spectrogram with τ time steps, we view it as an image where 
    the time axis is horizontal and the frequency axis is vertical. 
    A random point along the horizontal line passing through the center of the image 
    within the time steps (W, τ − W ) is to be warped either to the left or right by 
    a distance w chosen from a uniform distribution from 0 to the time warp parameter 
    W along that line. We fix six anchor points on the boundary—the four corners and
    the mid-points of the vertical edges.
    
    Arguments:
        spec - torch.Tensor: 2d spectrogram with dimensions freq x time
    """
    use_log = (logger is not None)
    use_fixed = (fixed_params is not None)

    if W==0:
        return spec

    num_rows = spec.shape[1]    # freq dimension
    spec_len = spec.shape[2]    # time dimension

    assert spec_len>2*W, "time dimension is not large enough for W parameter"
    assert num_rows>0, "freq dimension must be greater than zero"

    y = num_rows // 2
    horizontal_line_at_ctr = spec[0][y]
    # assert len(horizontal_line_at_ctr) == spec_len
    if use_fixed:
        point_to_warp = fixed_params.get("point_to_warp")
    else:
        point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len-W)]
    
    if use_fixed:
        dist_to_warp = fixed_params.get("dist_to_warp")
        assert -W <= dist_to_warp <= W, f"dist_to_warp {dist_to_warp} outside bounds: {-W}, {W}"
    else:
        # Uniform distribution from (0,W) with chance to be up to W negative
        dist_to_warp = random.randrange(-W, W)
    
    if use_log: logger.info(f"spec_aug: W is: {W}")
    if use_log: logger.info(f"spec_aug: point_to_warp: {point_to_warp}")
    if use_log: logger.info(f"spec_aug: dist_to_warp: {dist_to_warp}")

    src_pts = torch.tensor([[[y, point_to_warp]]])
    dest_pts = torch.tensor([[[y, point_to_warp + dist_to_warp]]])
    warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)

    return warped_spectro.squeeze(3)


def spec_augment(mel_spectrogram:torch.Tensor, 
                time_warping_para:float=5, 
                frequency_masking_para:float=50,
                time_masking_para:float=50, 
                frequency_mask_num:float=1, 
                time_mask_num:float=1, 
                logger:Logger=None,
                fixed_params:dict=None):
    """Spec augmentation Calculation Function.
    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    Arguments:
      spectrogram(torch tensor): audio file path of you want to warping and masking.
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
      frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
      frequency_mask_num(float): number of frequency masking lines, "m_F".
      time_mask_num(float): number of time masking lines, "m_T".
      fixed_params(dict): if given, used parameters in dict instead of random values
    Returns:
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    use_log = (logger is not None)
    use_fixed = (fixed_params is not None)
    
    mel_spectrogram = mel_spectrogram.unsqueeze(0)

    v = mel_spectrogram.shape[1]
    tau = mel_spectrogram.shape[2]
    if use_log: logger.info(f"spec_aug: nu is: {v}")
    if use_log: logger.info(f"spec_aug: tau is: {tau}")

    # Step 1 : Time warping
    warped_mel_spectrogram = time_warp(mel_spectrogram, W=time_warping_para, 
                                        logger=logger, fixed_params=fixed_params)
    if use_log: logger.info(f"spec_aug: finished time_warp")
    #warped_mel_spectrogram = mel_spectrogram

    # Step 2 : Frequency masking
    for i in range(frequency_mask_num):
        if use_fixed:
            f = fixed_params.get("f")[i]
            assert 0 <= f <= frequency_masking_para, f"f {f} is out of bounds 0, {frequency_masking_para}"
        else:
            f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)

        if v - f < 0:
            continue

        if use_fixed:
            f0 = fixed_params.get("f0")[i]
            #assert 0 <= f0 <= v-f, f"f0 {f0} value out of bounds: 0, {v-f}"
        else:
            f0 = random.randint(0, v-f)
        if use_log: logger.info(f"spec_aug: f is: {f} at: {f0}")

        warped_mel_spectrogram[:, f0:f0+f, :] = 0
    # Step 3 : Time masking
    for i in range(time_mask_num):
        if use_fixed:
            t = fixed_params.get("t")[i]
            assert 0 <= t <= time_masking_para, f"t {t} is out of bounds: 0, {time_masking_para}"
        else:
            t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)

        if tau - t < 0:
            continue

        if use_fixed:
            t0 = fixed_params.get("t0")[i]
            #assert 0 <= t0 <= tau-t, f"t0 {t0} is out of bounds: 0, {tau-t}"
        else:
            t0 = random.randint(0, tau-t)
        if use_log: logger.info(f"spec_aug: t is: {t} at: {t0}")

        warped_mel_spectrogram[:, :, t0:t0+t] = 0

    return warped_mel_spectrogram.squeeze()


def visualize_spectrogram(mel_spectrogram, title, ax=None):
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

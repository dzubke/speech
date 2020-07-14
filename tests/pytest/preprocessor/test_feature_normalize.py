# third-party libraries
import numpy as np
import pytest
# project libraries
from speech.loader import feature_normalize, log_spectrogram_from_file
from tests.pytest.utils import get_all_test_audio



def test_audio_feature_normalize():
    """

    """
    audio_files = get_all_test_audio()
    
    for audio_file in audio_files:
        feature = log_spectrogram_from_file(audio_file)
        normalized_feature = feature_normalize(feature)
        np.testing.assert_almost_equal(normalized_feature.mean(), 0)
        np.testing.assert_almost_equal(normalized_feature.std(), 1)

def test_asserts_feature_normalize():

    test_arrays = [
        np.zeros((1,1), dtype=np.float32), #checks std!=0 assert
        np.zeros((100,257), dtype=np.float32), #checks std!=0 assert
        np.ones((100,257), dtype=np.float32), #checks std!=0 assert
        np.empty((1,1), dtype=np.float32), # checks nan (std==std) assert
        np.empty((0,0), dtype=np.float32), # checks nan (std==std) assert
        np.empty((0,), dtype=np.float32), #checks std!=0 assert
        np.random.randn(10,10).astype('float64')  # checks dtype assert
    ]
    # adding array with NaN value
    test_arr = np.random.randn(100,257).astype('float32')
    test_arr[0,0] = np.nan
    test_arrays.append(test_arr)    # checks nan (std==std) assert

    # there is an assertion in feature_normalize that checks if std !=0
    # if std ==0 the function will raise an AsssertionError
    for test_num, test_array in enumerate(test_arrays):
        with pytest.raises(AssertionError) as execinfo:
            print(f"test number: {test_num}")
            normalized_feature = feature_normalize(test_array)
            #np.testing.assert_almost_equal(normalized_feature.mean(), 0)
            #np.testing.assert_almost_equal(normalized_feature.std(), 1)
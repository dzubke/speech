# standard libs
import os
# third-party libs
import numpy as np
# project libs
from speech.utils.io import read_pickle
from speech.utils import get_native_model_optimizer


def native_loss():
   
    file_dir =  os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(file_dir, "ctc_config_ph3.yaml")
    model, optimizer = get_native_model_optimizer.get_model_optimizer(config_path)
    print(model)
    batch = get_saved_batch()

    loss = model.native_loss(batch)   


def get_saved_batch():
    batch_path = "/home/dzubke/awni_speech/speech/saved_batch/2020-06-17_v2_ph2_withBatchNorm_ted_batch.pickle"
    return read_pickle(batch_path)
       

if __name__ == "__main__":
    native_loss()

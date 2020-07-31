# this is based on the pytorch tutorial: https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html#run-the-model

# standard libraries
import argparse
import json
import os
# third-party libs 
import numpy as np 
import pytest 
import torch  
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import functions.ctc as ctc #awni hannun's ctc bindings
# project libraries          
from speech.models.ctc_model_train import CTC_train  
import speech.loader as loader  
from speech.utils.compat import get_main_dir_path 
from speech.utils.data_structs import Batch 
from speech.utils.io import read_pickle, load_from_trained, load_config
from speech.utils.model_debug import check_nan_params_grads, get_logger


class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = np.random.randn(length, size).astype(np.float32)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(), "output size", output.size())
        print("\tIn Model: input device", input.device, "output device", output.device)

        return output


def tutorial():

    # Parameters and DataLoaders
    input_size = 5
    output_size = 2

    batch_size = 30
    data_size = 100 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)

    model = Model(input_size, output_size)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)
    for data in rand_loader:
        features = data
        output = model(features)
        print("Outside: input size", features.size(), "output_size", output.size())
        print("Outside: input type: ", type(features), "output type: ", type(output))
        print("Outside: input device: ", features.device, "output device: ", output.device) 

def test_real_model(use_saved_batch:bool=False, only_infer:bool=False):
    """
    Arguments
    ---------
    use_saved_batch - bool: if true, a saved batch will be fed into the model
    only_infer - bool: if true, items from the audio-dataset will be fed into the model forward function
    """

    config_path = "/home/dzubke/awni_speech/speech/examples/librispeech/ctc_config_ph3.yaml"
    config = load_config(config_path)
    data_cfg = config["data"] 
    preproc_cfg = config["preproc"] 
    opt_cfg = config["optimizer"] 
    model_cfg = config["model"]
    logger = get_logger()
    
    preproc = loader.Preprocessor(data_cfg["train_set"], preproc_cfg, logger,   
                        max_samples=20, start_and_end=data_cfg["start_and_end"]) 
    batch_size=8
    dev_ldr = loader.make_loader(data_cfg["dev_sets"]['speak'], 
                            preproc, batch_size, num_workers=data_cfg["num_workers"]) 
    
    dataset = loader.AudioDataset(data_cfg["dev_sets"]['speak'], preproc, batch_size)
    device_ids = [i for i in range(torch.cuda.device_count())] 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model = CTC_train(preproc.input_dim,  
                                preproc.vocab_size,  
                                model_cfg,
                                device)
    print("model type: ", type(model)) 
    dataparallel_model  = torch.nn.DataParallel(model)
    print("datap model type: ", type(dataparallel_model))
    print(dataparallel_model)
    assert isinstance(dataparallel_model, torch.nn.DataParallel)
    model = dataparallel_model.module
    dataparallel_model = dataparallel_model.to(device)
    model.set_train()
    optimizer = torch.optim.SGD(model.parameters(),  
                            lr= opt_cfg['learning_rate'],  
                            momentum=opt_cfg["momentum"],  
                            dampening=opt_cfg["dampening"])

    if use_saved_batch:
        batch_path = "./saved_batch/2020-06-17_v2_ph3_withBatchNorm_lib-ted-cv_batch.pickle"
        batch = read_pickle(batch_path)
        # rename dev_lder
        num_batches = 8
        dev_ldr = [batch for _ in range(num_batches)]
        
    if only_infer: 
        for datum in dataset:
            features, labels = datum
            features = torch.from_numpy(features)
            features = features.unsqueeze(0)
            dummy_features = torch.randn(features.shape)
            features = torch.cat((features, dummy_features), dim=0)
            print("Outside: input size before infer", features.size())
            output, rnn_args = model.forward(features)
            print("Outside: input size", features.size(), "output_size", output.size())
            print("Outside: input type: ", type(features), "output type: ", type(output))
            print("Outside: input device: ", features.device, "output device: ", output.device)
    else:
        for temp_batch in dev_ldr:
            optimizer.zero_grad() 
            x, y, x_lens, y_lens = model.collate(*temp_batch)
            out, rnn_args = dataparallel_model(x)
            loss_fn = ctc.CTCLoss()
            loss = loss_fn(out, y, x_lens, y_lens)
            loss.backward() 
            optimizer.step() 
            print(f"loss: {loss.item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Tests using multiple GPU's.")
    parser.add_argument("--tutorial", action="store_true", default=False,
        help="if use, tutorial script will be run.")
    parser.add_argument("--real-model", action="store_true", default=False,
        help="if used, real-model script will be run")
    parser.add_argument("--use-saved-batch", action="store_true", default=False,
        help="if used, a saved batch will be used in real-model instead of data loader.")
    parser.add_argument("--only-infer", action="store_true", default=False,
        help="if used, data from the audiodataset will be fed into model's forward function.")

    args = parser.parse_args()

    if args.tutorial:
        tutorial()
    elif args.real_model:
        test_real_model(args.use_saved_batch, args.only_infer)

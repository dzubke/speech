# third-party libs
import torch
# project libs
import speech.loader as loader 
from speech.models.ctc_model_train import CTC_train as CTC_train_awni
from speech.utils.io import load_config, load_from_trained


def get_model_optimizer(config_path:str):
    """
    Creates a model and optimizer object from the confing file in config_path
    """

    config = load_config(config_path)

    data_cfg = config["data"]
    preproc_cfg = config["preproc"]
    opt_cfg = config["optimizer"]
    model_cfg = config["model"]

    logger = None
    preproc = loader.Preprocessor(data_cfg["train_set"], preproc_cfg, logger,
                       max_samples=100, start_and_end=data_cfg["start_and_end"])
    
    model = CTC_train_awni(preproc.input_dim,
                             preproc.vocab_size,
                             model_cfg)

    if model_cfg["load_trained"]:
        model = load_from_trained(model, model_cfg)
        print(f"Succesfully loaded weights from trained model: {model_cfg['trained_path']}")
    model.cuda() if torch.cuda.is_available() else model.cpu()

    optimizer = torch.optim.SGD(model.parameters(),
                         lr= opt_cfg['learning_rate'],
                         momentum=opt_cfg["momentum"],
                         dampening=opt_cfg["dampening"])

    return model, optimizer

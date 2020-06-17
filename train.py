# compability methods
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# standard libraries
import argparse
from collections import OrderedDict
import itertools
import json
import logging
import math
import random
import time
# third-party libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import tqdm
# project libraries
import speech
import speech.loader as loader
from speech.models.ctc_model_train import CTC_train
from speech.utils.model_debug import check_nan, log_model_grads, plot_grad_flow_line, plot_grad_flow_bar
from speech.utils.model_debug import save_batch_log_stats, log_batchnorm_mean_std, log_param_grad_norms
from speech.utils.model_debug import get_logger_filename
# TODO, (awni) why does putting this above crash..
import tensorboard_logger as tb

# turn on the set_detect_anomly
torch.autograd.set_detect_anomaly(True)


def run_epoch(model, optimizer, train_ldr, logger, it, avg_loss):
    """
    Performs a forwards and backward pass through the model
    """
    use_log = (logger is not None)
    model_t = 0.0; data_t = 0.0
    end_t = time.time()
    tq = tqdm.tqdm(train_ldr)
    for batch in tq:
        if use_log: logger.info(f"train: ====== Inside run_epoch =======")
        
        temp_batch = list(batch)    # this was added as the batch generator was being exhausted when it was called

        if use_log: save_batch_log_stats(temp_batch, logger)
        if use_log: log_batchnorm_mean_std(model.state_dict(), logger)
 
        start_t = time.time()
        optimizer.zero_grad()
        if use_log: logger.info(f"train: Optimizer zero_grad")

        loss = model.loss(temp_batch)
        if use_log: logger.info(f"train: Loss calculated")

        #print(f"loss value 1: {loss.data[0]}")
        loss.backward()
        if use_log: logger.info(f"train: Backward run ")
        #if use_log: plot_grad_flow_line(model.named_parameters())
        if use_log: plot_grad_flow_bar(model.named_parameters(),  get_logger_filename(logger))
        if use_log: log_param_grad_norms(model.named_parameters(), logger)

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 200)
        if use_log: logger.info(f"train: Grad_norm clipped ")

        loss = loss.item()
        if use_log: logger.info(f"train: loss reassigned ")

        #loss = loss.data[0]

        optimizer.step()
        if use_log: logger.info(f"train: Optimizer step taken")

        prev_end_t = end_t
        end_t = time.time()
        model_t += end_t - start_t
        data_t += start_t - prev_end_t
        if use_log: logger.info(f"train: time calculated ")


        exp_w = 0.99
        avg_loss = exp_w * avg_loss + (1 - exp_w) * loss
        if use_log: logger.info(f"train: Avg loss: {avg_loss}")
        tb.log_value('train_loss', loss, it)
        tq.set_postfix(iter=it, loss=loss,
                avg_loss=avg_loss, grad_norm=grad_norm,
                model_time=model_t, data_time=data_t)
        
       
        if use_log: logger.info(f'train: loss is inf: {loss == float("inf")}')
        if use_log: logger.info(f"train: iter={it}, loss={round(loss,3)}, grad_norm={round(grad_norm,3)}")
        inputs, labels, input_lens, label_lens = model.collate(*temp_batch)
        
        if check_nan(model.parameters()):
            if use_log: logger.error(f"train: labels: {[labels]}, label_lens: {label_lens} state_dict: {model.state_dict()}")
            if use_log: log_model_grads(model.named_parameters(), logger)
        it += 1

    return it, avg_loss

def eval_dev(model, ldr, preproc,  logger):
    losses = []; all_preds = []; all_labels = []
        
    model.set_eval()
    preproc.set_eval()  # this turns off dataset augmentation
    use_log = (logger is not None)
    if use_log: logger.info(f"eval_dev: set_eval ")


    with torch.no_grad():
        for batch in tqdm.tqdm(ldr):
            if use_log: logger.info(f"eval_dev: =====Inside batch loop=====")
            temp_batch = list(batch)
            if use_log: logger.info(f"eval_dev: batch converted")
            preds = model.infer(temp_batch)
            if use_log: logger.info(f"eval_dev: infer call")
            loss = model.loss(temp_batch)
            if use_log: logger.info(f"eval_dev: loss calculated as: {loss.item():0.3f}")
            if use_log: logger.info(f"eval_dev: loss is nan: {math.isnan(loss.item())}")
            losses.append(loss.item())
            if use_log: logger.info(f"eval_dev: loss appended")
            #losses.append(loss.data[0])
            all_preds.extend(preds)
            if use_log: logger.info(f"eval_dev: preds: {preds}")
            all_labels.extend(temp_batch[1])        #add the labels in the batch object
            if use_log: logger.info(f"eval_dev: labels: {temp_batch[1]}")

    model.set_train()
    preproc.set_train()
    if use_log: logger.info(f"eval_dev: set_train")

    loss = sum(losses) / len(losses)
    if use_log: logger.info(f"eval_dev: Avg loss: {loss}")

    results = [(preproc.decode(l), preproc.decode(p))              # decodes back to phoneme labels
               for l, p in zip(all_labels, all_preds)]
    if use_log: logger.info(f"eval_dev: results {results}")
    cer = speech.compute_cer(results)
    print("Dev: Loss {:.3f}, CER {:.3f}".format(loss, cer))
    if use_log: logger.info(f"CER: {cer}")

    return loss, cer

def run(config):

    data_cfg = config["data"]
    log_cfg = config["logger"]
    preproc_cfg = config["preproc"]
    opt_cfg = config["optimizer"]
    model_cfg = config["model"]
    
    use_log = log_cfg["use_log"]
    if use_log:
        # create logger
        logger = logging.getLogger("train_log")
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(log_cfg["log_file"])
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    else:
        logger = None

    # Loaders
    batch_size = opt_cfg["batch_size"]
    preproc = loader.Preprocessor(data_cfg["train_set"], preproc_cfg, logger, 
                  start_and_end=data_cfg["start_and_end"])
    train_ldr = loader.make_loader(data_cfg["train_set"],
                        preproc, batch_size, num_workers=data_cfg["num_workers"])
    dev_ldr = loader.make_loader(data_cfg["dev_set"],
                        preproc, batch_size, num_workers=data_cfg["num_workers"])

    # Model
    model = CTC_train(preproc.input_dim,
                        preproc.vocab_size,
                        model_cfg)
    if model_cfg["load_trained"]:
        model = load_from_trained(model, model_cfg)
        print("Succesfully loaded weights from trained model")
    model.cuda() if use_cuda else model.cpu()

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                    lr=opt_cfg["learning_rate"],
                    momentum=opt_cfg["momentum"],
                    dampening=opt_cfg["dampening"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
        step_size=opt_cfg["sched_step"], 
        gamma=opt_cfg["sched_gamma"])

    if use_log: logger.info(f"train: ====== Model, loaders, optimimzer created =======")
    if use_log: logger.info(f"train: model: {model}")
    if use_log: logger.info(f"train: preproc: {preproc}")
    if use_log: logger.info(f"train: optimizer: {optimizer}")

    # printing to the output file
    print(f"====== Model, loaders, optimimzer created =======")
    print(f"model: {model}")
    print(f"preproc: {preproc}")
    print(f"optimizer: {optimizer}")

    run_state = (0, 0)
    best_so_far = float("inf")
    for e in range(opt_cfg["epochs"]):
        start = time.time()
        scheduler.step()
        for g in optimizer.param_groups:
            print(f'learning rate: {g["lr"]}')
        
        try:
            run_state = run_epoch(model, optimizer, train_ldr, logger, *run_state)
        finally: # used to ensure that plots are closed even if exception raised
            plt.close('all')
            if use_log: logger.error(f"train: ====In finally block====")
            if use_log: logger.error(f"train: state_dict: {model.state_dict()}")
            if use_log: log_model_grads(model.named_parameters(), logger)
        
        if use_log: logger.info(f"train: ====== Run_state finished =======") 
        if use_log: logger.info(f"train: preproc type: {type(preproc)}")

        msg = "Epoch {} completed in {:.2f} (s)."
        print(msg.format(e, time.time() - start))
        if use_log: logger.info(msg.format(e, time.time() - start))

        if use_log: preproc.logger = None
        speech.save(model, preproc, config["save_path"])
        if use_log: logger.info(f"train: ====== model saved =======")
        if use_log: preproc.logger = logger

        dev_loss, dev_cer = eval_dev(model, dev_ldr, preproc, logger)
        if use_log: logger.info(f"train: ====== eval_dev finished =======")

        # Log for tensorboard
        tb.log_value("dev_loss", dev_loss, e)
        tb.log_value("dev_cer", dev_cer, e)
           
        if use_log: preproc.logger = None 
        # Save the best model on the dev set
        if dev_cer < best_so_far:
            best_so_far = dev_cer
            speech.save(model, preproc,
                    config["save_path"], tag="best")
        if use_log: preproc.logger = logger


def load_from_trained(model, model_cfg):
    """
    loads the model with pretrained weights from the model in
    model_cfg["trained_path"]
    Arguments:
        model (torch model)
        model_cfg (dict)
    """
    trained_model = torch.load(model_cfg["trained_path"], map_location=torch.device('cpu'))
    trained_state_dict = trained_model.state_dict()
    trained_state_dict = filter_state_dict(trained_state_dict, remove_layers=model_cfg["remove_layers"])
    model_state_dict = model.state_dict()
    model_state_dict.update(trained_state_dict)
    model.load_state_dict(model_state_dict)
    return model


def filter_state_dict(state_dict, remove_layers=[]):
    """
    filters the inputted state_dict by removing the layers specified
    in remove_layers
    Arguments:
        state_dict (OrderedDict): state_dict of pytorch model
        remove_layers (list(str)): list of layers to remove 
    """

    state_dict = OrderedDict(
        {key:value for key,value in state_dict.items() 
        if key not in remove_layers}
        )
    return state_dict




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Train a speech model.")
    parser.add_argument("config",
        help="A json file with the training configuration.")
    parser.add_argument("--deterministic", default=False,
        action="store_true",
        help="Run in deterministic mode (no cudnn). Only works on GPU.")
    args = parser.parse_args()

    with open(args.config, 'r') as fid:
        config = json.load(fid)

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    tb.configure(config["save_path"])

    use_cuda = torch.cuda.is_available()

    if use_cuda and args.deterministic:
        torch.backends.cudnn.enabled = False
    run(config)

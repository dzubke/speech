# compability methods
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# standard libraries
import argparse
import os
import itertools
import json
import logging
import math
import random
import time
# third-party libraries
import functions.ctc as ctc #awni hannun's ctc bindings
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import tqdm
import yaml
# project libraries
import speech
import speech.loader as loader
from speech.models.ctc_model_train import CTC_train
from speech.utils.io import read_pickle, write_pickle, load_from_trained, load_config
from speech.utils.model_debug import check_nan_params_grads, log_model_grads, plot_grad_flow_line, plot_grad_flow_bar
from speech.utils.model_debug import save_batch_log_stats, log_batchnorm_mean_std, log_param_grad_norms
from speech.utils.model_debug import get_logger_filename, log_cpu_mem_disk_usage



def run_epoch(model, optimizer, train_ldr, logger, debug_mode, tbX_writer, iter_count, avg_loss):
    """
    Performs a forwards and backward pass through the model
    Arguments
        iter_count - int: count of iterations
    """
    use_log = (logger is not None)
    model_t = 0.0; data_t = 0.0
    end_t = time.time()
    tq = tqdm.tqdm(train_ldr)
    log_modulus = 5  # limits certain logging function to only log every "log_modulus" iterations
    exp_w = 0.985        # exponential weight for exponential moving average loss        
    avg_grad_norm = 0.0

    # model compatibility for using multiple gpu's
    multi_gpu = isinstance(model, torch.nn.DataParallel)
    if multi_gpu:
        model_module = model.module
    else: 
        model_module = model

    for batch in tq:
        if use_log: logger.info(f"train: ====== Iteration: {iter_count} in run_epoch =======")
        
        temp_batch = list(batch)    # this was added as the batch generator was being exhausted when it was called

        if use_log: 
            if debug_mode:  
                save_batch_log_stats(temp_batch, logger)
                log_batchnorm_mean_std(model_module.state_dict(), logger)
 
        start_t = time.time()
        optimizer.zero_grad()
        if use_log: logger.info(f"train: Optimizer zero_grad")

        # calcuating the loss outside of model.loss to allow multi-gpu use
        inputs, labels, input_lens, label_lens = model_module.collate(*temp_batch)
        out, rnn_args = model(inputs, softmax=False)
        loss_fn = ctc.CTCLoss()
        loss = loss_fn(out, labels, input_lens, label_lens)
        
        if use_log: logger.info(f"train: Loss calculated")

        #print(f"loss value 1: {loss.data[0]}")
        loss.backward()
        if use_log: logger.info(f"train: Backward run ")
        if use_log: 
            if debug_mode: 
                plot_grad_flow_bar(model_module.named_parameters(),  get_logger_filename(logger))
                log_param_grad_norms(model_module.named_parameters(), logger)

        grad_norm = nn.utils.clip_grad_norm_(model_module.parameters(), 200)
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

        if iter_count == 0:
            avg_loss = loss
            avg_grad_norm = grad_norm
        else: 
            avg_loss = exp_w * avg_loss + (1 - exp_w) * loss
            avg_grad_norm = exp_w * avg_grad_norm + (1 - exp_w) * grad_norm
        if use_log: logger.info(f"train: Avg loss: {avg_loss}")
        tbX_writer.add_scalars('train/loss', {"loss": loss}, iter_count)
        tbX_writer.add_scalars('train/loss', {"avg_loss": avg_loss}, iter_count)
        tbX_writer.add_scalars('train/grad', {"grad_norm": avg_loss}, iter_count)
        tq.set_postfix(iter=iter_count, loss=loss, 
                avg_loss=avg_loss, grad_norm=grad_norm,
                model_time=model_t, data_time=data_t)
        
        if use_log: logger.info(f'train: loss is inf: {loss == float("inf")}')
        if use_log: logger.info(f"train: iter={iter_count}, loss={round(loss,3)}, grad_norm={round(grad_norm,3)}")
        
        if check_nan_params_grads(model_module.parameters()):
            if use_log:
                logger.error(f"train: labels: {[labels]}, label_lens: {label_lens} state_dict: {model_module.state_dict()}")
                log_model_grads(model_module.named_parameters(), logger)
                save_batch_log_stats(temp_batch, logger)
                log_param_grad_norms(model_module.named_parameters(), logger)
                plot_grad_flow_bar(model_module.named_parameters(), get_logger_filename(logger))
            debug_mode = True
            torch.autograd.set_detect_anomaly(True)

        if iter_count % log_modulus == 0:
            if use_log: log_cpu_mem_disk_usage(logger)
        
        iter_count += 1

    return iter_count, avg_loss

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
    debug_mode = log_cfg["debug_mode"]
    
    if debug_mode: torch.autograd.set_detect_anomaly(True)

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

    tbX_writer = SummaryWriter(logdir=config["save_path"])
    
    # Load previous train state: dict with contents:
    #   {'start_epoch': int, 'run_state': (int, float), 'best_so_far': float, 'learning_rate': float}
    train_state_path = os.path.join(config["save_path"], "train_state.pickle")
    if os.path.exists(train_state_path):
        print(f"load train_state from: {train_state_path}")
        train_state = read_pickle(train_state_path)
    else:   # if train_path doesn't exist, create empty dict to load from config
        print(f"load train_state from config")
        train_state = dict()
    # the get-statements will load from train_state if key exists, and from opt_cfg otherwise
    run_state = train_state.get('run_state',  opt_cfg['run_state'])
    best_so_far = train_state.get('best_so_far', opt_cfg['best_so_far'])
    start_epoch =  train_state.get('start_epoch', # train_state used to use 'next_epoch' 
                                    train_state.get('next_epoch', opt_cfg['start_epoch'])
    )
    learning_rate = opt_cfg['learning_rate']

    # Loaders
    batch_size = opt_cfg["batch_size"]
    preproc = loader.Preprocessor(data_cfg["train_set"], preproc_cfg, logger, 
                  start_and_end=data_cfg["start_and_end"])
    train_ldr = loader.make_loader(data_cfg["train_set"],
                        preproc, batch_size, num_workers=data_cfg["num_workers"])
    dev_ldr_dict = dict() # dict that includes all the dev_loaders
    for dev_name, dev_path in data_cfg["dev_sets"].items():
        dev_ldr = loader.make_loader(dev_path,
                        preproc, batch_size, num_workers=data_cfg["num_workers"])
        dev_ldr_dict.update({dev_name: dev_ldr})

    # Model
    model = CTC_train(preproc.input_dim,
                        preproc.vocab_size,
                        model_cfg)
    if model_cfg["load_trained"]:
        model = load_from_trained(model, model_cfg)
        print(f"Succesfully loaded weights from trained model: {model_cfg['trained_path']}")
    if model_cfg["multi_gpu"]:
        assert torch.cuda.device_count() > 1, "multi_gpu selected but less than on GPU available"
        model = torch.nn.DataParallel(model)
        model_module = model.module
    else:
        # allows for compatbility with data-parallel models
        model_module = model
    model.cuda() if use_cuda else model.cpu()

    # Optimizer
    optimizer = torch.optim.SGD(model_module.parameters(),
                    lr=learning_rate,   # from train_state or opt_config
                    momentum=opt_cfg["momentum"],
                    dampening=opt_cfg["dampening"])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
        step_size=opt_cfg["sched_step"], 
        gamma=opt_cfg["sched_gamma"])

    if use_log: logger.info(f"train: ====== Model, loaders, optimimzer created =======")
    if use_log: logger.info(f"train: model: {model}")
    if use_log: logger.info(f"train: preproc: {preproc}")
    if use_log: logger.info(f"train: optimizer: {optimizer}")
    if use_log: logger.info(f"train: config: {config}")

    # printing to the output file
    print(f"====== Model, loaders, optimimzer created =======")
    print(f"model: {model}")
    print(f"preproc: {preproc}")
    print(f"optimizer: {optimizer}")
    print(f"config: {config}")

    for epoch in range(start_epoch, opt_cfg["epochs"]):
        if use_log: logger.info(f"Starting epoch: {epoch}")
        start = time.time()
        for group in optimizer.param_groups:
            print(f'learning rate: {group["lr"]}')
            if use_log: logger.info(f"train: learning rate: {group['lr']}")
        

        try:
            run_state = run_epoch(model, optimizer, train_ldr, logger, debug_mode, tbX_writer, *run_state)
        except Exception as err:
            if use_log: logger.error(f"Exception raised: {err}")
            if use_log: logger.error(f"train: ====In except block====")
            if use_log: logger.error(f"train: state_dict: {model_module.state_dict()}")
            if use_log: log_model_grads(model_module.named_parameters(), logger)
            raise Exception('Failure in run_epoch').with_traceback(err.__traceback__)
        finally: # used to ensure that plots are closed even if exception raised
            plt.close('all')
    
        # update the learning rate
        lr_scheduler.step()       
 
        if use_log: logger.info(f"train: ====== Run_state finished =======") 
        if use_log: logger.info(f"train: preproc type: {type(preproc)}")

        msg = "Epoch {} completed in {:.2f} (hr)."
        epoch_time_hr = (time.time() - start)/60/60
        print(msg.format(epoch, epoch_time_hr))
        if use_log: logger.info(msg.format(epoch, epoch_time_hr))
        tbX_writer.add_scalars('train/stats', {"epoch_time_hr": epoch_time_hr}, epoch)

        # the logger needs to be removed to save the model
        if use_log: preproc.logger = None
        speech.save(model_module, preproc, config["save_path"])
        if use_log: logger.info(f"train: ====== model saved =======")
        if use_log: preproc.logger = logger

        # creating the dictionaries that hold the PER and loss values
        dev_loss_dict = dict()
        dev_per_dict = dict()
        # iterating through the dev-set loaders to calculate the PER/loss
        for dev_name, dev_ldr in dev_ldr_dict.items():
            print(f"evaluating devset: {dev_name}")
            if use_log: logger.info(f"train: === evaluating devset: {dev_name} ==")
            dev_loss, dev_per = eval_dev(model_module, dev_ldr, preproc, logger)

            dev_loss_dict.update({dev_name: dev_loss})
            dev_per_dict.update({dev_name: dev_per})

            if use_log: logger.info(f"train: ====== eval_dev {dev_name} finished =======")
            
            # Save the best model on the dev set
            if dev_name == data_cfg['dev_set_save_reference']:
                print(f"dev_reference {dev_name}: current PER: {dev_per} vs. best_so_far: {best_so_far}")
                logger.info(f"dev_reference {dev_name}: current PER: {dev_per} vs. best_so_far: {best_so_far}")
                if dev_per < best_so_far:
                    if use_log: preproc.logger = None   # remove the logger to save the model
                    best_so_far = dev_per
                    speech.save(model_module, preproc,
                            config["save_path"], tag="best")
                    if use_log: preproc.logger = logger
                    
                    print(f"UPDATED: best_model based on PER {best_so_far} for {dev_name} devset")
                    logger.info(f"model saved based per on: {dev_name} dataset")
            
        tbX_writer.add_scalars('dev/loss', dev_loss_dict, epoch)
        tbX_writer.add_scalars('dev/per', dev_per_dict, epoch)
        learning_rate = list(optimizer.param_groups)[0]["lr"]
        # save the current state of training
        train_state = {"start_epoch": epoch + 1, 
                       "run_state": run_state, 
                       "best_so_far": best_so_far,
                       "learning_rate": learning_rate}
        write_pickle(os.path.join(config["save_path"], "train_state.pickle"), train_state)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Train a speech model.")
    parser.add_argument("config",
        help="A json file with the training configuration.")
    parser.add_argument("--deterministic", default=False,
        action="store_true",
        help="Run in deterministic mode (no cudnn). Only works on GPU.")
    args = parser.parse_args()

    config = load_config(args.config)

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    use_cuda = torch.cuda.is_available()

    if use_cuda and args.deterministic:
        torch.backends.cudnn.enabled = False
    run(config)

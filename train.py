from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import random
import time
import torch
import torch.nn as nn
import torch.optim
import tqdm
import pickle
import itertools

import speech
import speech.loader as loader
import speech.models as models

# TODO, (awni) why does putting this above crash..
import tensorboard_logger as tb

def run_epoch(model, optimizer, train_ldr, it, avg_loss):
    r"""This performs a forwards and backward pass through the NN

    Arguements
    ------------
    model
    optimizer
    train_ldr
    it: int
        the current iteration of the training model
    avg_loss

    Returns
    ------------
    it: int
        the current iteration of the model after the epoch has run

    avg_loss: 

    """
    model_t = 0.0; data_t = 0.0
    end_t = time.time()
    tq = tqdm.tqdm(train_ldr)
    for batch in tq:
        # print(batch)
        temp_batch = list(batch)
        start_t = time.time()
        optimizer.zero_grad()
        loss = model.loss(temp_batch)
        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 200)
        loss = loss.data[0]

        optimizer.step()
        prev_end_t = end_t
        end_t = time.time()
        model_t += end_t - start_t
        data_t += start_t - prev_end_t

        exp_w = 0.99
        avg_loss = exp_w * avg_loss + (1 - exp_w) * loss
        tb.log_value('train_loss', loss, it)
        tq.set_postfix(iter=it, loss=loss,
                avg_loss=avg_loss, grad_norm=grad_norm,
                model_time=model_t, data_time=data_t)
        it += 1

    return it, avg_loss

def eval_dev(model, ldr, preproc):
    losses = []; all_preds = []; all_labels = []

    model.set_eval()

    for batch in tqdm.tqdm(ldr):
        # debugging print statements
        #print(f"ldr len: {ldr.__len__()}, ldr contents: {[i for i in  itertools.islice(ldr,0,1)]}")
        #print(f"# of elem in zip: {len([j for i in  itertools.islice(ldr,0,1) for j in i])}, len zip elem 1 aka # of arrays: {len([k for i in itertools.islice(ldr,0,1) for j in itertools.islice(i,0,1) for k in j])}, len of zip elem 2 aka # of phon lists: {len([k for i in itertools.islice(ldr,1) for j in itertools.islice(i,1,2) for k in j])}")
        #print(f"shape of first 5 arrays: {[k.shape for i in itertools.islice(ldr,0,1) for j in itertools.islice(i,0,1) for k in itertools.islice(j,0,5)]}, # of phones in first 5 lists: {[len(k) for i in itertools.islice(ldr,1) for j in itertools.islice(i,1,2) for k in itertools.islice(j,0,5)]}")
        #print(f"# of elem in zip: {len([j for i in  batch for j in i])}, len zip elem 1 aka # of arrays: {len([k for i in batch for j in itertools.islice(i,0,1) for k in j])}, len of zip elem 2 aka # of phon lists: {len([k for i in batch for j in itertools.islice(i,1,2) for k in j])}")
        
        temp_batch = list(batch)
        preds = model.infer(temp_batch)
        loss = model.loss(temp_batch)
        losses.append(loss.data[0])
        all_preds.extend(preds)
        all_labels.extend(temp_batch[1])

    model.set_train()

    loss = sum(losses) / len(losses)
    results = [(preproc.decode(l), preproc.decode(p))
               for l, p in zip(all_labels, all_preds)]
    cer = speech.compute_cer(results)
    print("Dev: Loss {:.3f}, CER {:.3f}".format(loss, cer))
    return loss, cer

def run(config):

    opt_cfg = config["optimizer"]
    data_cfg = config["data"]
    model_cfg = config["model"]

    # Loaders
    batch_size = opt_cfg["batch_size"]
    preproc = loader.Preprocessor(data_cfg["train_set"],
                  start_and_end=data_cfg["start_and_end"], use_mfcc=data_cfg["use_mfcc"])
    train_ldr = loader.make_loader(data_cfg["train_set"],
                        preproc, batch_size)
    dev_ldr = loader.make_loader(data_cfg["dev_set"],
                        preproc, batch_size)

    # Model
    # dustin: I don't understand how the line below works. I can infer what it does but am not sure how it does it
    model_class = eval("models." + model_cfg["class"])
    model = model_class(preproc.input_dim,
                        preproc.vocab_size,
                        model_cfg)
    model.cuda() if use_cuda else model.cpu()

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                    lr=opt_cfg["learning_rate"],
                    momentum=opt_cfg["momentum"])

    run_state = (0, 0)
    best_so_far = float("inf")
    for e in range(opt_cfg["epochs"]):
        start = time.time()
        # print([i for ex in train_ldr for i in ex])

        # the if-statement pickles the run_state_object to be retreived if there are bugs in the 
        # in any of the lines below run_epoch, so that you don't have to wait for run_epoch to complete
        # before encountering the bugs
        
        if model_cfg["load_model"] and e==0: 
            with open(model_cfg["it_loss_path"], 'rb') as f:
                run_state = pickle.load(f)  # loads tuple of previous iteration and avg_loss: (it and avg_loss)
                print(run_state)
                with open(model_cfg["model_path"], 'rb') as model_f:
                    model = torch.load(model_f)
                    print("inside the load model")
            print(run_state)

        
        else:
            run_state = run_epoch(model, optimizer, train_ldr, *run_state)
            with open(model_cfg["it_loss_path"],'wb') as model_fn:
                pickle.dump(run_state, model_fn)

        msg = "Epoch {} completed in {:.2f} (s)."
        print(msg.format(e, time.time() - start))

        dev_loss, dev_cer = eval_dev(model, dev_ldr, preproc)

        # Log for tensorboard
        tb.log_value("dev_loss", dev_loss, e)
        tb.log_value("dev_cer", dev_cer, e)

        speech.save(model, preproc, config["save_path"])

        # Save the best model on the dev set
        if dev_cer < best_so_far:
            best_so_far = dev_cer
            speech.save(model, preproc,
                    config["save_path"], tag="best")

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


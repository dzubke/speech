# 
import logging
import argparse
from datetime import date

import torch
import tqdm

from train import eval_dev
import speech
import speech.loader as loader


def main(model_path:str, json_path:str, use_cuda:bool):
    """
    runs the eval_dev loop in train continually while saving
    relevant date to a log file
    """

    # create logger
    logger = logging.getLogger("eval-dev_log")
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler("eval-dev_"+str(date.today())+".log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    #loading model and preproc
    model, preproc = speech.load(model_path, tag="best")
    
    # creating loader
    dev_ldr = loader.make_loader(json_path,
                        preproc, batch_size=1)
    
    iterations = 500

    for i in range(iterations):
        logger.info(f"\n=================================================\n")
        logger.info(f"Iteration: {i}")

        loss, cer = eval_dev(model, dev_ldr, preproc, logger)


def eval_dev(model, ldr, preproc, logger):
    losses = []; all_preds = []; all_labels = []

    model.set_eval()
    preproc.set_eval()
    logger.info(f"--------set_eval and entering loop---------")

    with torch.no_grad():
        for batch in tqdm.tqdm(ldr):
            temp_batch = list(batch)
            logger.info(f"temp_batch created as list")
            preds = model.infer(temp_batch)
            logger.info(f"model.infer called with {len(preds[0])}")
            loss = model.loss(temp_batch)
            logger.info(f"loss calculated as: {loss.item()}")
            losses.append(loss.item())
            #losses.append(loss.data[0])
            logger.info(f"loss appended")
            all_preds.extend(preds)
            logger.info(f"preds extended")
            all_labels.extend(temp_batch[1])        #add the labels in the batch object
            logger.info(f"labels extended")


    model.set_train()
    preproc.set_train()       
    logger.info(f"set to train")

    loss = sum(losses) / len(losses)
    logger.info(f"Avg loss: {loss}")
    results = [(preproc.decode(l), preproc.decode(p))              # decodes back to phoneme labels
               for l, p in zip(all_labels, all_preds)]
    logger.info(f"results {results}")
    cer = speech.compute_cer(results)
    logger.info(f"CER: {cer}")

    return loss, cer

if __name__=="__main__":

    parser = argparse.ArgumentParser(
            description="Testing the eval_dev loop")

    parser.add_argument("--model-path", type=str,
        help="path to the directory with the model and preproc object.")
    parser.add_argument("--json-path", type=str,
        help="Path to the data json file eval_dev will be called upon.")
    args = parser.parse_args()

    main(args.model_path, args.json_path, args.use_cuda)
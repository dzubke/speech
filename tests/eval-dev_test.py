# standard libraries 
import logging
import argparse
import math
# third-party libraries
import torch
import tqdm
# project libraries
from train import eval_dev
import speech
import speech.loader as loader


def main(model_path:str, json_path:str, use_cuda:bool, log_name:str, use_augmentation:bool):
    """
    runs the eval_dev loop in train continually while saving
    relevant date to a log file
    """

    # create logger
    logger = logging.getLogger("eval-dev_log")
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_name+".log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    
    #loading model and preproc
    model, preproc = speech.load(model_path, tag="best")
    model.cuda() if use_cuda else model.cpu()
    print(f"spec_aug status:{preproc.spec_augment}") 
    # creating loader
    dev_ldr = loader.make_loader(json_path,
                        preproc, batch_size=1)
    
    iterations = 500
    
    logger.info("============= Trial info ============")
    logger.info(f"model path: {model_path}")
    logger.info(f"json path: {json_path}")
    logger.info(f"use_augmentation: {use_augmentation}")
    logger.info(f"preproc: {preproc}")
    logger.info(f"model: {model}")

    for i in range(iterations):
        logger.info(f"\n=================================================\n")
        logger.info(f"Iteration: {i}")

        loss, cer = eval_dev(model, dev_ldr, preproc, logger, use_augmentation)


def eval_dev(model, ldr, preproc, logger, use_augmentation):
    losses = []; all_preds = []; all_labels = []

    model.set_eval()
    if not use_augmentation:
        print("prepoc set to eval")
        preproc.set_eval()
    logger.info(f"--------set_eval and entering loop---------")

    with torch.no_grad():
        for batch in tqdm.tqdm(ldr):
            temp_batch = list(batch)
            logger.info(f"temp_batch created as list")
            preds = model.infer(temp_batch)
            logger.info(f"model.infer called with {len(preds[0])}")
            loss = model.loss(temp_batch)
            logger.info(f"loss calculated as: {loss.item():0.3f}")
            logger.info(f"loss is nan: {math.isnan(loss.item())}")
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
    parser.add_argument("--use-cuda", action='store_true', default=False,
        help="sets the model to use cuda in inference")
    parser.add_argument("--use-augmentation", action='store_true', default=False,
        help="if true, data augmentation is used during eval loop.")
    parser.add_argument("--model-path", type=str,
        help="path to the directory with the model and preproc object.")
    parser.add_argument("--json-path", type=str,
        help="Path to the data json file eval_dev will be called upon.")
    parser.add_argument("--log-name", type=str,
        help="Name of log file created for logs.")
    args = parser.parse_args()

    main(args.model_path, args.json_path, args.use_cuda, args.log_name, args.use_augmentation)

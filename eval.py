from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import torch
import tqdm
import speech
import speech.loader as loader

def eval_loop(model, ldr):
    all_preds = []; all_labels = []; all_preds_dist=[]
    for batch in tqdm.tqdm(ldr):
        #dustin: my modification because the iteratable batch was being exhausted when it was called
        temp_batch = list(batch)
        print(f"# examples in batch: {len(temp_batch)}")   
        print(f"batch : {len(temp_batch[0])}")   
        print(f"batch: {temp_batch}")
        #print(f"numpy [0][0]: {type(temp_batch[0])}")   
        # print(f"temp_bach: {temp_batch}")
        preds = model.infer(temp_batch)
        #preds_dist, prob_dist = model.infer_distribution(temp_batch, 5)
        all_preds.extend(preds)
        all_labels.extend(temp_batch[1])
        #all_preds_dist.extend(((preds_dist, temp_batch[1]),prob_dist))
    # print(f"all labels: {all_labels}, all preds: {all_preds}")
    return list(zip(all_labels, all_preds)) #, all_preds_dist

def run(model_path, dataset_json,
        batch_size=1, tag="best",
        out_file=None):

    use_cuda = torch.cuda.is_available()

    model, preproc = speech.load(model_path, tag=tag)
    # print(f"preproc dean shape: {len(preproc.mean)}, preproc std shape: {len(preproc.std)}")
    ldr =  loader.make_loader(dataset_json,
            preproc, batch_size)
    # print(f"ldr: {[i for i in ldr]}")
    model.cuda() if use_cuda else model.cpu()
    model.set_eval()

    results = eval_loop(model, ldr)
    print(f"number of examples: {len(results)}")
    #print(f"results_dist: {results_dist}")
    #results_dist = [[(preproc.decode(pred[0]), preproc.decode(pred[1]), prob)] 
    #                for example_dist in results_dist
    #                for pred, prob in example_dist]
    #print(f"results_dist: {results_dist}")
    results = [(preproc.decode(label), preproc.decode(pred))
               for label, pred in results]
    print(f"results: {results}")
    per = speech.compute_per(results)



    print("PER for 48-phonemes {:.3f}".format(per))

    if out_file is not None:
        with open(out_file, 'w') as fid:
            for label, pred in results:
                
                res = {'prediction' : pred,
                       'label' : label}
                json.dump(res, fid)
                fid.write("\n") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Eval a speech model.")

    parser.add_argument("model",
        help="A path to a stored model.")
    parser.add_argument("dataset",
        help="A json file with the dataset to evaluate.")
    parser.add_argument("--last", action="store_true",
        help="Last saved model instead of best on dev set.")
    parser.add_argument("--save",
        help="Optional file to save predicted results.")
    args = parser.parse_args()

    run(args.model, args.dataset,
        tag=None if args.last else "best",
        out_file=args.save)

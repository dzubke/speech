# standard libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import json
# third-party libraries
import torch
import tqdm
# project libraries
import speech
import speech.loader as loader
from speech.utils.io import read_data_json

def eval_loop(model, ldr):
    all_preds = []; all_labels = []; all_preds_dist=[]
    with torch.no_grad():
        for batch in tqdm.tqdm(ldr):
            temp_batch = list(batch)
            preds = model.infer(temp_batch)
            #preds_dist, prob_dist = model.infer_distribution(temp_batch, 5)
            all_preds.extend(preds)
            all_labels.extend(temp_batch[1])
            #all_preds_dist.extend(((preds_dist, temp_batch[1]),prob_dist))
    return list(zip(all_labels, all_preds)) #, all_preds_dist


def run(model_path, dataset_json, batch_size=1, tag="best", add_filename=False, out_file=None):
    """
    calculates the  distance between the predictions from
    the model in model_path and the labels in dataset_json

    Arguments:
        added_filename - bool: if true, the filename is added
            to the output json
        tag - str: if best,  the "best_model" is used 
            if not, "model" is used. 
    """

    use_cuda = torch.cuda.is_available()
    model, preproc = speech.load(model_path, tag=tag)
    ldr =  loader.make_loader(dataset_json,
            preproc, batch_size)
    model.cuda() if use_cuda else model.cpu()
    model.set_eval()
    print(f"spec_augment before set_eval: {preproc.spec_augment}")
    preproc.set_eval()
    preproc.use_log = False
    print(f"spec_augment after set_eval: {preproc.spec_augment}")


    results = eval_loop(model, ldr)
    print(f"number of examples: {len(results)}")
    #results_dist = [[(preproc.decode(pred[0]), preproc.decode(pred[1]), prob)] 
    #                for example_dist in results_dist
    #                for pred, prob in example_dist]
    results = [(preproc.decode(label), preproc.decode(pred))
               for label, pred in results]
    cer = speech.compute_cer(results, verbose=True)

    print("PER {:.3f}".format(cer))

    output_results = []
    if out_file is not None:
        for label, pred in results: 
            if add_filename:
                filename = match_filename(label, dataset_json)
                PER = speech.compute_cer([(label,pred)], verbose=False)
                res = {'filename': filename,
                    'prediction' : pred,
                    'label' : label,
                    'PER': round(PER, 3)}
            else:   
                res = {'prediction' : pred,
                    'label' : label}
            output_results.append(res)
        
        # if including filename, add the suffix "_fn" before extension
        if add_filename: 
            out_file, ext = os.path.splitext(out_file)
            out_file = out_file + "_fn" + ext
            output_results = sorted(output_results, key=lambda x: x['PER'], reverse=True) 
        with open(out_file, 'w') as fid:
            for sample in output_results:
                json.dump(sample, fid)
                fid.write("\n") 

def match_filename(label:list, dataset_json:str) -> str:
    """
    returns the filename in dataset_json that matches
    the phonemes in label
    """
    dataset = read_data_json(dataset_json)
    matches = []
    for sample in dataset:
        if sample['text'] == label:
            matches.append(sample["audio"])
    
    assert len(matches) < 2, f"multiple matches found {matches} for label {label}"
    assert len(matches) >0, f"no matches found for {label}"
    match = matches[0]
    return match
        


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
    parser.add_argument("--filename", action="store_true", default=False,
        help="Include the filename for each sample in the json output.")
    args = parser.parse_args()

    run(args.model, args.dataset, tag=None if args.last else "best", 
        add_filename=args.filename, out_file=args.save)

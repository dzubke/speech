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
    all_confidence = []
    with torch.no_grad():
        for batch in tqdm.tqdm(ldr):
            temp_batch = list(batch)
            preds, confidence = model.infer_confidence(temp_batch)
            #preds_dist, prob_dist = model.infer_distribution(temp_batch, 5)
            all_preds.extend(preds)
            all_confidence.extend(confidence)
            all_labels.extend(temp_batch[1])
            #all_preds_dist.extend(((preds_dist, temp_batch[1]),prob_dist))
    return list(zip(all_labels, all_preds, all_confidence)) #, all_preds_dist


def run(model_path, dataset_json, batch_size=8, tag="best", 
    add_filename=False, add_maxdecode=False, formatted=False, 
    config_path = None, out_file=None):
    """
    calculates the  distance between the predictions from
    the model in model_path and the labels in dataset_json

    Arguments:
        tag - str: if best,  the "best_model" is used. if not, "model" is used. 
        add_filename - bool: if true, the filename is added to the output json
        add_maxdecode - bool: if true, predictions from the max decoder will be added
    """

    use_cuda = torch.cuda.is_available()
    model, preproc = speech.load(model_path, tag=tag)
    preproc.update()

    if config_path is not None:
        with open(config_path, 'r') as fid:
            config = json.load(fid)
        new_preproc = loader.Preprocessor(dataset_json, config["preproc"], start_and_end=config["data"]["start_and_end"])
        new_preproc.mean, new_preproc.std = preproc.mean, preproc.std
        new_preproc.int_to_char, new_preproc.char_to_int = preproc.int_to_char, preproc.char_to_int
        print(f"preproc attr: {preproc}")
        print(f"preproc sum of mean, std: {preproc.mean.shape},{preproc.std.shape}")
        print(f"new_preproc sum of mean, std: {new_preproc.mean.sum()},{new_preproc.std.sum()}")
        print(f"new preproc attr: {new_preproc}")
        preproc = new_preproc
    
    ldr =  loader.make_loader(dataset_json,
            preproc, batch_size)
    model.cuda() if use_cuda else model.cpu()
    model.set_eval()
    print(f"preproc train_status before set_eval: {preproc.train_status}")
    preproc.set_eval()
    preproc.use_log = False
    print(f"preproc train_status after set_eval: {preproc.train_status}")


    results = eval_loop(model, ldr)
    print(f"number of examples: {len(results)}")
    #results_dist = [[(preproc.decode(pred[0]), preproc.decode(pred[1]), prob)] 
    #                for example_dist in results_dist
    #                for pred, prob in example_dist]
    results = [(preproc.decode(label), preproc.decode(pred), conf)
               for label, pred, conf in results]
    #maxdecode_results = [(preproc.decode(label), preproc.decode(pred))
    #           for label, pred in results]
    cer = speech.compute_cer(results, verbose=True)

    print("PER {:.3f}".format(cer))
    
    if out_file is not None:
        compile_save(results, dataset_json, out_file, formatted, add_filename)


def compile_save(results, dataset_json, out_file, formatted=False, add_filename=False):
    output_results = []
    if formatted:
        format_save(results, dataset_json, out_file)
    else: 
        json_save(results, dataset_json, out_file, add_filename)
        

def format_save(results, dataset_json, out_file):
    out_file = create_filename(out_file, "compare", "txt") 
    print(f"file saved to: {out_file}")
    with open(out_file, 'w') as fid:
        write_list = list()
        for label, pred, conf in results:
            filepath, order = match_filename(label, dataset_json, return_order=True)
            filename = os.path.splitext(os.path.split(filepath)[1])[0]
            PER, (dist, length) = speech.compute_cer([(label,pred)], verbose=False, dist_len=True)
            write_list.append({"order":order, "filename":filename, "label":label, "preds":pred,
            "metrics":{"PER":round(PER,3), "dist":dist, "len":length, "confidence":round(conf, 3)}})
        write_list = sorted(write_list, key=lambda x: x['order'])
            
        for write_dict in write_list: 
            fid.write(f"{write_dict['filename']}\n") 
            fid.write(f"label: {' '.join(write_dict['label']).upper()}\n") 
            fid.write(f"preds: {' '.join(write_dict['preds']).upper()}\n")
            PER, dist = write_dict['metrics']['PER'], write_dict['metrics']['dist'] 
            length, conf = write_dict['metrics']['len'], write_dict['metrics']['confidence']
            fid.write(f"metrics: PER: {PER}, dist: {dist}, len: {length}, conf: {conf}\n")
            fid.write("\n") 

def json_save(results, dataset_json, out_file, add_filename):
    output_results = []
    for label, pred, conf in results: 
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
        out_file = create_filename(out_file, "pred-fn", "json")
        output_results = sorted(output_results, key=lambda x: x['PER'], reverse=True) 
    else: 
        out_file = create_filename(out_file, "pred", "json")
    print(f"file saved to: {out_file}") 
    with open(out_file, 'w') as fid:
        for sample in output_results:
            json.dump(sample, fid)
            fid.write("\n") 

def match_filename(label:list, dataset_json:str, return_order=False) -> str:
    """
    returns the filename in dataset_json that matches
    the phonemes in label
    """
    dataset = read_data_json(dataset_json)
    matches = []
    for i, sample in enumerate(dataset):
        if sample['text'] == label:
            matches.append(sample["audio"])
            order = i
    
    assert len(matches) < 2, f"multiple matches found {matches} for label {label}"
    assert len(matches) >0, f"no matches found for {label}"
    if return_order:
        output = (matches[0], order)
    else:
        output = matches[0]
    return output

def create_filename(base_fn, suffix, ext):
    if "." in ext:
        ext = ext.replace(".", "")
    return base_fn + "_" + suffix + os.path.extsep + ext  


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
    parser.add_argument("--maxdecode", action="store_true", default=False,
        help="Include the filename for each sample in the json output.")
    parser.add_argument("--filename", action="store_true", default=False,
        help="Include the filename for each sample in the json output.")
    parser.add_argument("--formatted", action="store_true", default=False,
        help="Output will be written to file in a cleaner format.")
    parser.add_argument("--config-path", type=str, default=None,
        help="Replace the preproc from model path a  preproc copy using the config file.")
    args = parser.parse_args()

    run(args.model, args.dataset, tag=None if args.last else "best", 
        add_filename=args.filename, add_maxdecode=args.maxdecode, 
        formatted=args.formatted, config_path=args.config_path, out_file=args.save)

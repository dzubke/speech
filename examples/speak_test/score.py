from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import editdistance
import json


def save_predictions(data, out_file):
    if out_file is not None:
        with open(out_file, 'w') as fid:
            for example in data:
                json.dump(example, fid)
                fid.write("\n")

def save_distance(data, out_file):
    if out_file is not None:
        with open(out_file, 'w') as fid:
            for d in data: 
                dist =editdistance.eval(d['label'], d['prediction'])
                total = len(d['label'])
                PER = dist/total
                d_dict = {"predi": d['prediction'],
                          "label": d['label'], 
                          "dist" : dist,
                          "label_length": total,
                          "PER": PER}
                json.dump(d_dict, fid)
                fid.write("\n")


if __name__ == "__main__":
    ## the format of command is:
    # python score.py <path_to_input:predictions_json> <path_to_output:score_json>
    parser = argparse.ArgumentParser(
        description="PER on Timit with reduced phoneme set.")

    parser.add_argument("data_json",
        help="JSON with the transcripts.")

    parser.add_argument("score_json",
        help="the path to the outputed score_json JSON with the transcripts.")

    args = parser.parse_args()

    with open(args.data_json, 'r') as fid:
        data = [json.loads(l) for l in fid]

    out_base = args.data_json.rstrip(".json")
    save_distance(data, args.score_json)
    dist = sum(editdistance.eval(d['label'], d['prediction'])
                for d in data)
    total = sum(len(d['label']) for d in data)

    print("PER: {:.3f}".format(dist / total))
    



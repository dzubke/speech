# standard libraries
import argparse
import json
import editdistance

def load_phone_map():
    with open("phones.60-48-39.map", 'r') as fid:
        lines = (l.strip().split() for l in fid)
        lines = [l for l in lines if len(l) == 3]
    m60_48 = {l[0] : l[1] for l in lines}
    m48_39 = {l[1] : l[2] for l in lines}
    return m60_48, m48_39

def remap48_39(data):
    _, m48_39 = load_phone_map()
    data = [m48_39[p] for p in data]
    return data

def consolidate_score(score_path: str, test_path: str, cons_path:str):
    """this function takes in the score_path with the distance metrics and the test_path
    with the example filenames and writes to a consolidated json the filenames and 
    distance metrics in the same json
    
    Arguments
    ---------
    score_path: str, the filepath to read the score_json
    test_path: str, the fielpath to read the test_json
    cons_path: str, the file path to write the consolidated_json

    Returns
    --------
    None: writes to json

    """

    with open(score_path, 'r') as score_fid:
        score_json = [json.loads(l) for l in score_fid]
        with open(test_path, 'r') as test_fid:
            test_json = [json.loads(l) for l in test_fid]

            with open(cons_path, 'w') as fid:
                for i in range(len(score_json)):
                    dist = score_json[i]['dist']
                    length = score_json[i]['label_length']
                    per = score_json[i]['PER']
                    label = score_json[i]['label']
                    predi = score_json[i]['predi']
                    #sorted(test_json, key = lambda x: x['duration'])

                    for j in range(len(test_json)):     # find the filename for the matching label
                        if score_json[i]['label'] == remap48_39(test_json[j]['text']):
                            filename = test_json[j]['audio']
                            cons_entry = {'audio': filename,
                                            'dist': dist, 
                                            'length': length,
                                            'PER': per,
                                            'label': label,
                                            'predi': predi}
                            print(cons_entry)
                            json.dump(cons_entry, fid)
                            fid.write("\n")
                            
                            


if __name__ == "__main__":
    ## format of command is >>python score.py <path_to_score_json> <path_to_test_json> <path_to_cons_json>  
    parser = argparse.ArgumentParser(
            description="Consolidate the score jsons.")

    parser.add_argument("score_json",
        help="Path where the score json is saved.")

    parser.add_argument("test_json",
        help="Path where the test json is saved.")

    parser.add_argument("cons_json",
        help="Name of the consolidated json to save.")
    
    args = parser.parse_args()

    consolidate_score(args.score_json, args.test_json, args.cons_json)
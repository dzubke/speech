# standard libraries
import argparse
import json
import editdistance



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
    use_timit = False #flag used to alter code if the timit dataset is being used because a remapping for a 48-phoneme set occurs to a 39-phoneme set


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

                    match = False # if the score example and test json examples are matches, create a dictionary entry
                    for j in range(len(test_json)):     # find the filename for the matching label
                        
                        if use_timit:       # if timit-flag at top of function is true
                            #if editdistance.eval(score_json[i]['label'], remap48_39(test_json[j]['text'])) < 15 :
                            if score_json[i]['label'] == remap48_39(test_json[j]['text']):
                                match = True

                        else: 
                            if score_json[i]['label'] == test_json[j]['text']:
                                match = True
                        
                        if match:
                            for j in range(len(test_json)):     # find the filename for the matching label
                                if editdistance.eval(score_json[i]['label'], remap48_39(test_json[j]['text'])) < 15 :
                                # if score_json[i]['label'] == remap48_39(test_json[j]['text']):
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

                        match = False

                            


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
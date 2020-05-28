from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import editdistance

def compute_cer(results, verbose=True, dist_len=False):
    """
    Arguments:
        results (list): list of 2 elements tuples made of lists of the ground truth labels
         and phoneme predicted sequences
    Keyword Arguments:
        verbose - bool: if true, the distance and length will be printedd
        dist_len - bool: if true, the distance and length will be returned
    Returns the PER for the full set.

    >>>results = [(["dh", "ah","space", "r"], ["dh", "ah", "r"])] 
    ### dist = 1
    ### total = 4
    >>>compute_cer(results) = 0.25      #dist/total = 1/4
    """
    if len(results[0]) == 3:
        results = [(label, pred) for label, pred, conf in results]
    dist = sum(editdistance.eval(label, pred)
                for label, pred in results)
    total = sum(len(label) for label, _ in results)
    if verbose: print(f"dist:{dist}, total: {total}")
    if dist_len:
        output = (dist/total, (dist, total))
    else: 
        output = dist/total
    return output

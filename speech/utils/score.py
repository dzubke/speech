from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import editdistance

def compute_cer(results):
    """
    Arguments:
        results (list): list of 2 elements tuples made of lists of the ground truth labels
         and phoneme predicted sequences

    Returns the PER for the full set.

    >>>results = [(["dh", "ah","space", "r"], ["dh", "ah", "r"])] 
    ### dist = 1
    ### total = 4
    >>>compute_cer(results) = 0.25      #dist/total = 1/4
    """
    dist = sum(editdistance.eval(label, pred)
                for label, pred in results)
    total = sum(len(label) for label, _ in results)
    print(f"dist:{dist}, total: {total}")
    return dist / total

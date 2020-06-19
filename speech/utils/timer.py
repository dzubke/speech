import time
from typing import Callable

def time_count(function_call:Callable, cum_time: float, cum_count:int):
    """
    a wrapper that records the time for the funciton call in the accumlating
    variable cum_time and a accumulating count in cum_count
    """
    start = time.time()
    function_call
    cum_time += time.time() - start
    cum_count += 1
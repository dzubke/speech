# These functions help to debug the model by saving, plotting, and 
# printing the features, parameters, and gradients of the model
# Author: Dustin Zubke
# Date: 2020-06-12

# standard libraries
from datetime import datetime, date
import logging
from logging import Logger
import math
import os
import pickle
from typing import Generator, Iterable, Tuple, List
# third-party libraries
from graphviz import Digraph
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from torch.autograd import Variable, Function
# project libraries
from speech.utils.data_structs import TorchNamedParams, TorchParams, Batch


def get_logger(logger_path:str='')->Logger:
        # create logger
        logger = logging.getLogger("log")
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        if logger_path:
            fh = logging.FileHandler(logger_path)
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        
        return logger



def check_nan_params_grads(model_params:TorchParams)->bool:
    """
    checks an iterator of model parameters and gradients if any of them have nan values
    Arguments:
        model_params - Iterable[torch.nn.parameter.Parameter]: output of model.parameters()
    """
    for param in model_params:
        # checks all parameters for NaN (param!=param is NaN check)
        if (param!=param).any():
            return True
        # checks all gradients from NaN's
        if param.requires_grad:
            if (param.grad != param.grad).any():
                return True
    return False



def log_model_grads(named_params:TorchNamedParams, logger:Logger)->None:
    """
    records the gradient values of the parameters in the model
    Arguments:
        named_params - Generator[str, torch.nn.parameter.Parameter]: output of model.named_parameters()
    """
    for name, params in named_params:
        if params.requires_grad:
            logger.error(f"log_model_grads: {name}: {params.grad}")


def save_batch_log_stats(batch:Batch, logger:Logger)->None:
    """
    saves the batch to disk and logs a variety of information from a batch. 
    Arguments:
        batch - tuple(tuple(np.2darray), tuple(list(str))): a tuple of inputs and phoneme labels
    """
    filename = get_logger_filename(logger) + "_batch.pickle"
    batch_save_path = os.path.join("./saved_batch", filename)
    with open(batch_save_path, 'wb') as fid: # save current batch to disk for debugging purposes
        pickle.dump(batch, fid)

    if logger is not None:
        # temp_batch is (inputs, labels) so temp_batch[0] is the inputs
        batch_feature_stds = list(map(np.std, batch[0]))
        batch_feature_means = list(map(np.mean, batch[0]))
        batch_feature_maxes = list(map(np.max, batch[0]))
        batch_feature_mins = list(map(np.min, batch[0]))
        input_feature_lengths = list(map(lambda x: x.shape[0], batch[0]))
        label_lengths = list(map(len, batch[1]))
        stacked_batch = np.vstack(batch[0])
        batch_mean = np.mean(stacked_batch)
        batch_std = np.std(stacked_batch)

        logger.info(f"batch_stats: batch_length: {len(batch[0])}, inputs_length: {input_feature_lengths}, labels_length: {label_lengths}")
        logger.info(f"batch_stats: batch_feature_mean: {batch_feature_means}")
        logger.info(f"batch_stats: batch_feature_std: {batch_feature_stds}")
        logger.info(f"batch_stats: batch_feature_max: {batch_feature_maxes}")
        logger.info(f"batch_stats: batch_feature_min: {batch_feature_mins}")
        logger.info(f"batch_stats: batch_mean: {batch_mean}")
        logger.info(f"batch_stats: batch_std: {batch_std}")
        
        # error checks for std values nearly zero and for nan values
        if any([math.isclose(std, 0, abs_tol=1e-6) for std in batch_feature_stds]):
            logger.error(f"batch_stats: batch_std is nearly zero in {batch_feature_stds}")
            print(f"batch_stats: batch_std is nearly zero in {batch_feature_stds}")
        if any(np.isnan(batch_feature_means)):
            logger.error(f"batch_stats: batch_mean is NaN in {batch_feature_means}")
            print(f"batch_stats: batch_mean is NaN in {batch_feature_means}")
        if any(np.isnan(batch_feature_stds)):
            logger.error(f"batch_stats: batch_std is NaN in {batch_feature_stds}")
            print(f"batch_stats: batch_std is NaN in {batch_feature_stds}")


def log_batchnorm_mean_std(state_dict:dict, logger:Logger)->None:
    """
    logs the running mean and variance of the batch_norm layers.
    Both the running mean and variance have the word "running" in the name which is
    how they are selected amongst the other layers in the state_dict.
    Arguments:
        state_dict - dict: the model's state_dict
    """

    for name, values in state_dict.items():
        if "running" in name:
            logger.info(f"batch_norm_mean_var: {name}: {values}")


def log_param_grad_norms(named_parameters:TorchNamedParams, logger:Logger)->None:
    """
    Calculates and logs the norm of the gradients of the parameters
    and the norm of all the gradients together.
    Note: norm_type is hardcoded to 2.0
    Arguments:
        named_params - Generator[str, torch.nn.parameter.Parameter]: output of model.named_parameters()
    """
    norm_type = 2.0
    total_norm = 0.0
    for name, param in named_parameters:
        if param.grad is not None:
            param_norm = param.grad.detach().norm(norm_type)
            logger.info(f"param_grad_norm: {name}: {param_norm}")
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    logger.info(f"param_grad_norm: total_norm: {total_norm}")        
        

def get_logger_filename(logger:Logger)->str:
    """
    Returns the filename of the logger
    """
    basename, filename = os.path.split(logger.handlers[0].baseFilename)
    filename, ext = os.path.splitext(filename)
    return filename


def format_bytes(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    Code from: https://www.thepythoncode.com/article/get-hardware-system-information-python
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def log_cpu_mem_disk_usage(logger:Logger)->None:
    """
    Logs the certain metrics on the current cpu, memory, disk, and CPU usage

    Code adapted from: https://www.thepythoncode.com/article/get-hardware-system-information-python
    """
    
    logger.info(f"{'='*10} VM stats begin {'='*10}")
    # CPU metrics
    logger.info(f"log_vm_stats: Total CPU Usage: {psutil.cpu_percent()}%")
    # memory metrics
    svmem = psutil.virtual_memory() 
    logger.info(f"log_vm_stats: Total Memory: {format_bytes(svmem.total)}")
    logger.info(f"log_vm_stats: Memory Available: {format_bytes(svmem.available)}")
    logger.info(f"log_vm_stats: Memory Used: {format_bytes(svmem.used)}")
    logger.info(f"log_vm_stats: Memory Used Percentage: {svmem.percent}%")
    # disk metrics
    partitions = psutil.disk_partitions()
    for partition in partitions:
        logger.info(f"--- Disk Device: {partition.device} ---")
        try:
            partition_usage = psutil.disk_usage(partition.mountpoint)
        except PermissionError as perm_error:
            # this can be catched due to the disk that
            # isn't ready
            logger.info(f"log_vm_stats: PermissionError for disk {perm_error}")
            continue
        logger.info(f"log_vm_stats: Total Size: {format_bytes(partition_usage.total)}")
        logger.info(f"log_vm_stats: Used: {format_bytes(partition_usage.used)}")
        logger.info(f"log_vm_stats: Free: {format_bytes(partition_usage.free)}")
        logger.info(f"log_vm_stats: Percentage: {partition_usage.percent}%")
    logger.info(f"{'='*10} VM stats end {'='*10}")



# plot_grad_flow comes from this post:
# https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7

def plot_grad_flow_line(named_parameters:TorchNamedParams)->None:
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    save_path = "./plots/grad_flow_line_{}.png".format(datetime.now().strftime("%Y-%m-%d_%Hhr"))
    plt.savefig(save_path, bbox_inches="tight")
    # clears and closes the figure so memory doesn't overfill
    plt.close('all')


def plot_grad_flow_bar(named_parameters:TorchNamedParams, filename:str="grad_flow_bar.png"):
    '''
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    plot_grad_flow(self.model.named_parameters()) to visualize the gradient flow
    Note: currently this function creates and closes a new figure for each batch.
        If cumulative information across batches is desired without making new figures
        one could create a figure object outside of this function and pass that figure
        to this function for cumulative plots. 
    '''
    ave_grads = []
    max_grads= []
    layers = []
    for name, param in named_parameters:
        if(param.requires_grad) and ("bias" not in name):
            layers.append(name)
            ave_grads.append(param.grad.abs().mean())
            max_grads.append(param.grad.abs().max())
    fig, ax1 = plt.subplots()
    ax1_color = 'c'
    ax2_color = 'b'
    ax1.bar(np.arange(len(max_grads)), max_grads, alpha=0.4, lw=1, color=ax1_color)
    ax1.tick_params(axis='y', labelcolor=ax1_color)
    ax1.tick_params(axis='x', labelrotation=90)
    ax2 = ax1.twinx()
    ax2.bar(np.arange(len(max_grads)), ave_grads, alpha=0.4, lw=1, color=ax2_color)
    ax2.tick_params(axis='y', labelcolor=ax2_color)
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    #plt.ylim(bottom = -0.001, top=1.0) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    ax1.set_ylabel("max gradients", color=ax1_color)
    ax2.set_ylabel("average gradients", color=ax2_color)
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color=ax1_color, lw=4),
                Line2D([0], [0], color=ax2_color, lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    # formatted_date-hour = datetime.now().strftime("%Y-%m-%d_%Hhr")
    filename = filename + "_bar.png"
    save_path = os.path.join("./plots", filename)
    plt.savefig(save_path, bbox_inches="tight")
    # clears and closes the figure so memory doesn't overfill
    fig.clear()
    plt.close(fig)



# bad_grad_viz functions come from here:
# https://gist.github.com/apaszke/f93a377244be9bfcb96d3547b9bc424d
def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)

def register_hooks(var):
    fn_dict = {}
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        grad_output = grad_output.data
        return grad_output.ne(grad_output).any() or grad_output.gt(1e6).any()

    def make_dot():
        node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '('+(', ').join(map(str, size))+')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                assert fn in fn_dict, fn
                fillcolor = 'white'
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'red'
                dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot

if __name__ == '__main__':
    x = Variable(torch.randn(10, 10), requires_grad=True)
    y = Variable(torch.randn(10, 10), requires_grad=True)

    z = x / (y * 0)
    z = z.sum() * 2
    get_dot = register_hooks(z)
    z.backward()
    dot = get_dot()
    dot.save('tmp.dot')

# These functions help to debug the model by saving, plotting, and 
# printing the features, parameters, and gradients of the model
# Author: Dustin Zubke
# Date: 2020-06-12

# standard libraries
from datetime import datetime, date
import pickle
from typing import Generator
# third-party libraries
from graphviz import Digraph
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable, Function
# project libraries


def check_nan(model_params:Generator[torch.nn.parameter.Parameter]):
    """
    checks an iterator of model parameters and gradients if any of them have nan values
    Arguments:
        model_params - Generator[torch.nn.parameter.Parameter]: output of model.parameters()
    """
    for param in model_params:
        if (param!=param).any():
            return True
        if param.requires_grad:
            if (param.grad != param.grad).any():
                return True
    return False



def log_model_grads(named_params:Generator, logger):
    """
    records the gradient values of the parameters in the model
    Arguments:
        named_params - Generator[str, torch.nn.parameter.Parameter]: output of model.named_parameters()
    """
    for name, params in named_params:
        if params.requires_grad:
            logger.error(f"log_model_grads: {name}: {params.grad}")


def save_batch_log_stats(batch:tuple, logger):
    """
    saves the batch to disk and logs a variety of information from a batch. 
    Arguments:
        batch - tuple(list(np.2darray), list(list)): a tuple of inputs and phoneme labels
    """
    today = str(date.today())
    batch_save_path = "./current-batch_{}.pickle".format(today)
    with open(batch_save_path, 'wb') as fid: # save current batch to disk for debugging purposes
        pickle.dump(batch, fid)

    if logger is not None:
        # temp_batch is (inputs, labels) so temp_batch[0] is the inputs
        batch_sample_stds = list(map(np.std, batch[0]))
        batch_sample_means = list(map(np.mean, batch[0]))
        batch_sample_maxes = list(map(np.max, batch[0]))
        batch_sample_mins = list(map(np.min, batch[0]))
        input_sample_lengths = list(map(lambda x: x.shape[0], batch[0]))
        label_sample_lengths = list(map(len, batch[1]))
        stacked_batch = np.vstack(batch[0])
        batch_mean = np.mean(stacked_batch)
        batch_std = np.std(stacked_batch)

        logger.info(f"batch_stats: batch_length: {len(batch[0])}, inputs_length: {input_sample_lengths}, labels_length: {label_sample_lengths}")
        logger.info(f"batch_stats: batch_sample_mean: {batch_sample_means}")
        logger.info(f"batch_stats: batch_sample_std: {batch_sample_stds}")
        logger.info(f"batch_stats: batch_sample_max: {batch_sample_maxes}")
        logger.info(f"batch_stats: batch_sample_min: {batch_sample_mins}")
        logger.info(f"batch_stats: batch_mean: {batch_mean}")
        logger.info(f"batch_stats: batch_std: {batch_std}")


def log_batchnorm_mean_std(state_dict:dict, logger):
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


def log_layer_grad_norms(named_parameters:Generator, logger):
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
            param_norm = param.grad.data.norm(norm_type)
            logger.info(f"layer_grad_norm: {name}: {param_norm}")
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    logger.info(f"layer_grad_norm: total_norm: {total_norm}")
            
        


# plot_grad_flow comes from this post:
# https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7

def plot_grad_flow_line(named_parameters):
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

def plot_grad_flow_bar(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
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
    save_path = "./plots/grad_flow_bar_{}.png".format(datetime.now().strftime("%Y-%m-%d_%Hhr"))
    plt.savefig(save_path, bbox_inches="tight")


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

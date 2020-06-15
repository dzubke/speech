# These functions help to debug the model by saving, plotting, and 
# printing the features, parameters, and gradients of the model
# Author: Dustin Zubke
# Date: 2020-06-12

# standard libraries
from datetime import datetime, date
import pickle
# third-party libraries
from graphviz import Digraph
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable, Function
# project libraries


def check_nan(model:torch.nn.Module):
    """
    checks an iterator of model parameters and gradients if any of them have nan values
    """
    for param in model.parameters():
        if (param!=param).any():
            return True
        if param.requires_grad:
            if (param.grad != param.grad).any():
                return True
    return False



def log_conv_grads(model:torch.nn.Module, logger):
    """
    records the gradient values for the weight values in model into
    the logger
    """
    # layers with weights
    weight_layer_types = [torch.nn.modules.conv.Conv2d, torch.nn.modules.batchnorm.BatchNorm2d]
    # only iterating through conv layers in first elemment of model children
    for layer in [*model.children()][0]:
        if type(layer) in weight_layer_types:
            logger.error(f"grad: {layer}: {layer.weight.grad}")


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
        batch_std = list(map(np.std, batch[0]))
        batch_mean = list(map(np.mean, batch[0]))
        batch_max = list(map(np.max, batch[0]))
        batch_min = list(map(np.min, batch[0]))
        inputs_length = list(map(lambda x: x.shape[0], batch[0]))
        labels_length = list(map(len, batch[1]))
        logger.info(f"batch_stats: batch_length: {len(batch[0])}, inputs_length: {inputs_length}, labels_length: {labels_length}")
        logger.info(f"batch_stats: batch_mean: {batch_mean}")
        logger.info(f"batch_stats: batch_std: {batch_std}")
        logger.info(f"batch_stats: batch_max: {batch_max}")
        logger.info(f"batch_stats: batch_min: {batch_min}")


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
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    fig, ax1 = plt.subplots()
    ax1.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    ax2 = ax1.twinx()
    ax2.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    ax2.tick_params(axis='y', labelcolor=color)
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=1.0) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    save_path = "./plots/grad_flow_bar_{}.png".format(datetime.now().strftime("%Y-%m-%d_%Hhr"))
    plt.savefig(save_path, bbox_inches="tight")

def plot_grad_flow_bar(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=1.0) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
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

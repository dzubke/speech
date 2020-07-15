#!/bin/bash

# usage: bash updated_tensorboard.sh

# description: will updated the tensorboard in the awni_env36 environment on July 10 2020

pip uninstall -y tensorflow-tensorboard
pip uninstall -y tensorboard-logger
pip install tensorflow==1.15.3
pip install -U tensorboardX

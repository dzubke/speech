#!/bin/bash
# Run `source setup.sh` from this directory.
export PYTHONPATH=`pwd`:`pwd`/libs/warp-ctc/pytorch_binding:`pwd`/libs:$PYTHONPATH

export PYTHONPATH=`pwd`:`pwd`/libs/warp-ctc/:`pwd`/libs:$PYTHONPATH

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/libs/warp-ctc/build

#export PYTHONPATH=$PYTHONPATH:/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech

conda activate awni_env36

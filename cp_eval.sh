#!/bin/bash
# commmand structure bash eval.sh <model_start_date> <model_checkpoint_date> <--last (opt)> 
# --last flag passed to evel.sh 

mkdir ./examples/librispeech/models/ctc_models/$1/$2

cp ./examples/librispeech/models/ctc_models/$1/* ./examples/librispeech/models/ctc_models/$1/$2

bash eval.sh $1 $2 $3

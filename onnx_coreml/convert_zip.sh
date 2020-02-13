#!/bin/bash



python torch_to_onnx.py $1 --num_frames $2 --use_state_dict True 
python onnx_to_coreml.py $1
python validation.py $1 --num_frames $2
bash transfer_zip.sh $1

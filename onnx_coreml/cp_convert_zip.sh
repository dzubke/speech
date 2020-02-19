#!/bin/bash
# this command should be called from inside the onnx_coreml/ directory
# command structure: bash cp_convert_zip.sh <base_path_to_model> <model_name> <num_frames>
cp $1/best_model ./torch_models/$2_model.pth
cp $1/best_preproc.pyc ./preproc/$2_preproc.pyc
cp $1/ctc_config.json ./config/$2_config.json

python torch_to_onnx.py $2 --num_frames $3 --use_state_dict True 
python onnx_to_coreml.py $2
python validation.py $2 --num_frames $3
bash transfer_zip.sh $2

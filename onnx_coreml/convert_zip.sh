#!/bin/bash
# this command should be called from inside the onnx_coreml/ directory
# command structure: bash cp_convert_zip.sh <model_name> <num_frames>

sed -i '' 's/"convert_model": false,/"convert_model": true,/g' ./config/$1_config.json
sed -i '' 's/import functions\.ctc/#import functions\.ctc/g' ~/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/speech/models/ctc_model.py


python torch_to_onnx.py $1 --num_frames $2 --use_state_dict True 
python onnx_to_coreml.py $1
python validation.py $1 --num_frames $2
bash transfer_zip.sh $1

sed -i '' 's/#import functions\.ctc/import functions\.ctc/g' ~/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/speech/models/ctc_model.py

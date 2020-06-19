#!/bin/bash
# this command should be called from inside the onnx_coreml/ directory
# command structure: bash cp_convert_zip.sh <base_path_to_model> <model_name> <num_frames>
cp $1/best_model ./torch_models/$2_model.pth
cp $1/best_preproc.pyc ./preproc/$2_preproc.pyc
cp $1/ctc_config.json ./config/$2_config.json

#sed -i '' 's/"convert_model": false,/"convert_model": true,/g' ./config/$2_config.json
#sed -i '' 's/import functions\.ctc/#import functions\.ctc/g' ~/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/speech/models/ctc_model.py

python torch_to_onnx.py $2 --num_frames $3 --use_state_dict True 
python onnx_to_coreml.py $2
python validation.py $2 --num_frames $3
bash transfer_zip.sh $2

#sed -i '' 's/#import functions\.ctc/import functions\.ctc/g' ~/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/speech/models/ctc_model.py

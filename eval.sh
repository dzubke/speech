#!/bin/bash
# commmand structure bash eval.sh <model_start_date> <model_checkpoint_date>

mkdir ./examples/librispeech/models/ctc_models/$1/$2
cp ./examples/librispeech/models/ctc_models/$1/* ./examples/librispeech/models/ctc_models/$1/$2
python eval.py ./examples/librispeech/models/ctc_models/$1/$2 ~/awni_speech/data/speak_test_data/speak_test.json --save ./predictions/$1-$2_speak_test_predictions.json
python eval.py ./examples/librispeech/models/ctc_models/$1/$2 ~/awni_speech/data/dustin_test_data/20191202_clean/drz_test.json --save ./predictions/$1-$2_1202_predictions.json
python eval.py ./examples/librispeech/models/ctc_models/$1/$2 ~/awni_speech/data/dustin_test_data/20191118_plane/simple/drz_test.json --save ./predictions/$1-$2_1118-simple_predictions.json
python eval.py ./examples/librispeech/models/ctc_models/$1/$2 ~/awni_speech/data/LibriSpeech/test-combo.json --save ./predictions/$1-$2_libsp-test-combo_predictions.json

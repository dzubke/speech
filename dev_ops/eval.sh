#!/bin/bash
# commmand structure bash eval.sh <model_start_date> <model_checkpoint_date> <--last (optional)>

echo -e "\nEvaluating the New Speak Test Set"
python eval.py $3 ./examples/librispeech/models/ctc_models/$1/$2 /mnt/disks/data_disk/home/dzubke/awni_speech/data/speak_test_data/2020-05-27/speak-test_2020-05-27.json --save ./predictions/$1-$2_speak-test
echo -e "\nEvaluating the Old Speak Test Set"
python eval.py $3 ./examples/librispeech/models/ctc_models/$1/$2 /mnt/disks/data_disk/home/dzubke/awni_speech/data/speak_test_data/2019-11-29/speak-test_2019-11-29.json --save ./predictions/$1-$2_old-speak-test
echo -e "\nEvaluating Dustin Clean Testset"
python eval.py $3 ./examples/librispeech/models/ctc_models/$1/$2 /mnt/disks/data_disk/home/dzubke/awni_speech/data/dustin_test_data/20191202_clean/drz_test.json --save ./predictions/$1-$2_dustin-1202
echo -e "\nEvaluating the Dustin Noisy Testset"
python eval.py $3 ./examples/librispeech/models/ctc_models/$1/$2 /mnt/disks/data_disk/home/dzubke/awni_speech/data/dustin_test_data/20191118_plane/simple/drz_test.json --save ./predictions/$1-$2_dustin-1118-simple
echo -e "\nEvaluating Librispeech Combo Devset"
python eval.py $3 ./examples/librispeech/models/ctc_models/$1/$2 /mnt/disks/data_disk/home/dzubke/awni_speech/data/LibriSpeech/dev-combo.json  --save ./predictions/$1-$2_libsp-dev-combo
echo -e "\nEvaluating Tedlium Dev set"
python eval.py $3 ./examples/librispeech/models/ctc_models/$1/$2 /mnt/disks/data_disk/home/dzubke/awni_speech/data/tedlium/TEDLIUM_release-3/dev.json  --save ./predictions/$1-$2_ted-dev
echo -e "\nEvaluating Common Voice Dev set"
python eval.py $3 ./examples/librispeech/models/ctc_models/$1/$2 /mnt/disks/data_disk/home/dzubke/awni_speech/data/common-voice/dev.json  --save ./predictions/$1-$2_cv-dev

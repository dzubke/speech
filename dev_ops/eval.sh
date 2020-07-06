#!/bin/bash
# commmand structure bash eval.sh <model_start_date> <model_checkpoint_date> <--last (optional)>

# if the model_start_date contains a '/' this will screw up the saving of predictions in the eval.py call
# so the if statement below creates a new string that will replace the '/' with a '-'
# The string replacement command: "${1////-}" has the format: "${original_string/string_to_replace/string_replaced_with}"
# inside "${1////-}":  the 1st and 4th '/' separate the strings and the 2nd '/' is an escape character for the 3rd '/' which is the string_to_replace
# confusing, right? 
SUB=/
if [[ "$1" == *"$SUB"* ]]; then 
    SAVE_DATE="${1////-}"
    echo SAVE_DATE updated from $1 to $SAVE_DATE
else
     SAVE_DATE="$1"
    echo SAVE_DATE unchanged from $1 to $SAVE_DATE
fi

echo -e "\nEvaluating the New Speak Test Set"
python eval.py $3 ./examples/librispeech/models/ctc_models/$1/$2 /mnt/disks/data_disk/home/dzubke/awni_speech/data/speak_test_data/2020-05-27/speak-test_2020-05-27.json --save ./predictions/$SAVE_DATE-$2_speak-test

echo -e "\nEvaluating the Old Speak Test Set"
python eval.py $3 ./examples/librispeech/models/ctc_models/$1/$2 /mnt/disks/data_disk/home/dzubke/awni_speech/data/speak_test_data/2019-11-29/speak-test_2019-11-29.json --save ./predictions/$SAVE_DATE-$2_old-speak-test

echo -e "\nEvaluating Dustin Clean Testset"
python eval.py $3 ./examples/librispeech/models/ctc_models/$1/$2 /mnt/disks/data_disk/home/dzubke/awni_speech/data/dustin_test_data/20191202_clean/drz_test.json --save ./predictions/$SAVE_DATE-$2_dustin-1202

echo -e "\nEvaluating the Dustin Noisy Testset"
python eval.py $3 ./examples/librispeech/models/ctc_models/$1/$2 /mnt/disks/data_disk/home/dzubke/awni_speech/data/dustin_test_data/20191118_plane/simple/drz_test.json --save ./predictions/$SAVE_DATE-$2_dustin-1118-simple

echo -e "\nEvaluating Common Voice Dev set"
python eval.py $3 ./examples/librispeech/models/ctc_models/$1/$2 /mnt/disks/data_disk/home/dzubke/awni_speech/data/common-voice/dev.json  --save ./predictions/$SAVE_DATE-$2_cv-dev

echo -e "\nEvaluating Librispeech Clean Devset"
python eval.py $3 ./examples/librispeech/models/ctc_models/$1/$2 /mnt/disks/data_disk/home/dzubke/awni_speech/data/LibriSpeech/dev-clean.json  --save ./predictions/$SAVE_DATE-$2_libsp-dev-clean

echo -e "\nEvaluating Tedlium Dev set"
python eval.py $3 ./examples/librispeech/models/ctc_models/$1/$2 /mnt/disks/data_disk/home/dzubke/awni_speech/data/tedlium/TEDLIUM_release-3/dev.json  --save ./predictions/$SAVE_DATE-$2_ted-dev

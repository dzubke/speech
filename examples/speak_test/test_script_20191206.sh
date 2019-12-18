#!/bin/bash
cd ~/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/examples/speak_test
python test_preprocess.py /Users/dustin/CS/consulting/firstlayerai/data/dustin_test_data/20191203/batch1 20191203_b1_test --list_transcripts False 
python test_preprocess.py /Users/dustin/CS/consulting/firstlayerai/data/dustin_test_data/20191203/batch2 20191203_b2_test --list_transcripts False 
python test_preprocess.py /Users/dustin/CS/consulting/firstlayerai/data/dustin_test_data/20191203/batch3 20191203_b3_test --list_transcripts False 
python test_preprocess.py /Users/dustin/CS/consulting/firstlayerai/data/dustin_test_data/20191203/batch4 20191203_b4_test --list_transcripts False 
cd ~/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech
python eval.py ./examples/timit/models/GCP_ctc_best/20191121/   /Users/dustin/CS/consulting/firstlayerai/data/dustin_test_data/20191203/batch1/20191203_b1_test.json --save ./predictions/20191203_b1_predictions.json
python eval.py ./examples/timit/models/GCP_ctc_best/20191121/   /Users/dustin/CS/consulting/firstlayerai/data/dustin_test_data/20191203/batch2/20191203_b2_test.json --save ./predictions/20191203_b2_predictions.json
python eval.py ./examples/timit/models/GCP_ctc_best/20191121/   /Users/dustin/CS/consulting/firstlayerai/data/dustin_test_data/20191203/batch3/20191203_b3_test.json --save ./predictions/20191203_b3_predictions.json
python eval.py ./examples/timit/models/GCP_ctc_best/20191121/   /Users/dustin/CS/consulting/firstlayerai/data/dustin_test_data/20191203/batch4/20191203_b4_test.json --save ./predictions/20191203_b4_predictions.json
cd /Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/examples/timit
python score.py /Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/predictions/20191203_b1_predictions.json
python score.py /Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/predictions/20191203_b2_predictions.json
python score.py /Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/predictions/20191203_b3_predictions.json
python score.py /Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/predictions/20191203_b4_predictions.json
cd ~/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/examples/speak_test
python json_match.py /Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/predictions/20191203_b1_predicti_dist_39.json /Users/dustin/CS/consulting/firstlayerai/data/dustin_test_data/20191203/batch1/20191203_b1_test.json 20191203_b1_cons
python json_match.py /Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/predictions/20191203_b2_predicti_dist_39.json /Users/dustin/CS/consulting/firstlayerai/data/dustin_test_data/20191203/batch2/20191203_b2_test.json 20191203_b2_cons
python json_match.py /Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/predictions/20191203_b3_predicti_dist_39.json /Users/dustin/CS/consulting/firstlayerai/data/dustin_test_data/20191203/batch3/20191203_b3_test.json 20191203_b3_cons
python json_match.py /Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/predictions/20191203_b4_predicti_dist_39.json /Users/dustin/CS/consulting/firstlayerai/data/dustin_test_data/20191203/batch4/20191203_b4_test.json 20191203_b4_cons


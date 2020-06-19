#!/bin/bash
# this command should be run within a VM that already has common-voice, voxforge,
# and tatoeba processed data already installed. 
# command structure: bash cv-vox-tat_transfer.sh <vm_name>
gcloud compute scp --recurse ~/awni_speech/data/common-voice/ dzubke@$1:~/awni_speech/data/ --zone us-central1-c
gcloud compute scp --recurse ~/awni_speech/data/voxforge/ dzubke@$1:~/awni_speech/data/ --zone us-central1-c
gcloud compute scp --recurse ~/awni_speech/data/tatoeba/ dzubke@$1:~/awni_speech/data/ --zone us-central1-c
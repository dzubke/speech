#!/bin/bash
# commmand structure: bash vm_create_setup.sh <$1=instance_name>

USERNAME=`gcloud compute os-login describe-profile | awk '/username:/ {print $2}'`

gcloud compute instances create $1 \
    --machine-type n1-standard-4 --zone us-central1-c \
    --create-disk size=150 \
    --accelerator type=nvidia-tesla-p100,count=1 \
    --image-family ubuntu-1604-lts --image-project speak-ml-dev

gcloud compute instances start $1
gcloud compute scp --zone us-central1-c ./vm_setup.sh $USERNAME@$1:~/ 
gcloud compute ssh $USERNAME:$1 'bash ~/vm_setup.sh'

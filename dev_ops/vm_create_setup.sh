#!/bin/bash
# commmand structure: bash vm_create_setup.sh <$1=instance_name>

GCLOUD_USER=`gcloud compute os-login describe-profile | awk '/username:/ {print $2}'`
GCLOUD_USER="$(cut -d '_' -f1 <<<$GCLOUD_USER)"

echo "gcloud username is: $GCLOUD_USER"

gcloud compute instances create $1 \
    --machine-type n1-standard-4 --zone us-central1-c \
    --create-disk size=150 \
   # --accelerator type=nvidia-tesla-p100,count=1 \
    --image-family ubuntu-1604-lts --image-project ubuntu-os-cloud

gcloud compute instances attach-disk $1 \
    --disk data-disk-readonly-2020-06-19 --mode=ro --zone=us-central1-c

gcloud compute instances start $1
gcloud compute scp --zone us-central1-c ./vm_setup.sh $GCLOUD_USER@$1:~/ 
gcloud compute ssh $USERNAME:$1 'bash ~/vm_setup.sh'

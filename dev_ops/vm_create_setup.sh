#!/bin/bash
# commmand structure: bash vm_create_setup.sh <$1=instance_name>

# parsing gcloud username
GCLOUD_USER=`gcloud compute os-login describe-profile | awk '/username:/ {print $2}'`
GCLOUD_USER="$(cut -d '_' -f1 <<<$GCLOUD_USER)"
echo "gcloud username is: $GCLOUD_USER"

# creating an instance
gcloud compute instances create $1 \
    --custom-cpu=10 --custom-memory=15GB \
    --custom-vm-type=n1 --zone us-central1-c \
    --boot-disk-size=150GB \
    --image-family ubuntu-1604-lts --image-project ubuntu-os-cloud \
    --accelerator type=nvidia-tesla-p100,count=1 \
    --maintenance-policy=TERMINATE

# attaching the data disk, hardcoded as `data-disk-readonly-2020-06-19`
gcloud compute instances attach-disk $1 \
    --disk data-disk-readonly-2020-06-19 --mode=ro --zone=us-central1-c

# starting the instance, copying vm_setup.sh file, and running that through ssh
gcloud compute instances start $1
gcloud compute scp --zone us-central1-c ./vm_setup.sh $GCLOUD_USER@$1:~/ 
# # the ssh command may fail if timed out. If so, ssh to the VM using this command:
# # `gcloud compute ssh $USERNAME:$1` where $USERNAME is your username and $1 is the VM name
# # then, run `bash ~/vm_setup.sh`
# gcloud compute ssh $USERNAME:$1 -- 'bash ~/vm_setup.sh'

# # restarting VM to ensure data-disk is mounted using /etc/fstab file
# gcloud compute instances stop $1
# gcloud compute instances start $1
# gcloud compute ssh $USERNAME:$1
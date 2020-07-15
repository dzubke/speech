# 2. update the vm
echo "updating VM" > ~/setup_log.txt
sudo apt-get update
sudo apt-get update && sudo apt-get --only-upgrade install kubectl google-cloud-sdk google-cloud-sdk-app-engine-grpc google-cloud-sdk-pubsub-emulator google-cloud-sdk-app-engine-go google-cloud-sdk-firestore-emulator google-cloud-sdk-cloud-build-local google-cloud-sdk-datastore-emulator google-cloud-sdk-app-engine-python google-cloud-sdk-cbt google-cloud-sdk-bigtable-emulator google-cloud-sdk-app-engine-python-extras google-cloud-sdk-datalab google-cloud-sdk-app-engine-java

# 4. install miniconda - the 3-4.5.4 version was the last to use python 3.6 - 283M size of miniconda3/
echo "installing miniconda" >> ~/setup_log.txt
sudo apt-get -y install bzip2  # need to install bzip2 to install miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh
bash Miniconda3-4.5.4-Linux-x86_64.sh -b -p $HOME/miniconda
rm Miniconda3-4.5.4-Linux-x86_64.sh
echo '
# adding conda path
export PATH=$PATH:~/miniconda/bin/
. /home/dzubke/miniconda/etc/profile.d/conda.sh' >> ~/.bashrc
# TODO: This line below still doens't work resulting in "conda not found" error
. ~/.bashrc

# 5. creae conda environment
echo "creating conda venv" >> ~/setup_log.txt
conda create -y -n awni_env36 python=3.6.5
conda activate awni_env36
pip install --upgrade pip

# 6. install git
echo "installed git" >> ~/setup_log.txt
sudo apt-get -y install git

# 7. create awni_speech dir
mkdir awni_speech
cd awni_speech

# 8. clone repo
echo "cloning repo" >> ~/setup_log.txt
git clone https://github.com/dzubke/speech.git

# 9. install the requirements.txt with conda and other modules not supported by conda with pip
echo "installing python requirements" >> ~/setup_log.txt
while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements.txt
# old way for installations below 
# conda install -y --file requirements.txt
# conda install -y ipython
# pip install editdistance==0.4 protobuf==3.4.0 pytest==3.2.3 scipy==0.18.1 SoundFile==0.10.2 tensorboard-logger==0.0.4 python_speech_features==0.6
# pip install onnx onnxruntime coremltools onnx-coreml graphviz==0.14 librosa==0.7.2 tensorflow-tensorboard==0.1.8 tensorflow==1.3.0 matplotlib==3.1.1

# 10. install CUDA
    # before adding GPU to VM install the driver - Tesla K80 can use CUDA 9.0, 9.2, 10.0, 10.1, and maybe others
    # location of CUDA archive - http://developer.download.nvidia.com/compute/cuda/repos/
    # source for below: https://developer.nvidia.com/cuda-92-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=debnetwork & https://cloud.google.com/compute/docs/gpus/add-gpus#install-gpu-driver

#Cuda 10
echo "installing CUDA" >> ~/setup_log.txt
# cuda 10.0
# curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
# sudo dpkg -i cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
# sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
# sudo apt-get -y update
# sudo apt-get -y install cuda
# rm ~/awni_speech/speech/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb

# cuda 10.2
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-ubuntu1604.pin
# sudo mv cuda-ubuntu1604.pin /etc/apt/preferences.d/cuda-repository-pin-600
# sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
# sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/ /"
# sudo apt-get update
# sudo apt-get install -y cuda=10.2.89-1

# cuda 10.0
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install -y cuda=10.0.130-1



# 15. install linux packages
sudo apt-get -y install ffmpeg sox soxi vim
echo 'set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab' >> ~/.vimrc


# 13. install pytorch
echo "installing pytorch" >> ~/setup_log.txt
conda install -y pytorch=0.4.1 cuda100 -c pytorch
# pytorch 1.0 version for linux
#conda install -y pytorch==1.0.0 torchvision==0.2.1 cuda100 -c pytorch
# pytorch 1.3 version for linux
#conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch 
# pytoch 1.3 version for mac
#conda install -y pytorch torchvision -c pytorch

# 14. run the makefile
echo "making project libraries" >> ~/setup_log.txt
cd ~/awni_speech/speech/
sudo apt-get -y install make
sudo apt-get -y install cmake
make

# add the data disk to the /etc/fstab file
sudo mkdir -p /mnt/disks/data_disk`
echo UUID=`sudo blkid -s UUID -o value /dev/sdb1` /mnt/disks/data_disk ext4 ro,discard,suid,dev,exec,auto,nouser,async,nofail,noload 0 2 | sudo tee -a /etc/fstab

mkdir logs

echo "configuring ~/.bashrc" >> ~/setup_log.txt
echo '# setup
conda activate awni_env36
cd ~/awni_speech/speech
source setup.sh

#aliases
alias ..="cd .."

alias ..2="cd ../.."

alias ..3="cd ../../.."

alias rl="readlink -f"' >> ~/.bashrc

echo "setup complete" >> ~/setup_log.txt

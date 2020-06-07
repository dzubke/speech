# 2. update the vm
sudo apt-get update
sudo apt-get update && sudo apt-get --only-upgrade install kubectl google-cloud-sdk google-cloud-sdk-app-engine-grpc google-cloud-sdk-pubsub-emulator google-cloud-sdk-app-engine-go google-cloud-sdk-firestore-emulator google-cloud-sdk-cloud-build-local google-cloud-sdk-datastore-emulator google-cloud-sdk-app-engine-python google-cloud-sdk-cbt google-cloud-sdk-bigtable-emulator google-cloud-sdk-app-engine-python-extras google-cloud-sdk-datalab google-cloud-sdk-app-engine-java

# 4. install miniconda - the 3-4.5.4 version was the last to use python 3.6 - 283M size of miniconda3/
sudo apt-get install bzip2  # need to install bzip2 to install miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh
sh Miniconda3-4.5.4-Linux-x86_64.sh 
rm Miniconda3-4.5.4-Linux-x86_64.sh
source ~/.bashrc

# 5. create conda venv
conda create -n awni_env36 python=3.6.5
source activate awni_env36


# 9. install the requirements.txt with conda and other modules not supported by conda with pip
cd speech/
conda install --file requirements.txt
conda install ipython
pip install editdistance==0.4 protobuf==3.4.0 pytest==3.2.3 scipy==0.18.1 SoundFile==0.10.2 tensorboard-logger==0.0.4 python_speech_features==0.6
pip install onnx onnxruntime coremltools onnx-coreml

# 10. install CUDA
    # before adding GPU to VM install the driver - Tesla K80 can use CUDA 9.0, 9.2, 10.0, 10.1, and maybe others
    # location of CUDA archive - http://developer.download.nvidia.com/compute/cuda/repos/
    # source for below: https://developer.nvidia.com/cuda-92-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=debnetwork & https://cloud.google.com/compute/docs/gpus/add-gpus#install-gpu-driver

Cuda 10
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda

# 11. add the GPU to the VM

# 15. install ffmpeg and vim
sudo apt-get install ffmpeg
sudo apt-get install vim



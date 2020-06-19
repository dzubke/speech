

# 13. install pytorch
conda install pytorch=0.4.1 cuda92 -c pytorch
conda install pytorch==1.0.0 torchvision==0.2.1 cuda100 -c pytorch
#latest, 1.3
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch 
#for mac, pytoch 1.3
conda install pytorch torchvision -c pytorch

# 14. run the makefile
cd ~/awni_speech/speech/
sudo apt-get install make
sudo apt-get install cmake
make

cmake ../ && make; cd ../pytorch_binding; python build.py

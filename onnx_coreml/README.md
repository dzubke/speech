0) following this tutorial:
https://pytorch.org/tutorials/advanced/ONNXLive.html

1) Had an error importing PILLOW_VERSION. run the commands below to resolve: 
pip uninstall Pillow
conda install pillow=6.1

2) when running onnx_to_coreml.py, there was an error:TypeError: __init__() got an unexpected keyword argument 'file'
The resolution here: https://github.com/onnx/onnx/issues/363, was to install protoc (brew install protobuf) and make sure the version of protoc (protoc --version) was the same as the version in python-protobuff (pip install protobuf=3.11.2)
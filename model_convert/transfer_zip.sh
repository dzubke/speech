#!/bin/bash

echo $1.zip

zip -j ./zips/$1.zip ./coreml_models/$1_model.mlmodel ./preproc/$1_preproc.json ./config/$1_config.json ./output/$1_output.json

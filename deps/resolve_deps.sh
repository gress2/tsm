#!/bin/bash

git submodule update --init --recursive

if [ "$(uname)" == "Darwin" ]; then
  ZIP=libtorch-macos-latest.zip
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
  ZIP=libtorch-shared-with-deps-latest.zip
fi

wget https://download.pytorch.org/libtorch/cpu/$ZIP
unzip $ZIP && rm $ZIP 



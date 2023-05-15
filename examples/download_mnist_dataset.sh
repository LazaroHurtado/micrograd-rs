#! /bin/bash

HOST=http://yann.lecun.com/exdb/mnist/
FOLDER=./examples/mnist_dataset/
declare -a FILENAMES=(
  "t10k-images-idx3-ubyte.gz"
  "t10k-labels-idx1-ubyte.gz"
  "train-images-idx3-ubyte.gz"
  "train-labels-idx1-ubyte.gz"
)

if [ ! -d "$FOLDER" ]; then
  mkdir "$FOLDER"
fi

for filename in "${FILENAMES[@]}"; do
  basename=${filename%.gz}
  if [ ! -e "$FOLDER$basename" ]; then
    curl -s --output "$FOLDER$filename" "$HOST$filename"
    gunzip "$FOLDER$filename"
  fi
done
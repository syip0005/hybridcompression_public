#!/bin/bash

# TO BE REMOVED IN FINAL REPO, TEMPORARY USE

# Create temporary folder
mkdir ./temp
mkdir ./raw/ideal

# Download CERN, UMASS and DRED
gdown -O ./temp https://drive.google.com/uc?id=1HwrjCNdduTGZWGLCnZfKkeOoXC0unmTS

cd ./raw/ideal

# Unzip
for f in ../../temp/*; do
    7za -e "$f"
done

# Remove temporary folder
rm -r ../../temp


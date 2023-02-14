#!/bin/bash

# Create temporary folder
mkdir ./temp
mkdir ./raw
mkdir ./interim
mkdir ./final

# Download CERN, UMASS and DRED - RAW
gdown -O ./temp https://drive.google.com/uc?id=1_zRek7V_zp8sDG5sKOim8YwVahaNd5-r

# Unzip
for f in ./temp/*; do
    unzip -d ./raw "$f"
done

# Download CERN, UMASS and DRED - INTERIM
gdown -O ./temp https://drive.google.com/uc?id=1uc7PDJDbIpY8v8HV0dgfK5qZzT6P-1lW

# Unzip
for f in ./temp/*; do
    unzip -d ./interim "$f"
done

# Remove temporary folder
rm -r ./temp
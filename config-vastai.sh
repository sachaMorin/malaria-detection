#!/bin/bash

# Simple script to set up experiment via SSH on a vast.ai GPU

# Install bash utilities
apt update
apt install wget
apt install unzip
apt install python3-pip

# Python packages
conda install matplotlib

# Create directories
mkdir data
mkdir reports
mkdir data/external
mkdir data/processed
mkdir data/raw

# Download dataset from google drive and unzip
echo -n 'Downloading dataset from a zip file hosted on Google Drive...'
echo -n 'Please enter dataset file name : '
read file_name

# Id can be found in the google drive URL
echo -n 'Please enter dataset Google Drive id (can be found in share url) : '
read file_id


wget --save-cookies cookies.txt \
    'https://docs.google.com/uc?export=download&id='$file_id -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

wget --load-cookies cookies.txt -O $file_name \
     'https://docs.google.com/uc?export=download&id='$file_id'&confirm='$(<confirm.txt)

# Unzip dataset
mv $file_name data/external/$file_name
unzip data/external/$file_name -d data/raw

rm cookies.txt
rm confirm.txt

# Install requirements and project
pip3 install -r requirements.txt
pip3 install .



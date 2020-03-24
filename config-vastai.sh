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

# Download dataset
wget 'ftp://lhcftp.nlm.nih.gov/Open-Access-Datasets/Malaria/cell_images.zip'

# Unzip dataset
mv $file_name data/external/$file_name
unzip data/external/$file_name -d data/raw

rm cookies.txt
rm confirm.txt

# Install requirements and project
pip3 install -r requirements.txt
pip3 install .



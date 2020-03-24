#!/bin/bash

# Simple script to set up experiment via SSH on a vast.ai GPU

# Install utilities
apt update
apt install wget
apt install unzip
apt install python3-pip

# Install requirements and project
pip install -r requirements.txt
pip install .

# Create directories
mkdir data
mkdir reports
mkdir data/external
mkdir data/processed
mkdir data/raw

# Download dataset
wget 'ftp://lhcftp.nlm.nih.gov/Open-Access-Datasets/Malaria/cell_images.zip' -P data/external
wget 'ftp://lhcftp.nlm.nih.gov/Open-Access-Datasets/Malaria/patientid_cellmapping_parasitized.csv' -P data/external
wget 'ftp://lhcftp.nlm.nih.gov/Open-Access-Datasets/Malaria/patientid_cellmapping_uninfected.csv' -P data/external

# Unzip dataset
unzip data/external/cell_images.zip -d data/raw

# Make dataset
python3 src/data/make_dataset.py



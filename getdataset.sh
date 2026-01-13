#!/usr/bin/env bash

wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/fwhytt5mzd-2.zip
unzip Axial\ CT\ Imaging\ Dataset\ for\ AI-Powered\ Kidney\ Stone\ Detection\ A\ Resource\ for\ Deep\ Learning\ Research.zip
unrar x Axial\ CT\ Imaging\ Dataset\ for\ AI-Powered\ Kidney\ Stone\ Detection\ A\ Resource\ for\ Deep\ Learning\ Research/Kindey\ Stone\ Dataset.rar
mkdir -p augmented
mkdir -p traindata
mv Kindey\ Stone\ Dataset/Augmented augmented
mv Kindey\ Stone\ Dataset/Original traindata

# print success in color red
echo -e "\e[31mDataset downloaded and prepared successfully!\e[0m"
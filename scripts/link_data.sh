#!/usr/bin/env bash

folder_source=$1  # /media/student/data
folder_project=$2  # /home/student/Workspace/lingofunk/lingofunk-generate
folder_target=$2/data

file=$3  # review.csv

ln -s $folder_source/$file $folder_target/$file
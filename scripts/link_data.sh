#!/usr/bin/env bash

file_source=$1  # /media/student/data/restaurant_reviews.csv
file_target=$2  # /home/student/Workspace/lingofunk/lingofunk-generate/data/reviews.scv

ln -s $file_source $file_target
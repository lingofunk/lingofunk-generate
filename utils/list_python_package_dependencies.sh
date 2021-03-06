#!/bin/sh
#
# Joshua Miller <pitt.joshua.miller@gmail.com> 
#
# Outputs the dependencies for a pip package name provided on the command line.
#

PACKAGE=$1
pip download $PACKAGE -d /tmp --no-binary :all: \
| grep Collecting \
| cut -d' ' -f2 \
| grep -Ev "$PACKAGE(~|=|\!|>|<|$)"


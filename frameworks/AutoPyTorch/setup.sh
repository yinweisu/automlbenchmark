#!/usr/bin/env bash
HERE=$(dirname "$0")
#AMLB_DIR="$1"
#VERSION=${2:-"v.0.6.0"}

# creating local venv
. $HERE/../shared/setup.sh $HERE

git clone https://github.com/automl/Auto-PyTorch.git
cat $HERE/requirements.txt | sed '/^$/d' | while read -r i; do PIP install "$i"; done
cat $HERE/Auto-PyTorch/requirements.txt | sed '/^$/d' | while read -r i; do PIP install "$i"; done

PIP install -e Auto-PyTorch

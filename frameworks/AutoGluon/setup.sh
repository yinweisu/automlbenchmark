#!/usr/bin/env bash
HERE=$(dirname "$0")

# creating local venv
. $HERE/../shared/setup.sh $HERE
#if [[ -x "$(command -v apt-get)" ]]; then
#    SUDO apt-get install -y libomp-dev
if [[ -x "$(command -v brew)" ]]; then
    brew install libomp
fi

cat $HERE/requirements.txt | sed '/^$/d' | while read -r i; do PIP install "$i"; done

# git clone https://github.com/awslabs/autogluon.git
git clone -b tabular_custom_model_support --single-branch https://github.com/awslabs/autogluon.git

PIP install -e autogluon

#PIP install --no-cache-dir -r $HERE/requirements.txt

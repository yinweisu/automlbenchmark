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

#####
PIP install -U setuptools
git clone -b tabular_stateful_preprocessing --single-branch https://github.com/awslabs/autogluon.git
# git clone -b tabular_xgboost --single-branch https://github.com/sackoh/autogluon.git
cd autogluon
PIP install -e core/
PIP install -e tabular/
PIP install -e mxnet/
PIP install -e extra/
PIP install -e text/
PIP install -e vision/
PIP install -e autogluon/
cd ..
#####

#PIP install --no-cache-dir -r $HERE/requirements.txt

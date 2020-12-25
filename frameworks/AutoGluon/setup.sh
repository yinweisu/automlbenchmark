#!/usr/bin/env bash
HERE=$(dirname "$0")
AMLB_DIR="$1"
VERSION=${2:-"stable"}
REPO=${3:-"https://github.com/awslabs/autogluon.git"}
PKG=${4:-"autogluon"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi
# FIXME: Hack
if [[ "$VERSION" == "stable" ]]; then
    VERSION="0.0.16b20201224"
fi

# creating local venv
. ${HERE}/../shared/setup.sh ${HERE}
#if [[ -x "$(command -v apt-get)" ]]; then
#    SUDO apt-get install -y libomp-dev
if [[ -x "$(command -v brew)" ]]; then
    brew install libomp
fi

PIP install --upgrade pip
PIP install --upgrade setuptools
PIP install "mxnet<=2.0.0"

if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir -U ${PKG}
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U ${PKG}==${VERSION}
else
#    PIP install --no-cache-dir -e git+${REPO}@${VERSION}#egg={PKG}
    TARGET_DIR="${HERE}/lib/${PKG}"
    rm -Rf ${TARGET_DIR}
    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    PIP install -U -e ${TARGET_DIR}
fi

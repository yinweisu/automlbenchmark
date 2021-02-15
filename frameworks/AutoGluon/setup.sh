#!/usr/bin/env bash
HERE=$(dirname "$0")
AMLB_DIR="$1"
VERSION=${2:-"stable"}
REPO=${3:-"https://github.com/awslabs/autogluon.git"}
PKG=${4:-"autogluon"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi
# TODO: Hacked in until 0.1 releases
if [[ "$VERSION" == "stable" ]]; then
    VERSION="0.0.16b20210211"
fi

# creating local venv
. ${HERE}/../shared/setup.sh ${HERE}
#if [[ -x "$(command -v apt-get)" ]]; then
#    SUDO apt-get install -y libomp-dev
if [[ -x "$(command -v brew)" ]]; then
    brew install libomp
fi

PIP install --upgrade pip
PIP install --upgrade setuptools wheel
PIP install 'numpy==1.19.5'
PIP install 'scipy==1.5.4'

# ConfigSpace MUST be installed after correct cython and numpy installed
# otherwise it will compile against the version in Conda (1.20.x)
PIP install 'ConfigSpace==0.4.14' --no-binary :all:
PIP install "mxnet<2.0.0"

if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir -U ${PKG}
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U ${PKG}==${VERSION}
else
#    PIP install --no-cache-dir -e git+${REPO}@${VERSION}#egg={PKG}

    # FIXME: HACK
    VERSION="update_versions"
    # REPO="https://github.com/gradientsky/autogluon.git"

    TARGET_DIR="${HERE}/lib/${PKG}"
    rm -Rf ${TARGET_DIR}
    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    cd ${TARGET_DIR}
    PIP install -e core/
    PIP install -e features/
    PIP install -e tabular/
    PIP install -e mxnet/
    PIP install -e extra/
    PIP install -e text/
    PIP install -e vision/
    PIP install -e autogluon/
fi

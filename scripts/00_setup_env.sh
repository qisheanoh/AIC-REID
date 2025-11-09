#!/usr/bin/env bash
set -e
conda create -n video-reid python=3.10 -y || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate video-reid
pip install -r requirements.txt

# Either: use pip package
pip install torchreid

# Or: add as submodule (comment pip line above if you choose this)
# git submodule add https://github.com/KaiyangZhou/deep-person-reid third_party/deep-person-reid || true
# pushd third_party/deep-person-reid
# pip install -r requirements.txt
# python setup.py develop
# popd
echo "Environment ready."

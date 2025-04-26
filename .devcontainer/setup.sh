#!/bin/bash

# configure git
git config --global user.name ${GIT_USERNAME}
git config --global user.email ${GIT_EMAIL}

# install dependencies
sudo apt update
sudo apt upgrade
sudo apt install cmake build-essential swig zlib1g zlib1g-dev python3-dev python3-opencv -y
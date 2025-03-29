#!/bin/bash

sudo apt update
sudo apt-get install cmake -y

# configure git
git config --global user.name ${GIT_USERNAME}
git config --global user.email ${GIT_EMAIL}

# # install rom for retro
# if [ ! -f data/sf2.md ]; then
#     sudo apt-get install p7zip-full -y
#     wget https://edgeemu.net/down.php?id=12765 -O sf2.7z
#     7z x sf2.7z
#     rm sf2.7z
#     mv "Street Fighter II' - Special Champion Edition (USA).md" data/sf2.md
#     uv run -- python -m retro.import data/
# fi
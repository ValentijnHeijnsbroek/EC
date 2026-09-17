#!/usr/bin/bash
# Ubuntu updates
sudo nala update
sudo nala upgrade --purge --assume-yes --simple

#  Python tool updates
uv self update
uv tool upgrade --all
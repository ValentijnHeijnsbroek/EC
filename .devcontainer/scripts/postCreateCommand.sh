#!/usr/bin bash
################################################
# Ubuntu universe and multiverse: https://help.ubuntu.com/community/Repositories/Ubuntu
################################################
sudo apt-get update --yes
sudo apt-get install software-properties-common --yes

################################################
# Nala: https://gitlab.com/volian/nala
################################################
sudo apt-get update --yes
sudo apt-get install nala --yes
# sudo nala fetch --sources --auto
sudo nala update
sudo nala upgrade --purge --assume-yes --simple

################################################
# Install tools
################################################
# Desired tools
sudo nala install \
    --update \
    curl \
    git \
    gnupg \
    locales \
    software-properties-common \
    zsh \
    wget \
    fd-find \
    bat \
    -y
sudo locale-gen en_US.UTF-8

# Copy zsh profile
cp .devcontainer/dotfiles/.zshrc ~

# F-Sy-H: github.com/z-shell/F-Sy-H
git clone https://github.com/z-shell/F-Sy-H.git \
    ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/F-Sy-H

# zsh-autosuggestions: https://github.com/zsh-users/zsh-autosuggestions/tree/master
git clone https://github.com/zsh-users/zsh-autosuggestions \
    ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

# fzf: https://github.com/junegunn/fzf
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
echo "y n n" | ~/.fzf/install

# starship:https://starship.rs/
curl -sS https://starship.rs/install.sh | sh -s -- -y

# Add automatic plugin update for oh-my-zsh
cat .devcontainer/dotfiles/auto-update-omz-plugins.sh >> $ZSH/tools/upgrade.sh

################################################
# Node: https://nodejs.org/en/download/
################################################
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
\. "$HOME/.nvm/nvm.sh"
nvm install 22.18.0

################################################
# Python
################################################
# Requirements
sudo nala install \
    --update \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    curl \
    git \
    libncurses-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    libedit-dev \
    -y

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup Python versions
uv python install 3.10.18 3.11.13 3.12.11 3.13.7
uv python install 3.12.11 --default

# Nox: https://nox.thea.codes/en/stable/
uv tool install nox

# Cookiecuter: https://github.com/cookiecutter/cookiecutter
uv tool install cookiecutter

# Ruff: https://docs.astral.sh/ruff/
uv tool install ruff

# Mypy: https://github.com/python/mypy
uv tool install mypy

# Install Dependencies
# uv sync

# Install pre-commit  hooks
# uv run pre-commit install --install-hooks

################################################
# Utils
################################################
# Easy update script
chmod +x .devcontainer/scripts/update.sh
# .devcontainer/scripts/update.sh

# Automatic install of missing types
chmod +x .devcontainer/scripts/mypy-missing-types.sh
# .devcontainer/scripts/mypy-missing-types.sh

################################################
# GUI
################################################
sudo nala install \
    --update \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    python3-opencv \
    libegl1 \
    x11-xserver-utils \
    -y

sudo nala install --update \
    apt-transport-https \
    ca-certificates \
    gnupg \
    -y

sudo nala install --update \
    libatk1.0-0 \
    libcairo2 \
    libcups2 \
    libexpat1 \
    libfontconfig1 \
    libfreetype6 \
    libgtk2.0-0 \
    libpango-1.0-0 \
    libx11-xcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxrandr2 \
    libxrender1 \
    libxss1 \
    libxtst6 \
    libxshmfence-dev \
    openssh-client \
    x11-apps \
    libgl1-mesa-dev \
    libosmesa6-dev \
    -y

# Clean up the package cache
sudo rm -rf /var/lib/apt/lists/*


# ------------------------------------------------
# Base Image
# ------------------------------------------------
ARG UV_VERSION=latest

FROM ghcr.io/astral-sh/uv:$UV_VERSION AS uv

FROM mcr.microsoft.com/devcontainers/base:noble

ENV PYTHONDONTWRITEBYTECODE=True \
    PYTHONUNBUFFERED=True \
    UV_LINK_MODE=copy

SHELL ["/bin/bash", "-c"]

# ------------------------------------------------
# Install dependencies and Nala
# ------------------------------------------------
RUN apt-get update -y && \
    apt-get install -y software-properties-common && \
    nala update && \
    nala upgrade --purge --assume-yes --simple && \
    nala install --update -y software-properties-common curl git gnupg locales zsh wget fd-find bat \
    make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncurses-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev libedit-dev \
    ffmpeg libsm6 libxext6 libgl1 python3-opencv libegl1 x11-xserver-utils \
    apt-transport-https ca-certificates \
    libatk1.0-0 libcairo2 libcups2 libexpat1 libfontconfig1 libfreetype6 libgtk2.0-0 \
    libpango-1.0-0 libx11-xcb1 libxcomposite1 libxcursor1 libxdamage1 libxfixes3 libxi6 \
    libxrandr2 libxrender1 libxss1 libxtst6 libxshmfence-dev openssh-client x11-apps && \
    locale-gen en_US.UTF-8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------
# Setup Zsh, Oh-My-Zsh, plugins
# ------------------------------------------------
RUN git clone https://github.com/ohmyzsh/ohmyzsh.git ~/.oh-my-zsh && \
    git clone https://github.com/z-shell/F-Sy-H.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/F-Sy-H && \
    git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf && \
    echo "y n n" | ~/.fzf/install && \
    curl -sS https://starship.rs/install.sh | sh -s -- -y

# Copy zsh config and scripts
COPY .devcontainer/dotfiles/.zshrc /root/.zshrc
COPY .devcontainer/dotfiles/auto-update-omz-plugins.sh /tmp/auto-update-omz-plugins.sh
RUN cat /tmp/auto-update-omz-plugins.sh >> ~/.oh-my-zsh/tools/upgrade.sh

COPY .devcontainer/scripts/update.sh /usr/local/bin/update.sh
COPY .devcontainer/scripts/mypy-missing-types.sh /usr/local/bin/mypy-missing-types.sh
RUN chmod +x /usr/local/bin/update.sh /usr/local/bin/mypy-missing-types.sh
# Insert auto-update script before "exit $ret"
RUN sed -i '/^exit \$ret$/i \
    # --- auto-update-omz-plugins ---\
    ' ~/.oh-my-zsh/tools/upgrade.sh && \
    sed -i '/^exit \$ret$/r .devcontainer/scripts/mypy-missing-types.sh' ~/.oh-my-zsh/tools/upgrade.sh

# awk '/# Exit with `1` if the update failed/ {
#     while ((getline line < ".devcontainer/dotfiles/auto-update-omz-plugins.sh") > 0) print line
# }
# { print }' ~/.oh-my-zsh/tools/upgrade.sh > tmp && mv -f tmp ~/.oh-my-zsh/tools/upgrade.sh

# ------------------------------------------------
# Node.js via NVM
# ------------------------------------------------
ENV NVM_DIR=/root/.nvm
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash && \
    source $NVM_DIR/nvm.sh && \
    nvm install 22.18.0

# ------------------------------------------------
# Python via uv
# ------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.cargo/bin/uv python install 3.10.18 3.11.13 3.12.11 3.13.7 && \
    /root/.cargo/bin/uv python install 3.12.11 --default && \
    /root/.cargo/bin/uv tool install nox && \
    /root/.cargo/bin/uv tool install cookiecutter && \
    /root/.cargo/bin/uv tool install ruff && \
    /root/.cargo/bin/uv tool install mypy

# ------------------------------------------------
# Default shell
# ------------------------------------------------
CMD [ "zsh" ]

COPY --from=uv --chown=vscode: /uv /uvx /bin/

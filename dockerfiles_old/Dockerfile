# Successfully built dockerfile to use Nsight Systems on Ubuntu 20.04 with pyenv and pyenv-virtualenv
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu20.04

RUN apt-get update && \
    apt-get install -y wget
# RUN git --version

# ## Install Nsight Systems
RUN wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_4/NsightSystems-linux-cli-public-2024.4.1.61-3431596.deb

# # Set the timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime 

RUN apt-get install -y ./NsightSystems-linux-cli-public-2024.4.1.61-3431596.deb

RUN rm ./NsightSystems-linux-cli-public-2024.4.1.61-3431596.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN sysctl -w kernel.perf_event_paranoid=1

# ## Install basic tools (list of necessary tools are taken from https://qiita.com/nyakiri_0726/items/a33f404b5e1be9352b85)
# `apt-get update` is necessary before `apt-get install` to avoid error
# RUN apt-get update && apt-get install -y make build-essential git vim zlib1g-dev
RUN apt-get update && apt-get install -y build-essential git vim libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev \
    liblzma-dev python-openssl git vim less

# # Environment variables
ENV HOME /root
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/bin:$PATH
ARG PYTHON_VERSION=3.7.17

RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv

RUN echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc && \
    echo 'eval "$(pyenv init --path)"' >> ~/.bashrc && \
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc && \
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc && \
    echo 'export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' >> ~/.bashrc

# it may takes time in `patching file setup.py`
RUN pyenv install 3.7.17

# Install pyenv-virtualenv
# NOTE: it should not be `$(PYENV_ROOT)/plugins/pyenv-virtualenv` but `$PYENV_ROOT/plugins/pyenv-virtualenv`
RUN git clone https://github.com/pyenv/pyenv-virtualenv.git $PYENV_ROOT/plugins/pyenv-virtualenv
RUN echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
# RUN source ~/.bashrc
RUN pyenv virtualenv 3.7.17 nerf_env


CMD [ "/bin/bash" ]

FROM nvcr.io/nvidia/pytorch:25.04-py3

# 预安装 zsh 和 git，这样启动容器只需 1 秒
RUN apt-get update && apt-get install -y zsh git && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workdir
CMD ["zsh"]
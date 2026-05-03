FROM python:3.13-slim

WORKDIR /langchain_learning

RUN apt-get update && apt-get install -y \
    build-essential \
    default-libmysqlclient-dev \
    pkg-config \
    curl \
    sudo \
    g++ \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=masashi
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Poetryのインストール
ENV POETRY_HOME="/opt/poetry"
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# venvを作らずコンテナに直接インストールする設定
RUN poetry config virtualenvs.create false

RUN chown -R $USERNAME:$USER_GID /langchain_learning

USER $USERNAME

# 依存関係のファイルコピー
COPY pyproject.toml poetry.lock* ./

# 依存関係のインストール
RUN poetry install --no-interaction --no-ansi --only main --no-root

COPY . .
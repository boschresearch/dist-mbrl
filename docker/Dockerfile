FROM ubuntu:20.04 as base

# Disable apt auto-cleanup to enable caching
RUN rm -f /etc/apt/apt.conf.d/docker-clean

##### APT ######
FROM base AS apt
# Install libraries needed for Mujoco
RUN --mount=type=cache,target=/var/cache/apt,id=apt \
    --mount=type=cache,target=/var/lib/apt/lists,id=apt-list \
    DEBIAN_FRONTEND=noninteractive apt update \
    && DEBIAN_FRONTEND=noninteractive apt install -y \
      wget \
      vim \
      # Software-rendering only
      libgl1-mesa-glx \
      libosmesa6-dev \
      libglfw3 \
      patchelf \
      xvfb \
      freeglut3-dev

# Set rendering backend for mujoco
ENV MUJOCO_GL="osmesa"

##### PYTHON ######
FROM apt AS python
# Install Python and setup virtual environment
WORKDIR /
RUN --mount=type=cache,target=/var/cache/apt,id=apt \
    --mount=type=cache,target=/var/lib/apt/lists,id=apt-list \
    DEBIAN_FRONTEND=noninteractive apt update \
    && DEBIAN_FRONTEND=noninteractive apt upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt install -y software-properties-common curl \
    && DEBIAN_FRONTEND=noninteractive add-apt-repository ppa:deadsnakes/ppa -y \
    && DEBIAN_FRONTEND=noninteractive apt install -y \
        git \
        unzip \
        python3.10-dev \
    # Need to install pip this way, see:
    # https://stackoverflow.com/questions/69503329/pip-is-not-working-for-python-3-10-on-ubuntu
    && DEBIAN_FRONTEND=noninteractive curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && DEBIAN_FRONTEND=noninteractive apt install -y build-essential

ENV VIRTUAL_ENV=/opt/venv \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN --mount=type=secret,id=pipconfig,dst=/etc/pip.conf \
    --mount=type=cache,target=/root/.cache/pip,id=pip \
    python3.10 -m pip install --upgrade pip virtualenv setuptools \
    && python3.10 -m virtualenv "$VIRTUAL_ENV"
# From now on, python will point to our virtualenv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

##### DEPENDENCIES ######
FROM python AS dependencies
ENV REPOS_DIR /root/repos
WORKDIR $REPOS_DIR

# Install packages in editable mode
COPY . $REPOS_DIR/dist-mbrl/
RUN --mount=type=cache,target=/root/.cache/pip,id=pip \
    --mount=type=secret,id=pipconfig,dst=/etc/pip.conf \
    python -m pip install -e dist-mbrl/ 

# Verify installation
COPY --chmod=755 ./docker/test_install.sh /tmp/test_install.sh
RUN /tmp/test_install.sh && rm /tmp/test_install.sh
WORKDIR $REPOS_DIR/dist-mbrl

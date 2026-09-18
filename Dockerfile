# Agentic BO on GTX TITAN X (Maxwell, sm_52).
#
# Why this image exists: the host is Ubuntu 16.04 (glibc 2.23, gcc 5.5), which
# cannot build any modern Python sdist -- pillow, numpy and scikit-learn all fail
# to compile there. The container supplies a modern userspace so every dependency
# installs from a prebuilt wheel.
#
# What the container does NOT change: the NVIDIA kernel driver. It is the host's
# (465.19.01, CUDA 11.3). CUDA 12.x needs driver >= 525, so this image stays on
# CUDA 11.8, which reaches the GPU through CUDA minor-version compatibility.
#
# Base is Ubuntu 20.04, not 22.04, on purpose. Docker 20.10.7's default seccomp
# profile blocks the clone3 syscall that glibc 2.35 (22.04) uses, so anything
# threaded fails there with "getaddrinfo() thread failed to start". 20.04's
# glibc 2.31 avoids clone3 and still satisfies manylinux_2_28 wheels.
# See docs/DOCKER.md before changing the base image tag.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# This host's default docker runtime is 'nvidia', so the container hook also runs
# during build. The base image declares NVIDIA_REQUIRE_CUDA=cuda>=11.8 and the
# 465 driver reports 11.3, which makes every RUN step fail with
#   nvidia-container-cli: requirement error: unsatisfied condition: cuda>=11.8
# Disabling the declaration check must therefore happen before the first RUN.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_DISABLE_REQUIRE=1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/bin:$PATH

# /etc/apt/apt.conf.d/docker-clean runs an APT::Update::Post-Invoke 'rm' that
# fails on this host's kernel 4.15 + overlay2 combination, turning every
# apt-get update into "E: Sub-process returned an error code". Drop it first.
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

# uv provides its own CPython, so the image does not depend on Ubuntu's Python.
# UV_PYTHON_INSTALL_DIR must point outside /root: the venv's bin/python is a
# symlink to the managed interpreter, and the compose service runs as the host
# uid, which cannot traverse root's 0700 home. Left at the default, every command
# in the container fails with a misleading "python: not found".
RUN curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh
RUN uv python install 3.12 \
    && uv venv --python 3.12 /opt/venv \
    && chmod -R a+rX /opt/uv-python /opt/venv

WORKDIR /workspace

# PyTorch first, from the cu118 index. Verified on this host: this wheel's
# torch.cuda.get_arch_list() is
#   ['sm_50','sm_60','sm_70','sm_75','sm_80','sm_86','sm_37','sm_90']
# and sm_50 binaries run on this sm_52 card. Before changing this pin, check that
# the replacement still lists sm_50 -- a wheel without it imports fine and then
# fails at the first kernel launch.
RUN uv pip install --python /opt/venv/bin/python \
        torch==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# The pinned set verified on this machine. Installing it before the project keeps
# resolution off the sdists that the old host could not build.
COPY requirements-ax-titan-x.lock ./
RUN uv pip install --python /opt/venv/bin/python -r requirements-ax-titan-x.lock

# Project last, so source edits do not invalidate the dependency layers. The
# source itself is bind-mounted at run time; this only registers the package.
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --python /opt/venv/bin/python -e . --no-deps \
    && chmod -R a+rX /opt/venv

# The CUDA base image ships a forward-compatibility driver
# (/usr/local/cuda-11.8/compat/libcuda.so.520.61.05). The nvidia runtime
# bind-mounts it over /usr/lib/x86_64-linux-gnu at container start, where
# ldconfig then resolves libcuda.so.1 to 520 instead of the host's real 465
# library sitting beside it. A 520 userspace against a 465 kernel module gives
#   RuntimeError: Error 803: system has unsupported display driver /
#   cuda driver combination
# even though nvidia-smi works. CUDA forward compatibility is a datacenter-GPU
# feature and does not apply to these GeForce cards, so drop the compat tree and
# let libcuda.so.1 resolve to the injected host driver. Removing it at run time
# is not possible: by then those paths are busy mounts.
RUN rm -rf /usr/local/cuda-11.8/compat /usr/local/cuda/compat && ldconfig

CMD ["python", "-m", "agenticbo", "--help"]

# Docker on the TITAN X host

## The two separate constraints

Keep these apart; conflating them leads to unusable images.

**1. Host userspace is too old — Docker fixes this.**
The host is Ubuntu 16.04: glibc 2.23, gcc 5.5. Modern Python sdists refuse to
build there. Installing directly on the host, `pillow` 12 fails on
`-fstack-clash-protection`, and `numpy` 2.5 stops with `NumPy requires GCC >= 10.3`.
The container's Ubuntu 20.04 userspace makes every dependency a prebuilt wheel.

**2. The GPU driver is old — Docker does NOT fix this.**
Containers share the host kernel and its NVIDIA driver. Measured here:

| | value |
|---|---|
| Driver | 465.19.01 |
| Driver's CUDA | 11.3 |
| GPUs | 4x GTX TITAN X, compute capability **5.2** (Maxwell) |

CUDA 12.x requires driver >= 525. No image can raise the host driver, so this
image stays on CUDA 11.8 and reaches the GPU through CUDA minor-version
compatibility — the same mechanism that makes the host venv work.

Two consequences:

- `nvidia-container-cli` compares the image's declared `NVIDIA_REQUIRE_CUDA`
  against the driver and refuses an 11.8 image outright:
  `unsatisfied condition: cuda>=11.8, please update your driver`.
  `NVIDIA_DISABLE_REQUIRE=1` skips that declaration check. It does not bypass
  anything in the driver itself; the CUDA 11.8 runtime genuinely works here.
- The card is sm_52 **permanently**. `torch==2.6.0+cu118` ships
  `['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_37', 'sm_90']`;
  sm_50 binaries run on sm_52. A PyTorch build whose arch list omits sm_50 will
  import successfully and then fail at the first kernel launch, which is a
  confusing way to find out. **Before adopting any newer PyTorch, check
  `torch.cuda.get_arch_list()` contains `sm_50`.** Whether current cu12x builds
  still ship it was not tested here, since the driver rules them out anyway.

So: Docker buys a modern Python toolchain and reproducible installs. It does not
buy a newer CUDA. Raising the CUDA ceiling needs a **host driver upgrade**
(root, reboot), which is a sysadmin action outside this repo. If you go that
way, confirm against NVIDIA's support matrix that the target driver branch still
supports Maxwell, and note that the sm_52 PyTorch limit above applies either
way — a newer driver does not make the GPU newer.

## Usage

```sh
docker-compose build
docker-compose run --rm bo python -m pytest -q
docker-compose run --rm bo python -m agenticbo benchmark \
  --problems branin_2d --seeds 0 --methods sobol dsp_gp --budget 8 \
  --output outputs/smoke
docker-compose run --rm bo python -m agenticbo report outputs/smoke
```

Pick GPUs with the `GPUS` variable (the compose file passes it through as
`NVIDIA_VISIBLE_DEVICES`):

```sh
GPUS=0 docker-compose run --rm bo python -m agenticbo protein --targets 22PE
```

`src/`, `configs/`, `data/`, `inputs/`, `reference/`, `outputs/` and `.cache/`
are bind-mounted, so source edits need no rebuild and Boltz weights survive one.
Rebuild only when dependencies change.

The service runs as uid/gid 1015 so that run artifacts and downloaded weights
belong to you rather than root. On a different account, export `HOST_UID` and
`HOST_GID` first. Because that uid has no home directory inside the image,
`HOME` and `MPLCONFIGDIR` are pointed at `/tmp`.

## The LLM server

Sara runs on the **host**, not in this container, and keeps GPU 3:

```sh
SARA_GPU=3 scripts/serve_sara_llamacpp.sh
```

`llama-server` binds to `127.0.0.1` only. A container on the default bridge
network therefore cannot reach it -- connecting through the docker gateway
(`172.17.0.1:8080`) returns `connection refused`. Publishing a port does not
help either, because that forwards host -> container, not the reverse.

The compose service uses `network_mode: host` for this reason. The container
shares the host network stack, so `SARA_BASE_URL=http://127.0.0.1:8080/v1`
works unchanged inside and outside the container, and `.env` needs no
container-specific variant.

The alternative -- restarting the server with `--host 0.0.0.0` -- is worth
avoiding here; see the warning in [SARA_LLAMA_CPP.md](SARA_LLAMA_CPP.md) about
the running server's deleted binary.

Keep Boltz on GPUs 0-2 while the server holds GPU 3.

## Host quirks this image works around

Docker here is 20.10.7 on kernel 4.15. Three things bite, and the Dockerfile
handles each. Do not "modernize" them away without re-testing.

**Base image must be Ubuntu 20.04, not 22.04.** Docker 20.10.7's default seccomp
profile does not permit the `clone3` syscall that glibc 2.35 (22.04) uses.
Anything threaded fails there, and the build dies at the first network call with
`curl: (6) getaddrinfo() thread failed to start`. Ubuntu 20.04's glibc 2.31 does
not use `clone3`, and at 2.31 it still satisfies `manylinux_2_28` wheels. The
alternative — `--security-opt seccomp=unconfined` — is not available to the
classic builder and weakens the sandbox, so the older base is preferred.

**`NVIDIA_DISABLE_REQUIRE` must be set before the first `RUN`.** The nvidia
runtime is this host's *default* runtime, so the container hook also runs during
`docker build`, not just at `docker run`. Setting the variable at the end of the
Dockerfile is too late; every build step fails first.

**`/etc/apt/apt.conf.d/docker-clean` has to go.** Its `APT::Update::Post-Invoke`
`rm` fails on this kernel/overlay2 pairing, so `apt-get update` exits non-zero
with `E: Sub-process returned an error code` even though the update succeeded.

**Compose is the standalone `docker-compose` v2 binary**, not the `docker compose`
plugin, which is not installed here.

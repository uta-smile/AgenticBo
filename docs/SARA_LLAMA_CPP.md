# Local Sara with Qwen3.5-9B and llama.cpp

Sara calls `http://127.0.0.1:8080/v1/chat/completions` with the model alias
`qwen3.5:9b`. The actual model file is the Q4_K_M quantization from
`unsloth/Qwen3.5-9B-GGUF`, revision
`3885219b6810b007914f3a7950a8d1b469d598a5`. This is a pinned GGUF artifact, not
a claim that it is byte-identical to Ollama's model package.

The runtime uses llama.cpp source revision
`972d2313bc0bf0a45f634f77d95c9fb03aeab12c`, built locally with GCC 9.4.0 and
CUDA 11.3.109 for compute capability 5.2. A source build avoids the installed
Ollama binary's glibc 2.27/2.28 requirement on this Ubuntu 16.04/glibc 2.23 host.
No host glibc or NVIDIA driver was upgraded.

## Start and use

```bash
scripts/serve_sara_llamacpp.sh
```

The launcher reserves GPU 3 for Sara. Use GPUs 0–2 for concurrent Boltz jobs;
the original four-GPU suite would otherwise compete with Sara on GPU 3.
`SARA_GPU` can select another GPU. Defaults are one request at a time, 8192
context tokens, Flash Attention off, Jinja tool calling, and reasoning on with
a 256-token thinking budget. The client requests at most 1024 output tokens
per response and uses a 600-second timeout for this hardware profile.

The project `.env` selects the endpoint/model. `run_poc.sh` and `run_suite.sh`
load it. For direct Python commands, export it first:

```bash
set -a
source .env
set +a
.venv/bin/python scripts/smoke_sara.py
```

The smoke test first checks `predict` and `EVALUATE` tool serialization and a
tool-result round trip. It then asks the actual Sara controller to select a
candidate using a real GP fitted to five synthetic observations. No Boltz model
is loaded, no structure is generated, and no protein evaluation is charged.
The result and visible agent trace are saved under `outputs/sara_llamacpp_smoke`.
Use `--output` with a fresh path for repeat probes.

The initial live probe passed in 60.6 seconds. In its natural controller turn,
Sara requested `suggest_local` and selected that candidate instead of the initial
advisory suggestion. GPU allocation was about 5.4 GB; sampled server timings
showed about 22–24 generated tokens/second. These are interface/performance
measurements, not evidence of protein optimization quality. All 65 project tests
also pass.

`outputs/sara_runtime.json` records model checksum, source revision, and serving
settings. `SARA_RUNTIME_MANIFEST` includes its stable identity fields in run
metadata. Update that manifest when changing the model or serving settings;
do not change a running experiment's controller identity.

## Build recipe for this host

Install the local build tools if necessary:

```bash
uv pip install --python .venv/bin/python --cache-dir .uv-cache cmake==3.31.6 ninja==1.11.1.4
```

Extract the pinned source tarball into `.cache/llama.cpp`, then configure/build:

```bash
.venv/bin/cmake -S .cache/llama.cpp -B .cache/llama.cpp/build \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-9 -DCMAKE_CXX_COMPILER=/usr/bin/g++-9 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-9 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.3/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=52 -DGGML_CUDA=ON \
  -DGGML_CUDA_FORCE_MMQ=ON -DGGML_CUDA_FA=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_UI=OFF -DCMAKE_BUILD_TYPE=Release
.venv/bin/cmake --build .cache/llama.cpp/build --target llama-server -j 6
```

Build/configuration logs are saved in `outputs/llama_build.log` and
`outputs/llama_configure.log`. Model weights live on the project's SSD under
`.cache/models`, not in the nearly full system filesystem.

## Relationship to the paper

The local PDF's Appendix A describes a different backing LLM (Claude Opus 4.8
by default) and studies reasoning levels separately. Qwen3.5-9B and its bounded
thinking budget are choices for this protein experiment, not a reproduction of
that model configuration. Sara still uses the existing four-tool-call limit and
selects candidate IDs; real protein scoring remains in the backend.

References: [llama.cpp build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md),
[tool calling](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md),
[GGUF source](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF).

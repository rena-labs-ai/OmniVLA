# rena/ — RENA VLA inference server

Hosts the OmniVLA family (edge or full) and exposes a `NavVla` predict call
over either **ROS service** or **HTTP**. Pairs with `nav_vla_client` in
`rena-control/rena_navigation/rena_navigation/vla/`.

## Layout

| File          | Role |
|---------------|------|
| `messages.py` | `Request` / `Response` dataclasses (transport-agnostic). |
| `transport.py`| `ServerTransport` ABC + `RosServerTransport` + `HttpServerTransport`. |
| `server.py`   | `Server` facade composing transport + handler. |
| `model.py`    | `OmniVlaEdgeModel` + `OmniVlaModel` (full). Each owns its preprocessing. |
| `handler.py`  | Model-agnostic `Request -> Response`: JPEG decode + PD controller. |
| `node.py`     | Entry point. Branches on `transport.type` and `model.name`. |
| `config.yaml` | All tunables. `model.name` + `device` + transport + handler PD knobs. |

`messages.py`, `transport.py`, `server.py` are duplicated from the client side
for now; will move to a shared pip package once the contract is stable.

## Pick a model

Set `model.name` in `config.yaml`:

| Name | Class | Where it runs | Extra setup |
|------|-------|---------------|-------------|
| `omnivla-edge`              | `OmniVlaEdgeModel` | Jetson edge **or** workstation | edge weights only |
| `omnivla-original`          | `OmniVlaModel` (full) | workstation | full deps + weights |
| `omnivla-original-balance`  | `OmniVlaModel` (full) | workstation | full deps + weights |
| `omnivla-finetuned-cast`    | `OmniVlaModel` (full) | workstation | full deps + weights |

A missing weights directory prints the exact `git clone https://huggingface.co/NHirose/<name>` to run.

## Setup — common to all models

```bash
python3 -m venv .venv && source .venv/bin/activate
cd /path/to/OmniVLA && pip install -e .     # picks up rena/ via setuptools.find
git lfs install                              # for the weight clones below
```

## Setup — edge model only (Jetson host)

The OmniVLA-edge wrapper runs fine on a Jetson Orin Nano (JetPack 6.2 /
L4T R36.4 / CUDA 12.6 / Python 3.10). It needs the Jetson-specific torch
wheel + a source-built torchvision; the full model's heavy deps are not
required.

```bash
pip install 'numpy<2'

# 1. cuSPARSELt 0.6.3 — NVIDIA's torch wheel links against libcusparseLt.so.0,
#    which JetPack 6.2 does not ship by default.
cd /tmp
wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-0.6.3.2-archive.tar.xz
tar -xf libcusparse_lt-linux-aarch64-0.6.3.2-archive.tar.xz
sudo cp -P libcusparse_lt-linux-aarch64-0.6.3.2-archive/lib/* /usr/local/cuda/lib64/
sudo cp -r libcusparse_lt-linux-aarch64-0.6.3.2-archive/include/* /usr/local/cuda/include/
sudo ldconfig

# 2. torch 2.5 from NVIDIA's official Jetson mirror.
pip install --no-cache-dir \
  https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# 3. torchvision 0.20 built from source against installed torch.
#    Build on the SSD (not /tmp = tmpfs); MAX_JOBS=2 to avoid OOM on 8 GB Orin.
sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev
pip install ninja 'pillow<11'
mkdir -p /mnt/jetson_data/sean/torchvision-build && cd /mnt/jetson_data/sean/torchvision-build
git clone --branch v0.20.0 https://github.com/pytorch/vision .
MAX_JOBS=2 python setup.py install     # ~30-40 min

# 4. Weights.
cd /path/to/OmniVLA
git clone https://huggingface.co/NHirose/omnivla-edge
```

Skip the official `SETUP.md`'s `pip install torch==2.2.0` (CPU-only on Jetson)
and `flash-attn` (not needed for edge inference).

## Setup — full OmniVLA (workstation)

Full OmniVLA is the OpenVLA-OFT family. Needs the moojink transformers fork
+ `flash-attn` + the prismatic library that ships with this repo.

```bash
# pyproject.toml's `transformers` line points at the moojink fork — uncomment
# it (it's already in install_requires for the full path).
pip install -e .

# Required for the OpenVLA-OFT forward path.
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation

# Download whichever checkpoint matches `model.name`:
cd /path/to/OmniVLA
git clone https://huggingface.co/NHirose/omnivla-original          # 120000 step
git clone https://huggingface.co/NHirose/omnivla-original-balance  # 120000 step
git clone https://huggingface.co/NHirose/omnivla-finetuned-cast    # 210000 step
```

Resume steps per variant are baked into `model.py` (`_FULL_RESUME_STEPS`); if
a checkpoint dir lists files under a different `--<step>_checkpoint.pt`
number, update that table.

## Run

```bash
# ROS transport (edge mode, co-located on Jetson):
source /opt/ros/<distro>/setup.bash
source ~/rena-control/install/setup.bash      # so rena_msgs is importable
cd /path/to/OmniVLA && python -m rena.node

# HTTP transport (workstation, full model). No ROS env needed.
cd /path/to/OmniVLA && python -m rena.node
```

Transport selection lives in `config.yaml` (`transport.type: ros | http`).
Pair with `rena start --robot-id <id> --nav-stack vla` on the robot, and set
the matching `transport.type` in `rena_navigation/rena_navigation/vla/config.yaml`.

## Gotchas

- **Swap matters.** Default Jetson swap is zram (RAM-backed). Add real disk
  swap on the SSD before any heavy build (`fallocate -l 8G /mnt/.../swapfile`).
- **`/tmp` is tmpfs on Ubuntu.** Building there eats RAM; build on the SSD.
- **`pypi.jetson-ai-lab.dev` is dead** — domain moved to `.io`, which has been
  flaky; NVIDIA's official mirror is the reliable fallback for the edge torch.
- **`flash-attn` build is slow** (~10-20 min on a workstation GPU). Required
  only for full OmniVLA, not for edge.

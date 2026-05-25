# rena/ — RENA edge VLA service

ROS 2 server that wraps OmniVLA-edge and exposes it as a `rena_msgs/srv/NavVla`
service. Mirrors the client scaffolding in `rena-control/rena_navigation/vla/`.

## Layout

| File          | Role |
|---------------|------|
| `messages.py` | `Request` / `Response` dataclasses (transport-agnostic). |
| `transport.py`| `ServerTransport` ABC + `RosServerTransport` impl. |
| `server.py`   | `Server` facade composing transport + handler. |
| `model.py`    | Loads OmniVLA-edge checkpoint once; exposes `forward()`. |
| `handler.py`  | Pure `Request -> Response` — preprocessing, model call, PD controller. |
| `node.py`     | rclpy entry point (the only file that imports ROS). |

`messages.py`, `transport.py`, `server.py` are currently duplicated from the
client side; they will move to a shared pip package once the contract is
stable.

## Setup on Jetson (JetPack 6.2 / L4T R36.4 / CUDA 12.6 / Python 3.10)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install 'numpy<2'

# 1. cuSPARSELt 0.6.3 — NVIDIA's torch wheel links against libcusparseLt.so.0,
#    which JetPack 6.2 does not ship by default.
cd /tmp
wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-0.6.3.2-archive.tar.xz
tar -xf libcusparse_lt-linux-aarch64-0.6.3.2-archive.tar.xz
sudo cp -P libcusparse_lt-linux-aarch64-0.6.3.2-archive/lib/* /usr/local/cuda/lib64/
sudo cp -r libcusparse_lt-linux-aarch64-0.6.3.2-archive/include/* /usr/local/cuda/include/
sudo ldconfig

# 2. torch 2.5 from NVIDIA's official Jetson mirror (no prebuilt for JP6.2 on
#    `pypi.jetson-ai-lab.io` at time of writing — that mirror was 502).
pip install --no-cache-dir \
  https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# 3. torchvision 0.20 built from source against the installed torch.
#    Build on the SSD (not /tmp = tmpfs = RAM) with MAX_JOBS=2 — each gcc/nvcc
#    job spikes ~1-2 GB and the 8 GB Orin Nano OOMs at the default nproc.
sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev
pip install ninja 'pillow<11'
mkdir -p /mnt/jetson_data/sean/torchvision-build && cd /mnt/jetson_data/sean/torchvision-build
git clone --branch v0.20.0 https://github.com/pytorch/vision .
MAX_JOBS=2 python setup.py install   # ~30-40 min on Orin Nano

# 4. Verify (cd out of the build dir, or Python imports the source tree)
cd ~
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
# expect: 2.5.0a0+872d972e41.nv24.8 0.20.0a0+afc54f7 True

# 5. OmniVLA + edge weights
cd /mnt/jetson_data/sean/OmniVLA
pip install -e .                     # picks up rena/ via setuptools.find
sudo apt-get install -y git-lfs && git lfs install
git clone https://huggingface.co/NHirose/omnivla-edge
```

Skip the official SETUP.md's `pip install torch==2.2.0` (CPU-only on Jetson)
and `flash-attn` (training-only).

## Run

```bash
source /opt/ros/<distro>/setup.bash
source ~/rena-control/install/setup.bash   # for rena_msgs
cd ~/OmniVLA && python -m rena.node
# Optional: --ros-args -p weights_path:=... -p service_name:=/vla/predict
```

Pair with `rena start --robot-id <id> --nav-stack vla` on the robot.

## Gotchas

- **Swap matters.** Default Jetson swap is zram (RAM-backed). Add real disk
  swap on the SSD before any heavy build (`fallocate -l 8G /mnt/.../swapfile`).
- **`/tmp` is tmpfs on Ubuntu.** Building there eats RAM; build on the SSD.
- **DNS for `pypi.jetson-ai-lab.dev` is dead** — domain moved to `.io`, which
  itself was returning 502 during setup. NVIDIA's official mirror is the
  reliable fallback.

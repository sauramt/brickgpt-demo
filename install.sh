#!/usr/bin/env bash
# install.sh: one-time setup of the BrickGPT demo.
#
#   ./install.sh                         Linux with an NVIDIA GPU (PyTorch with CUDA from PyPI), or macOS
#   ./install.sh --cpu                   machine without an NVIDIA GPU (smaller CPU-only PyTorch)
#   ./install.sh --reuse-torch VENV      reuse the PyTorch of an existing venv instead of downloading it
#                                        (same Python version needed), e.g. ~/gpu-work/venvs/torch
#   ./install.sh --skip-download         do not download the model weights now
#
# What it does:
#   1. creates .venv/ in this folder (Python >= 3.10; set PYTHON=python3.12 to choose one)
#   2. installs PyTorch, requirements.txt and the bundled brickgpt package (editable)
#   3. runs the brickgpt unit tests (they include the physics check)
#   4. logs you in to Hugging Face if needed and downloads the weights (~2.5 GB):
#      meta-llama/Llama-3.2-1B-Instruct (gated: request access first) + AvaLovelace/BrickGPT
# Re-running it is safe: finished steps are skipped or repeated quickly.

set -euo pipefail
cd "$(dirname "$0")"
REPO="$(pwd)"
VENV="${BRICKGPT_VENV:-$REPO/.venv}"
PY="${PYTHON:-python3}"
TORCH_MODE=default
REUSE_TORCH=""
DOWNLOAD=1

while [ $# -gt 0 ]; do
  case "$1" in
    --cpu) TORCH_MODE=cpu ;;
    --reuse-torch) REUSE_TORCH="${2:?--reuse-torch needs a venv path}"; shift ;;
    --skip-download) DOWNLOAD=0 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1 (see ./install.sh --help)"; exit 1 ;;
  esac
  shift
done

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# ---- 1. Python + venv ---------------------------------------------------------------------------
step "Python"
command -v "$PY" >/dev/null || { echo "$PY not found. Install Python 3.10-3.12 (Ubuntu: sudo apt install python3 python3-venv)."; exit 1; }
"$PY" - <<'EOF' || exit 1
import sys
v = sys.version_info
print(f"Python {v.major}.{v.minor}.{v.micro}")
if v < (3, 10):
    sys.exit("Python >= 3.10 is required (PyTorch and gradio 5 dropped 3.9).")
EOF

if [ ! -x "$VENV/bin/python" ]; then
  step "Creating virtual environment in $VENV"
  "$PY" -m venv "$VENV" || { echo "venv failed. Ubuntu: sudo apt install python3-venv"; exit 1; }
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip

# ---- 2. PyTorch -----------------------------------------------------------------------------------
step "PyTorch"
if [ -n "$REUSE_TORCH" ]; then
  # A .pth file adds the other venv's site-packages to this venv's search path (after its own
  # packages), so torch is shared without copying ~3 GB. Python minor versions must match.
  OTHER_SP="$("$REUSE_TORCH/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
  OWN_SP="$(python -c 'import site; print(site.getsitepackages()[0])')"
  [ "$(basename "$(dirname "$OTHER_SP")")" = "$(basename "$(dirname "$OWN_SP")")" ] || {
    echo "Python version of $REUSE_TORCH differs from this venv's; use PYTHON=<same python> ./install.sh ..."; exit 1; }
  echo "$OTHER_SP" > "$OWN_SP/zz_reuse_torch.pth"
  echo "Reusing torch from $OTHER_SP"
fi
if python -c 'import torch' 2>/dev/null; then
  echo "torch already available"
elif [ "$TORCH_MODE" = cpu ]; then
  pip install torch numpy --index-url https://download.pytorch.org/whl/cpu
else
  pip install torch numpy   # Linux: CUDA build from PyPI; macOS: Apple Silicon (MPS) build
fi
python - <<'EOF'
import torch
dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
name = torch.cuda.get_device_name(0) if dev == 'cuda' else dev
print(f"torch {torch.__version__} -> device: {name}")
if dev == 'cpu':
    print("WARNING: no GPU found. The demo still runs, but each structure takes several minutes.")
EOF

# ---- 3. Python packages + brickgpt ----------------------------------------------------------------
step "Python packages"
pip install -q -r requirements.txt
# --no-deps: the upstream pyproject also lists bpy (Blender, only for its renderer) and numpy<2,
# neither of which this demo needs.
pip install -q --no-deps -e brickgpt
chmod +x demo.sh tools/test_app.sh 2>/dev/null || true

step "Unit tests (brick format, collisions, physics check)"
python -m pytest -q brickgpt/tests/test_brick_structure.py

# ---- 4. Hugging Face login + weights ------------------------------------------------------------
if [ "$DOWNLOAD" = 1 ]; then
  step "Model weights"
  if ! python -c 'import sys; from huggingface_hub import get_token; sys.exit(0 if get_token() else 1)'; then
    echo "No Hugging Face token found. Paste a READ token from https://huggingface.co/settings/tokens"
    if [ -t 0 ]; then
      hf auth login || huggingface-cli login
    else
      echo "Run:  source .venv/bin/activate && hf auth login   then  ./install.sh again"; exit 1
    fi
  fi
  python - <<'EOF'
import sys
from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
try:
    snapshot_download('AvaLovelace/BrickGPT')
    # Only the files transformers loads (the tokenizer comes from the BrickGPT repo).
    snapshot_download('meta-llama/Llama-3.2-1B-Instruct',
                      allow_patterns=['config.json', 'generation_config.json', 'model.safetensors'])
except (GatedRepoError, RepositoryNotFoundError) as e:
    sys.exit("\nNo access to meta-llama/Llama-3.2-1B-Instruct with this token.\n"
             "Request access at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct (approval\n"
             "is usually quick), make sure the token belongs to that account, then re-run ./install.sh.\n"
             f"({type(e).__name__})")
print("Weights downloaded to the Hugging Face cache (~/.cache/huggingface).")
EOF
fi

step "Done"
echo "Next:  ./demo.sh check     then   ./demo.sh app   (see README.md)"

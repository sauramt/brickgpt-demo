#!/usr/bin/env bash
# demo.sh: launcher for the BrickGPT demo (run ./install.sh once first).
#
#   ./demo.sh check        self-test: GPU, cached weights, physics solver (~10 s)
#   ./demo.sh app          web demo at http://127.0.0.1:7860 (add --share for a public link)
#   ./demo.sh ablation     table prompt, seed 42, with and without physics rollback (terminal)
#   ./demo.sh cli          terminal demo: type prompts, see the log
#   ./demo.sh gen "A small boat with a flat deck." [--seeds 1 2 3]
#   ./demo.sh gallery      page with the reference runs + your runs (no GPU needed)
#   ./demo.sh validate     compare the HiGHS physics solver with Gurobi's free licence
#   ./demo.sh test-app     start the app, send one prompt through its API, stop it
#
# New runs are saved in results/runs/<prompt>_s<seed>_<mode>/ (bricks.txt, bricks.ldr,
# render.png, render_stability.png, build.gif, metrics.json).

cd "$(dirname "$0")" || exit 1
VENV="${BRICKGPT_VENV:-$(pwd)/.venv}"
if [ ! -f "$VENV/bin/activate" ]; then
  echo "No environment at $VENV. Run ./install.sh first."; exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export GRADIO_ANALYTICS_ENABLED=False PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false

# Work offline when the weights are already cached, so a weak classroom Wi-Fi cannot stall
# the model loading. If they are not cached, stay online so they can be downloaded.
if python - <<'EOF' 2>/dev/null
import sys
from huggingface_hub import try_to_load_from_cache as c
ok = all(isinstance(c(r, f), str) for r, f in [
    ('meta-llama/Llama-3.2-1B-Instruct', 'model.safetensors'),
    ('AvaLovelace/BrickGPT', 'adapter_model.safetensors')])
sys.exit(0 if ok else 1)
EOF
then export HF_HUB_OFFLINE=1; fi

open_file() {
  if command -v xdg-open >/dev/null; then xdg-open "$1" >/dev/null 2>&1 &
  elif command -v open >/dev/null; then open "$1"
  else echo "Open this file in a browser: $(realpath "$1")"; fi
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  app)      python app.py "$@" ;;
  cli)      python -u run_demo.py --interactive ;;
  gen)      python -u run_demo.py --prompts "$@" ;;
  ablation) python -u run_demo.py --seeds 42 --modes full no_rollback \
              --prompts "Table featuring a flat rectangular surface over four evenly spaced legs." ;;
  gallery)  python make_gallery.py && open_file results/gallery.html ;;
  validate) python tools/validate_solver.py "$@" ;;
  test-app) bash tools/test_app.sh ;;
  check)
    if command -v nvidia-smi >/dev/null; then
      nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
    fi
    python - <<'EOF'
import os, time
import torch
from brickgpt.data import BrickStructure
from huggingface_hub import try_to_load_from_cache
dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'torch {torch.__version__}, device: {dev}')
if dev == 'cuda':
    free, total = torch.cuda.mem_get_info()
    print(f'GPU memory free: {free / 2**30:.1f} of {total / 2**30:.1f} GB (the model needs ~5.5 GB)')
t = time.time()
ok = BrickStructure.from_txt('2x6 (0,0,0)\n2x6 (2,0,0)\n').is_stable()
bad = BrickStructure.from_txt('2x6 (0,0,0)\n2x6 (2,0,1)\n').is_stable()
print(f'physics check (HiGHS): stable pair -> {ok}, floating brick -> {bad}  ({time.time() - t:.2f} s)')
cached = True
for repo, f in [('meta-llama/Llama-3.2-1B-Instruct', 'model.safetensors'),
                ('AvaLovelace/BrickGPT', 'adapter_model.safetensors')]:
    p = try_to_load_from_cache(repo, f)
    cached &= isinstance(p, str)
    print(f'weights {repo}: {"cached" if isinstance(p, str) else "MISSING -> run ./install.sh"}')
print('offline mode:', 'on (weights cached)' if os.environ.get('HF_HUB_OFFLINE') else 'off')
print('ALL GOOD' if ok and not bad and cached else 'NOT READY (see the lines above)')
EOF
    ;;
  *) sed -n '2,15p' "$0" ;;
esac

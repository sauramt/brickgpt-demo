"""
run_demo.py: generates brick structures from text with BrickGPT and saves everything the
demo needs.

For each prompt (and each mode, see --modes) it writes results/runs/<name>/:
    bricks.txt      the structure, one "hxw (x,y,z)" brick per line (the model's own output)
    bricks.ldr      LDraw file (opens in Studio / LeoCAD / ldraw viewers)
    render.png      3D render; render_stability.png coloured by the physics stress score
    build.gif       brick-by-brick build animation
    metrics.json    bricks, rejections by reason, rollbacks, stability checks, seconds
and appends one line per run to results/summary.csv.

Modes:
    full          the paper's method: rejection sampling + physics-informed rollback,
                  stability = force-equilibrium model (solved with HiGHS, see highs_gp.py)
    no_rollback   rejection sampling only (max_regenerations=0); the final structure is
                  still analysed with the same physics model, to show what rollback fixes
    connectivity  rollback driven by the repo's simpler connectivity check (--use_gurobi False)

Run (venv active, or through ./demo.sh gen / cli / ablation):
    python run_demo.py --prompts "A sturdy wooden chair." --seeds 42
    python run_demo.py --preset paper --modes full no_rollback
    python run_demo.py --interactive          # type prompts live
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import torch

import demo_lib

OUT = Path(__file__).parent / 'results'

# Category-style prompts like the StableText2Brick captions (paper Fig. 1/5 categories).
PRESETS = {
    'paper': [
        'Table featuring a flat rectangular surface over four evenly spaced legs.',  # README example
        'A simple chair with a square seat, four legs and a tall straight backrest.',
        'A sofa with a long seat, a backrest and two armrests.',
        'A small boat with a pointed bow and a flat deck.',
        'A tall bookshelf with several horizontal shelves.',
        'A classic car with four wheels, a hood and a rounded roof.',
    ],
}
# Prompts for the live demo, fastest and clearest first (see README "Demo script").
DEMO_PROMPTS = [
    'Table featuring a flat rectangular surface over four evenly spaced legs.',
    'A small bench with a flat seat and two legs.',
    'A simple chair with a square seat, four legs and a tall straight backrest.',
    'A sofa with a long seat, a backrest and two armrests.',
    'A small boat with a pointed bow and a flat deck.',
    'A tall bookshelf with several horizontal shelves.',
]


def slug(text: str, n: int = 28) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')[:n].rstrip('_')


def run_one(model, mode: str, prompt: str, seed: int, render: bool = True) -> dict:
    name = f'{slug(prompt)}_s{seed}_{mode}'
    d = OUT / 'runs' / name
    d.mkdir(parents=True, exist_ok=True)
    print(f'\n=== [{mode}] "{prompt}"  seed={seed}', flush=True)
    r = demo_lib.generate(model, prompt, seed=seed)
    bricks = r.pop('bricks')
    (d / 'bricks.txt').write_text(bricks.to_txt())
    (d / 'bricks.ldr').write_text(bricks.to_ldr())
    r.update({'mode': mode, 'name': name, 'device': model.device,
              'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})
    (d / 'metrics.json').write_text(json.dumps(r, indent=2))
    print(f'  done in {r["seconds"]:.1f} s | {r["n_bricks"]} bricks | {r["n_rejections"]} rejections '
          f'{r["rejection_reasons"]} | {r["n_regenerations"]} rollbacks | '
          f'physically stable: {r["final_stable_physics"]}', flush=True)
    if render and len(bricks):
        t = time.time()
        demo_lib.render_png(bricks, d / 'render.png', title=None)
        try:
            scores = bricks.stability_scores()
            demo_lib.render_png(bricks, d / 'render_stability.png', scores=scores)
        except ValueError:
            pass
        demo_lib.render_gif(bricks, d / 'build.gif')
        print(f'  renders saved to {d} ({time.time() - t:.0f} s)', flush=True)

    summary = OUT / 'summary.csv'
    new = not summary.exists()
    with open(summary, 'a', newline='') as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(['name', 'mode', 'prompt', 'seed', 'seconds', 'n_bricks', 'n_rejections',
                         'n_regenerations', 'n_stability_checks', 'stability_time_s',
                         'final_stable_physics', 'final_connected', 'gpu'])
        wr.writerow([name, mode, prompt, seed, r['seconds'], r['n_bricks'], r['n_rejections'],
                     r['n_regenerations'], r['n_stability_checks'], r['stability_time_s'],
                     r['final_stable_physics'], r['final_connected'], r['gpu']])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompts', nargs='*', default=None)
    ap.add_argument('--preset', choices=list(PRESETS), default=None)
    ap.add_argument('--seeds', nargs='*', type=int, default=[42])
    ap.add_argument('--modes', nargs='*', default=['full'], choices=['full', 'no_rollback', 'connectivity'])
    ap.add_argument('--no-render', action='store_true')
    ap.add_argument('--interactive', action='store_true')
    args = ap.parse_args()

    prompts = list(args.prompts or []) + (PRESETS[args.preset] if args.preset else [])
    if not prompts and not args.interactive:
        prompts = PRESETS['paper'][:1]

    mode_kwargs = {'full': dict(rollback=True, physics=True),
                   'no_rollback': dict(rollback=False, physics=True),
                   'connectivity': dict(rollback=True, physics=False)}
    print('Loading Llama-3.2-1B-Instruct + BrickGPT weights ...', flush=True)
    t = time.time()
    # One model in memory; the modes only differ in config fields, switched per run.
    model = demo_lib.load_model(**mode_kwargs['full'])
    print(f'Loaded on {model.device} in {time.time() - t:.0f} s', flush=True)

    def set_mode(mode):
        kw = mode_kwargs[mode]
        model.max_regenerations = 100 if kw['rollback'] else 0
        model.use_gurobi = kw['physics']

    if args.interactive:
        while True:
            prompt = input('\nDescribe an object (Enter to quit): ').strip()
            if not prompt:
                break
            s = input('Seed [42]: ').strip()
            set_mode('full')
            run_one(model, 'full', prompt, int(s) if s else 42, render=not args.no_render)
        return

    for prompt in prompts:
        for seed in args.seeds:
            for mode in args.modes:
                set_mode(mode)
                run_one(model, mode, prompt, seed, render=not args.no_render)


if __name__ == '__main__':
    sys.exit(main())

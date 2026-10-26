"""
validate_solver.py: checks that the HiGHS backend (highs_gp.py) gives the same stability
analysis as Gurobi on every structure Gurobi's free pip licence can still solve.

Inputs : none (structures are built here: the repository's unit-test cases, hand-made
         stable/unstable cases, random small structures, and prefixes of any generated
         structure found in reference_runs/runs/ and results/runs/).
Outputs: results/solver_validation.csv  one row per structure and solver comparison
         results/solver_validation.json summary (agreement counts, timings)
Run    : python tools/validate_solver.py [--n-random 150] [--seed 0]   (or ./demo.sh validate)
Needs  : gurobipy (pip; without a licence file it uses the size-limited free licence,
         2,000 variables / constraints) and the patched brickgpt package.
"""

import argparse
import csv
import glob
import json
import random
import time
from pathlib import Path

import numpy as np

from brickgpt.data import Brick, BrickStructure, brick_library
from brickgpt.stability_analysis import StabilityConfig, stability_analysis, use_solver

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results'

# Hand-made cases: (name, bricks in txt format). The first two are the repo's unit tests.
HAND_CASES = [
    ('repo_test_stable', '2x6 (0,0,0)\n2x6 (2,0,0)\n'),
    ('repo_test_floating', '2x6 (0,0,0)\n2x6 (2,0,1)\n'),
    ('tower_1x1_x8', ''.join(f'1x1 (5,5,{z})\n' for z in range(8))),
    ('staircase_2x4', ''.join(f'2x4 ({z},0,{z})\n' for z in range(8))),
    ('long_cantilever_1x2', '1x2 (5,5,0)\n' + ''.join(f'1x2 (5,{5 + z},{z})\n' for z in range(1, 8))),
    ('table', '1x1 (0,0,0)\n1x1 (0,5,0)\n1x1 (5,0,0)\n1x1 (5,5,0)\n'
              '1x1 (0,0,1)\n1x1 (0,5,1)\n1x1 (5,0,1)\n1x1 (5,5,1)\n'
              '6x2 (0,0,2)\n6x2 (0,2,2)\n6x2 (0,4,2)\n'),
    ('hanging_from_top', '2x2 (0,0,0)\n2x2 (0,0,1)\n1x8 (0,0,2)\n1x1 (0,7,1)\n1x1 (0,7,0)\n'),
]


def random_structure(rng: random.Random, n_bricks: int, region: int = 8) -> str:
    """Stacks random bricks in a small region: each brick sits on the ground or on top of
    an existing brick, so it is connected, but overhangs can make it unstable."""
    sizes = [(h, w) for v in brick_library.values() for (h, w) in
             [(v['height'], v['width']), (v['width'], v['height'])]]
    sizes = sorted(set(sizes))
    bricks = BrickStructure([], world_dim=20)
    tries = 0
    while len(bricks) < n_bricks and tries < 2000:
        tries += 1
        h, w = rng.choice(sizes)
        x, y = rng.randrange(0, region), rng.randrange(0, region)
        z = 0 if len(bricks) == 0 or rng.random() < 0.2 else rng.randrange(1, 6)
        b = Brick.from_txt(f'{h}x{w} ({x},{y},{z})')
        if not bricks.brick_in_bounds(b) or bricks.brick_collides(b):
            continue
        if z > 0 and not np.any(bricks.voxel_occupancy[b.slice_2d[0], b.slice_2d[1], z - 1]):
            continue  # must rest on something
        bricks.add_brick(b)
    return bricks.to_txt()


def analyse(bricks_txt: str, solver: str) -> dict:
    """Runs the original stability model with the chosen solver backend."""
    use_solver(solver)
    bs = BrickStructure.from_txt(bricks_txt)
    cfg = StabilityConfig(world_dimension=(20, 20, 20))
    t0 = time.time()
    scores, n_vars, n_constr, total_t, solve_t = stability_analysis.stability_score(bs.to_json(), brick_library, cfg)
    return {'scores': scores, 'n_vars': n_vars, 'n_constr': n_constr, 'time_s': time.time() - t0,
            'stable': bool(scores.max() < 1 and not bs.has_floating_bricks())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-random', type=int, default=150)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    cases = list(HAND_CASES)
    rng = random.Random(args.seed)
    for i in range(args.n_random):
        cases.append((f'random_{i:03d}', random_structure(rng, rng.randint(2, 9))))
    # Prefixes of real BrickGPT outputs: the first k bricks of each generated structure.
    for p in sorted(glob.glob(str(ROOT / '*' / 'runs' / '*' / 'bricks.txt'))):
        lines = [l for l in Path(p).read_text().splitlines() if l.strip()]
        for k in (4, 8, 12):
            if k < len(lines):
                cases.append((f'prefix{k}_{Path(p).parent.name}', '\n'.join(lines[:k]) + '\n'))

    rows = []
    for name, txt in cases:
        if not txt.strip():
            continue
        h = analyse(txt, 'highs')
        row = {'case': name, 'n_bricks': len(BrickStructure.from_txt(txt)), 'n_vars': h['n_vars'],
               'highs_stable': h['stable'], 'highs_time_s': round(h['time_s'], 4)}
        try:
            g = analyse(txt, 'gurobi')
            row.update({'gurobi_stable': g['stable'], 'gurobi_time_s': round(g['time_s'], 4),
                        'verdict_agrees': g['stable'] == h['stable'],
                        'max_score_diff': float(np.abs(g['scores'] - h['scores']).max()),
                        'gurobi_note': ''})
        except Exception as e:  # e.g. "Model too large for size-limited license"
            row.update({'gurobi_stable': '', 'gurobi_time_s': '', 'verdict_agrees': '',
                        'max_score_diff': '', 'gurobi_note': str(e).splitlines()[0][:120]})
        rows.append(row)
        print(f"{name:32s} bricks={row['n_bricks']:3d} vars={row['n_vars']:5d} "
              f"highs={row['highs_stable']!s:5s} gurobi={row['gurobi_stable']!s:5s} "
              f"agree={row['verdict_agrees']!s:5s} {row['gurobi_note']}", flush=True)

    with open(OUT / 'solver_validation.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    compared = [r for r in rows if r['verdict_agrees'] != '']
    summary = {
        'n_structures': len(rows),
        'n_compared_with_gurobi': len(compared),
        'n_too_large_for_free_gurobi': len(rows) - len(compared),
        'n_verdict_agree': sum(r['verdict_agrees'] for r in compared),
        'n_stable_gurobi': sum(r['gurobi_stable'] for r in compared),
        'n_unstable_gurobi': sum(not r['gurobi_stable'] for r in compared),
        'max_score_diff': max((r['max_score_diff'] for r in compared), default=None),
        'mean_highs_time_s': float(np.mean([r['highs_time_s'] for r in rows])),
        'mean_gurobi_time_s': float(np.mean([r['gurobi_time_s'] for r in compared])) if compared else None,
        'largest_n_vars_compared': max((r['n_vars'] for r in compared), default=None),
    }
    (OUT / 'solver_validation.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

"""debug_disagree.py: prints both solvers' internals (objective, residual) for validation cases
where HiGHS and Gurobi disagree. Run: python tools/debug_disagree.py random_010 random_131"""
import random, sys, io, contextlib, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_solver as v
from brickgpt.data import BrickStructure, brick_library
from brickgpt.stability_analysis import StabilityConfig, stability_analysis as sa, use_solver
rng = random.Random(0)
cases = {}
for i in range(150):
    cases[f'random_{i:03d}'] = v.random_structure(rng, rng.randint(2, 9))
for name in sys.argv[1:]:
    txt = cases[name]
    print('=====', name); print(txt)
    for solver in ('gurobi', 'highs'):
        use_solver(solver)
        cfg = StabilityConfig(world_dimension=(20, 20, 20), print_log=True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s, nv, nc, _, _ = sa.stability_score(BrickStructure.from_txt(txt).to_json(), brick_library, cfg)
        out = buf.getvalue().splitlines()
        keep = [l for l in out if any(k in l for k in ('Obj Val', 'Eq obj', 'did not solve', 'Optimal', 'Infeasible', 'Model fingerprint', 'Best objective', 'Solution count', 'Explored', 'Time limit', 'Iteration'))]
        print(f'-- {solver}: max score {s.max():.4f}, vars {nv}, constrs {nc}'); print('\n'.join(keep[-8:]))

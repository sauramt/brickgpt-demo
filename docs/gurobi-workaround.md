# Running BrickGPT's physics check without Gurobi

## The problem

BrickGPT decides whether a structure stands up by solving a force-equilibrium model (from
StableLego, Liu et al.): every stud contact gets unknown push/pull/friction forces, and the solver
looks for forces that balance every brick's weight and torques without exceeding what a stud
connection can hold. The official code solves it with **Gurobi**, which needs a licence:

- the free academic licence must be activated from a university network;
- the licence bundled with `pip install gurobipy` is size-limited. On our test set it refused
  132 of 157 structures, including some with only 2–3 bricks ("Model too large for size-limited
  license"). Real outputs have 40–120 bricks.

The repo's own fallback, `--use_gurobi False`, swaps the physics for a plain connectivity check
(are all bricks attached to the ground?). It ignores forces, so it is not the paper's method.

## The fix

[`brickgpt/src/brickgpt/stability_analysis/highs_gp.py`](../brickgpt/src/brickgpt/stability_analysis/highs_gp.py)
implements the handful of `gurobipy` calls the stability code uses (`Model`, `addVar(s)`,
`addConstr`, `quicksum`, `abs_`, `max_`, `optimize`, `.X`, …) and solves the model with
**HiGHS**, an open-source LP solver shipped with SciPy (`scipy.optimize.linprog`, dual simplex).
The model itself, `stability_analysis.py`, is unchanged apart from its import line:

```python
from . import highs_gp as gp          # default
# BRICKGPT_SOLVER=gurobi  -> import gurobipy as gp   (original behaviour, needs a licence)
```

## Why this is exact, not an approximation

Gurobi's model is linear except for three constructs. Each is replaced by a linear form with the
same optimum:

| Gurobi constraint | Meaning | LP replacement | Why it is exact |
|---|---|---|---|
| `t == abs_(x)` | absolute net force / torque on a brick | `t ≥ x`, `t ≥ −x` | `t` is only ever minimised (it is part of the objective with weight 1), so at the optimum `t = |x|`. |
| `m == max_(f₁…f_k)` | largest pulling force on a brick | `m ≥ fᵢ` for every i | `m` enters the objective with weight α > 0, so at the optimum `m = max fᵢ`. |
| `n * f == 0` | a stud contact either pushes (`n`) or pulls (`f`), never both | dropped | In every force and torque equation, `n` and `f` of the same contact appear with the same lever arm and opposite signs, i.e. only `n − f` matters. Every `f` also carries the positive weight β in the objective. Lowering both by min(n, f) keeps all equations and lowers the objective, so every optimum already has `n · f = 0`. |

So the LP and Gurobi's model have the same optimal value, and every LP optimum is feasible for
the original model. A structure is called stable exactly as before: zero residual force/torque on
every brick and no connection over its capacity.

One implementation detail: values below 1e-9 are reported as exactly 0. The original code flags a
brick as failing when its residual is `> 0`, so solver round-off must not count as a force.
Gurobi's own default feasibility tolerance is 1e-6, so this is the stricter setting.

## Validation

`./demo.sh validate` (= `python tools/validate_solver.py`) runs both solvers on 157 structures: the
repo's unit-test cases, hand-made cases (tower, staircase, long cantilever, a table whose middle
top brick has no leg under it, a floating brick) and 150 random small stacks. Results committed in
[`reference_runs/solver_validation.csv`](../reference_runs/solver_validation.csv):

| | |
|---|---|
| Structures solved by HiGHS | 157 / 157, about 4 ms each |
| Structures the free Gurobi licence accepted | 25 (≤ 3 bricks, ≤ 197 variables) |
| Same stable/unstable verdict | 23 / 25 |
| The 2 differences | two bricks lying side by side on the ground (obviously stable). Gurobi returned a residual of 1.6e-9 (round-off) and the original `> 0` test calls that unstable; HiGHS returns exactly 0. `python tools/debug_disagree.py random_010 random_131` prints both solvers' objective values. |
| Hand-made cases | table with an unsupported middle top brick → unstable; floating brick → unstable; tower, staircase, cantilever → stable |
| Upstream unit tests (`brickgpt/tests`) | 13 / 13 pass, including the stable/unstable pair |

In the generation runs (`reference_runs/`), one stability check took 0.04–0.6 s.

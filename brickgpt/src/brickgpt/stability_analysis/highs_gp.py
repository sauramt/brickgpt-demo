"""
highs_gp.py: a drop-in replacement for the small part of `gurobipy` that BrickGPT's
stability analysis uses, solved with the open-source HiGHS LP solver (via SciPy).

Why it exists
-------------
BrickGPT checks physical stability by solving the force/torque-equilibrium model of
Liu et al. (StableLego) with Gurobi. Gurobi needs a licence; the free pip licence is
capped at 2,000 variables / constraints, too small for a 50-150-brick structure, and the
free academic licence must be activated from a university network. This module lets the
*unchanged* model in stability_analysis.py run without Gurobi.

Why an LP solver is enough (the model is exactly reformulated, not approximated)
--------------------------------------------------------------------------------
The Gurobi model is linear except for three constructs, each replaced by an exact
linear equivalent at the optimum:

1. t == abs_(x), where t only enters the objective with a positive weight
   (eq_obj = sum of |net force| + |net torque| per brick, minimised)
   -> epigraph  t >= x,  t >= -x.  At the optimum t = |x|.
2. m == max_(f_1..f_k), where m enters the objective with weight alpha > 0
   -> epigraph  m >= f_i  for all i.  At the optimum m = max f_i.
3. n * f == 0 (a stud contact either pushes, n, or pulls, f, never both): dropped.
   In every balance equation n and f of the same contact appear with the same lever arm
   and opposite sign, so only n - f matters; and every f also carries the strictly
   positive objective weight beta (sum_f_up). Lowering both by min(n, f) keeps every
   balance and strictly lowers the objective, so every optimum already has n * f = 0.

Hence the LP has the same optimal value as the original model, and its optimal solutions
are feasible for the original model. See validate_solver.py for a numerical check
against Gurobi on structures small enough for Gurobi's free size-limited licence.

Inputs/outputs: the same calls as gurobipy (Model, addVar(s), addConstr, quicksum, abs_,
max_, setObjective, optimize, .X, .objVal, .Status, NumVars, NumConstrs).
"""

import math

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

# Values whose magnitude is below this are reported as exactly 0. The stability score
# tests "residual > 0", so solver round-off (~1e-15) must not count as an imbalance.
# Gurobi's own default feasibility tolerance is 1e-6, so 1e-9 is the stricter choice.
ZERO_TOL = 1e-9


class GRB:
    """The constants stability_analysis.py reads from gurobipy.GRB."""
    CONTINUOUS = 'C'
    MINIMIZE = 1
    INFINITY = math.inf

    class Status:
        OPTIMAL = 2
        INFEASIBLE = 3
        UNBOUNDED = 5
        OTHER = 99


class LinExpr:
    """A linear expression sum(coef_i * var_i) + const, stored sparsely."""

    __slots__ = ('terms', 'const')

    def __init__(self, terms=None, const=0.0):
        self.terms = dict(terms) if terms else {}  # variable index -> coefficient
        self.const = float(const)

    @staticmethod
    def of(x):
        # Promote a Var, LinExpr or number to a LinExpr.
        if isinstance(x, LinExpr):
            return x
        if isinstance(x, Var):
            return LinExpr({x.idx: 1.0})
        if isinstance(x, (int, float, np.floating, np.integer)):
            return LinExpr(const=float(x))
        raise TypeError(f'Unsupported operand for a linear expression: {type(x)}')

    def __add__(self, other):
        other = LinExpr.of(other)
        terms = dict(self.terms)
        for i, c in other.terms.items():
            terms[i] = terms.get(i, 0.0) + c
        return LinExpr(terms, self.const + other.const)

    __radd__ = __add__

    def __neg__(self):
        return LinExpr({i: -c for i, c in self.terms.items()}, -self.const)

    def __sub__(self, other):
        return self + (-LinExpr.of(other))

    def __rsub__(self, other):
        return LinExpr.of(other) - self

    def __mul__(self, k):
        if isinstance(k, (Var, LinExpr)):
            raise TypeError('Only products of two single variables are supported (Var * Var).')
        return LinExpr({i: c * k for i, c in self.terms.items()}, self.const * k)

    __rmul__ = __mul__

    def __eq__(self, other):
        # "expr == other" builds a constraint object, as in gurobipy.
        return _LinConstr(self - LinExpr.of(other))

    __hash__ = object.__hash__


class Var:
    """A continuous decision variable. Arithmetic on it yields LinExpr objects."""

    __slots__ = ('idx', 'model', 'name')

    def __init__(self, model, idx, name):
        self.model, self.idx, self.name = model, idx, name

    @property
    def X(self):
        # Solution value, with round-off snapped to 0 (see ZERO_TOL).
        v = float(self.model._x[self.idx])
        return 0.0 if abs(v) < ZERO_TOL else v

    def __add__(self, other):
        return LinExpr.of(self) + other

    __radd__ = __add__

    def __sub__(self, other):
        return LinExpr.of(self) - other

    def __rsub__(self, other):
        return LinExpr.of(other) - LinExpr.of(self)

    def __neg__(self):
        return -LinExpr.of(self)

    def __mul__(self, other):
        if isinstance(other, Var):
            return _Bilinear(self, other)
        return LinExpr.of(self) * other

    def __rmul__(self, other):
        return LinExpr.of(self) * other

    def __eq__(self, other):
        if isinstance(other, _Abs):
            return _AbsConstr(self, other.x)
        if isinstance(other, _Max):
            return _MaxConstr(self, other.xs)
        return LinExpr.of(self) == other

    __hash__ = object.__hash__


# Markers for the three non-linear constructs (see the module docstring).
class _Bilinear:
    def __init__(self, a, b):
        self.a, self.b = a, b

    def __eq__(self, other):
        if other != 0:
            raise ValueError('Only complementarity constraints "x * y == 0" are supported.')
        return _ComplementarityConstr(self.a, self.b)


class _Abs:
    def __init__(self, x):
        self.x = x


class _Max:
    def __init__(self, xs):
        self.xs = list(xs)


class _LinConstr:
    def __init__(self, expr):
        self.expr = expr  # means expr == 0


class _AbsConstr:
    def __init__(self, t, x):
        self.t, self.x = t, x


class _MaxConstr:
    def __init__(self, m, xs):
        self.m, self.xs = m, xs


class _ComplementarityConstr:
    def __init__(self, a, b):
        self.a, self.b = a, b


def quicksum(xs):
    out = LinExpr()
    for x in xs:
        out = out + x
    return out


def abs_(x):
    return _Abs(x)


def max_(xs, *more):
    xs = list(xs) if not more else [xs, *more]
    return _Max(xs)


class _Params:
    """Accepts any Gurobi parameter assignment and ignores it (e.g. IterationLimit)."""

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)


class Model:
    def __init__(self, name=''):
        self.name = name
        self.Params = _Params()
        self.modelSense = GRB.MINIMIZE
        self.Status = None
        self.objVal = None
        self._lb, self._ub, self._names = [], [], []
        self._eq_rows = []   # list of (dict idx->coef, rhs) meaning row . x == rhs
        self._ub_rows = []   # list of (dict idx->coef, rhs) meaning row . x <= rhs
        self._n_dropped_complementarity = 0
        self._objective = LinExpr()
        self._x = None
        self.solve_info = {}

    # --- gurobipy API subset ---------------------------------------------------------
    def setParam(self, name, value):
        setattr(self.Params, name, value)

    def addVar(self, lb=0.0, ub=math.inf, vtype=GRB.CONTINUOUS, name=''):
        if vtype != GRB.CONTINUOUS:
            raise ValueError('Only continuous variables are supported.')
        v = Var(self, len(self._lb), name)
        self._lb.append(lb)
        self._ub.append(ub)
        self._names.append(name)
        return v

    def addVars(self, n, lb=0.0, ub=math.inf, vtype=GRB.CONTINUOUS, name=''):
        # gurobipy returns a tupledict keyed 0..n-1; a list indexes the same way here.
        return [self.addVar(lb, ub, vtype, f'{name}[{i}]') for i in range(n)]

    def addConstr(self, c, name=''):
        if isinstance(c, _LinConstr):
            self._eq_rows.append((c.expr.terms, -c.expr.const))
        elif isinstance(c, _AbsConstr):
            # t >= x  and  t >= -x   ->   x - t <= 0  and  -x - t <= 0
            x = LinExpr.of(c.x)
            for s in (1.0, -1.0):
                e = x * s - c.t
                self._ub_rows.append((e.terms, -e.const))
        elif isinstance(c, _MaxConstr):
            # m >= x_i for all i   ->   x_i - m <= 0
            for xi in c.xs:
                e = LinExpr.of(xi) - c.m
                self._ub_rows.append((e.terms, -e.const))
        elif isinstance(c, _ComplementarityConstr):
            # Implied at the optimum (module docstring, point 3), so not passed to the LP.
            self._n_dropped_complementarity += 1
        else:
            raise TypeError(f'Unsupported constraint: {type(c)}')

    def setObjective(self, expr, sense=None):
        self._objective = LinExpr.of(expr)
        if sense is not None:
            self.modelSense = sense

    def update(self):
        pass

    @property
    def NumVars(self):
        return len(self._lb)

    @property
    def NumConstrs(self):
        return len(self._eq_rows) + len(self._ub_rows)

    def optimize(self):
        n = self.NumVars
        c = np.zeros(n)
        for i, coef in self._objective.terms.items():
            c[i] += coef

        def to_sparse(rows):
            if not rows:
                return None, None
            r, col, val, rhs = [], [], [], np.empty(len(rows))
            for k, (terms, b) in enumerate(rows):
                for i, coef in terms.items():
                    r.append(k)
                    col.append(i)
                    val.append(coef)
                rhs[k] = b
            return coo_matrix((val, (r, col)), shape=(len(rows), n)).tocsr(), rhs

        A_eq, b_eq = to_sparse(self._eq_rows)
        A_ub, b_ub = to_sparse(self._ub_rows)
        bounds = [(lb, None if math.isinf(ub) else ub) for lb, ub in zip(self._lb, self._ub)]

        # Dual simplex returns a vertex solution, so variables that should be 0 are
        # exactly 0 (an interior-point solution would be spread over many variables).
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                      method='highs-ds',
                      options={'primal_feasibility_tolerance': 1e-10,
                               'dual_feasibility_tolerance': 1e-10})
        self.solve_info = {'status': res.status, 'message': res.message,
                           'n_vars': n, 'n_constrs': self.NumConstrs,
                           'n_dropped_complementarity': self._n_dropped_complementarity}
        if res.status == 0:
            self.Status = GRB.Status.OPTIMAL
            self._x = res.x
            self.objVal = float(res.fun) + self._objective.const
        elif res.status == 2:
            self.Status = GRB.Status.INFEASIBLE
        elif res.status == 3:
            self.Status = GRB.Status.UNBOUNDED
        else:
            self.Status = GRB.Status.OTHER

    def close(self):
        self._x = None

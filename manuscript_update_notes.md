# Manuscript update notes

This file records approved numerical changes that must be reflected in the
manuscript later.  It is a planning record only; no `.tex` file has been
edited or compiled.

## Reviewer A1 — Optimal-control stopping rule

- Replace any optimal-control statement that says the natural-residual
  tolerance is `1e-4` with `1e-5`.
- State that convergence requires the natural residual to be at or below this
  tolerance for **two consecutive iterations**.
- The double-integrator and harmonic-oscillator benchmarks and regenerated
  outputs use this rule.

## Reviewer A2 — Forward-evaluation accounting

### Convention to state

- An F-evaluation means one evaluation of the forward operator `B` that is
  required by an algorithmic update, line search, stepsize update, or a value
  cached for a later algorithmic update.
- The initialization evaluation is counted.
- A `B` evaluation made solely for residual monitoring/stopping diagnostics is
  excluded.
- For VAFBS, every Armijo line-search trial is counted.  Consequently, its
  F-evaluation total is larger than `2 × Iter` whenever the line search
  backtracks.

### Table/reporting rule to state

- Iteration and F-evaluation columns report the **unrounded median** over the
  ten converged initial points, displayed with one decimal place.
- Do not independently round the two medians to integers.  For example,
  `2907.5` iterations and `2908.5` F-evaluations must remain displayed as
  such, preserving the difference of one.

### Accounting checks used in the regenerated benchmarks

- AEFBFP: `F = N + 1` after `N` iterations.
- MDITSM: `F = 2N`.
- RFBSM: `F = 2N + 1`.
- IRFBSM: `F = 2N`.
- IFRAB: `F = N + 1`.
- VAFBS: `F = 1 + sum(line-search trials) + N`.

### Numerical outputs already regenerated

- Compressed sensing: benchmark database, table, and figures.
- Double-integrator optimal control: benchmark database, table, and figures.
- Harmonic-oscillator optimal control: benchmark database, table, and figures.

## Reviewer B1 — Compressed-sensing stopping and final accuracy

- All compressed-sensing methods now use the common LASSO fixed-point
  optimality residual
  `r(x) = ||x - prox_{lambda*gamma*||.||_1}(x - lambda*C'(C*x-y))||`
  with the protocol-fixed measurement step `lambda = 1/L`, rather than the successive-iterate quantity
  `||x_k - x_{k-1}||`.
- The benchmark stops after this common residual is below `1e-5` for two
  consecutive iterations, subject to the `5000`-iteration cap.
- For every final run, the database stores the LASSO objective,
  reconstruction MSE relative to the planted `x_star`, and the common
  residual.  These are final-run metrics, not per-iteration history fields.
- The compressed-sensing output now includes a separate final-accuracy table
  reporting the median final objective, median MSE, and median common residual
  over converged starts.
- The rerun under the common rule changes the success results materially:
  AEFBFP and VAFBS are DNC in all four current CS cases at the `5000` cap;
  IFRAB is DNC for the fourth case and solves 9/10 starts in the second case.

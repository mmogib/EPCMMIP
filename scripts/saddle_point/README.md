# Matrix saddle-point benchmark

This study solves zero-sum bilinear games on a product of simplices. It contains three dense random games (`100 x 100`, `500 x 500`, and `1000 x 1000`, entries uniform on `[-1,1]`) and the degenerate game `K=[I_q e_1]`, `q=100`, whose saddle segment and metric-projection formula are known exactly.

Ten paired starts per instance are generated as independent `Dirichlet(1)` points by normalizing exponential draws. The main six methods use the displayed manuscript parameters. A seventh local diagnostic is plain Halpern forward-backward with `alpha_k=1/(k+2)` and `lambda=0.99/norm(K,2)`.

The stopping monitor is the unit-step natural residual at tolerance `10^-6`, two consecutive hits, and cap 10000. Monitor evaluations are excluded from operator-evaluation counts. The observer records `tau_0` at event zero and the post-iteration value `tau_{k+1}` afterward, together with the minimum-branch fraction. The final game gap is computed at the feasible shadow `P_C(u_k)`.

## Run

From `jcode/`, with Julia and BLAS restricted to one thread:

```powershell
julia --project=. scripts/saddle_point/s01_smoke_test.jl
julia --project=. scripts/saddle_point/s30_benchmark.jl --pilot-1000
julia --project=. scripts/saddle_point/s30_benchmark.jl
julia --project=. scripts/saddle_point/s60_recompute_gaps.jl
julia --project=. scripts/saddle_point/s70_tables.jl
```

The production runner checkpoints completed cells. For every start it performs two warm-ups and three measured repetitions, asserting identical flag/iteration/evaluation signatures. `s60_recompute_gaps.jl` is a resume-safe, untimed deterministic replay: it asserts the stored signature and updates only the feasible-shadow gap diagnostic. `s70_tables.jl` creates the game performance and anchor tables without plotting.

Mohammed-only plotting entry point:

```powershell
julia --project=. scripts/saddle_point/s71_figures.jl
```

Expected figure stems are `game_random500_residual` and `game_random500_aefbfp_tau` in both PDF and PNG formats. Runtime data live under `results/saddle_point/`.

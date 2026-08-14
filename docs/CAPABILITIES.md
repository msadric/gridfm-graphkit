# gridfm-graphkit — Full Capability Reference

This is a from-source reference of everything `gridfm-graphkit` can do: every
CLI subcommand and flag, every task type, every model architecture, the
dataset/normalization/splitting pipeline, the loss/callback/registry system,
and the full example-config catalog — including behavior that is not
documented (or is documented incompletely or incorrectly) in `README.md` and
`docs/quick_start/*.md`. It was written by reading the package source
directly (`gridfm_graphkit/`), not by summarizing the existing docs, so where
the two disagree this document should be treated as authoritative for the
current code.

`gridfm-graphkit` is the training/inference sister package to
`gridfm-datakit`. It consumes the `bus_data.parquet` / `gen_data.parquet` /
`branch_data.parquet` tables that `gridfm-datakit` produces (Hive-partitioned
by `scenario_partition`, 200 scenarios/partition), turns each scenario into a
PyTorch Geometric `HeteroData` graph with `bus` and `gen` node types, and
trains/evaluates/runs-inference-with graph neural networks on them for three
task families: power flow (PF), optimal power flow (OPF), and state
estimation (SE).

---

## 1. Installation

`pip install gridfm-graphkit`, plus a **mandatory manual install** of
`torch-scatter` (and, for the GRIT model's RRWP positional encoding,
`torch-sparse`) matched to the installed PyTorch/CUDA build — neither is a
declared `pyproject.toml` dependency because the correct wheel URL depends on
the local torch/CUDA versions:

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_CUDA_VERSION}.html
pip install torch-sparse   -f https://data.pyg.org/whl/torch-${TORCH_CUDA_VERSION}.html
```

`torch-scatter` is imported unconditionally by `gridfm_graphkit/tasks/opf_task.py`
and `gridfm_graphkit/training/loss.py` (hard `ImportError` if missing), and
optionally (`try/except ImportError`) by `pf_task.py`,
`gridfm_graphkit/models/grit_transformer.py`, and
`gridfm_graphkit/datasets/posenc_stats.py` — the GRIT-only code paths raise a
clear `ImportError` with an install hint at the point of use rather than at
import time. `torch-sparse` is only needed for `RRWPLinearEdgeEncoder`
(GRIT + RRWP), and is likewise a soft import with a deferred error.

`pyproject.toml` pins exact versions for `torch==2.10.0`, `torchaudio==2.11.0`,
`torchvision==0.25.0`, and `gridfm-datakit==1.0.5` (used at runtime only by
`tasks/pf_ac_dc_baseline.py` / `opf_ac_dc_baseline.py` for the
`--compute_dc_ac_metrics` ground-truth power-balance baseline, via
`gridfm_datakit.utils.power_balance`). `dev` extras add `mkdocs-material`,
`mkdocstrings[python]`, `pre-commit`, `bandit`, `build`; `test` extras add
`pytest`, `pytest-cov`.

The single console-script entry point is `gridfm_graphkit = gridfm_graphkit.__main__:main`.

---

## 2. CLI overview

`gridfm_graphkit/__main__.py` defines five subcommands via `argparse`:
`train`, `finetune`, `evaluate`, `predict`, `benchmark`. All except
`benchmark` are dispatched to `main_cli()` in `gridfm_graphkit/cli.py`;
`benchmark` goes to `benchmark_cli()` in the same file.

On any Linux host, if `--mp_context` is left unset, or is `fork`/`forkserver`,
`_warn_mp_context_on_linux()` emits a `UserWarning` recommending `spawn`
(CUDA + fork is unsafe). This check runs for every subcommand, unconditionally.

If the process detects it's running under LSF (`LSB_JOBID`, `LSB_MCPU_HOSTS`,
and `LSF_ENVDIR` all set), `main()` calls `set_env()` (derives `MASTER_ADDR`,
`MASTER_PORT`, `NODE_RANK` from LSF host-list metadata) and `fix_infiniband()`
(probes `ibv_devinfo` and sets `NCCL_IB_HCA` to exclude Ethernet-only IB
ports) **before** parsing the subcommand — this happens for every command,
not just `train`.

### 2.1 `train`

```
gridfm_graphkit train --config <path> [options]
```

| Flag | Type | Notes |
| --- | --- | --- |
| `--config` | str, **required** | Path to YAML config. |
| `--exp_name` | str | MLflow experiment name. Default: current timestamp `%Y-%m-%d_%H-%M-%S`. |
| `--run_name` | str | MLflow run name. Default `run`. |
| `--log_dir` | str | MLflow tracking dir / Lightning `default_root_dir`. Default `mlruns`. |
| `--data_path` | str | Root dataset directory (each `config.data.networks[i]` is a subdirectory). Default `data`. |
| `--compile [MODE]` | str, optional value | `torch.compile(model.model, mode=MODE)`. `MODE ∈ {default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs}`; omitting the value uses `default` (`nargs="?", const="default"`). For the two `max-autotune*` modes, `torch._inductor.config.max_autotune_gemm_backends` is forced to `"ATEN,TRITON"` so Triton configs that exceed GPU shared memory fall back to ATen instead of erroring. |
| `--bfloat16` | flag | `model.to(torch.bfloat16)` equivalent via `Trainer(precision="bf16-true")`. |
| `--tf32` | flag | `torch.set_float32_matmul_precision("high")`. |
| `--dataset_wrapper` | str | Name registered in `DATASET_WRAPPER_REGISTRY`. **The registry ships empty** — see §8.1; only usable if a plugin registers a wrapper (e.g. the `SharedMemoryCacheDataset` mentioned in every `--help` string does not exist anywhere in this package). |
| `--plugins` | list[str] | Extra packages `importlib.import_module`'d before dispatch, so their `@REGISTRY.register(...)` decorators fire. Raises `ModuleNotFoundError` with a clear message if a named plugin isn't installed. |
| `--num_workers` | int | Overrides `data.workers` from YAML. `0` is explicitly suggested for debugging worker crashes. |
| `--dataset_wrapper_cache_dir` | str | Disk cache dir for the dataset wrapper (only meaningful with `--dataset_wrapper`). |
| `--profiler` | str | Lightning profiler: `simple`, `advanced`, `pytorch`. |
| `--compute_dc_ac_metrics` | flag | After the test loop, computes ground-truth AC/DC power-balance and runtime baseline metrics directly from parquet (bypasses the model) — see §11. |
| `--report-performance` | flag (`dest=report_performance`) | **Undocumented in README.** Prints `[performance] last epoch time : Xs` and `it/s`, plus one test metric, to stdout at the end of the run. Uses an `EpochTimerCallback` appended to the callback list only when this flag is set. |
| `--mp_context` | str | DataLoader multiprocessing start method: `spawn`\|`fork`\|`forkserver`. |
| `--deterministic [true\|warn]` | str, optional value | **Undocumented in README. Only available on `train`, not `finetune`/`evaluate`/`predict`/`benchmark`.** Maps to `Trainer(deterministic=...)`. Bare `--deterministic` → `"warn"` mode (non-fatal warning on non-deterministic ops); `--deterministic true` → strict (raises on non-deterministic ops). Per the help text, CUDA≥10.2 requires `CUBLAS_WORKSPACE_CONFIG` to be set externally (e.g. `:4096:8`) for this to work — the CLI does not set it for you. |

### 2.2 `finetune`

Same flags as `train` **except**: adds required `--model_path` (state-dict
path, loaded into the freshly-constructed task model via `load_state_dict`
before `trainer.fit`), and **does not** expose `--deterministic`.
`--report-performance` is present here too, contrary to what a quick README
skim suggests (the README's finetune argument table does list it in this
repo's current state, so this part matches).

State dict loading (shared by `finetune`/`evaluate`/`predict`) goes through
`_normalize_loaded_state_dict_keys()`, which strips a `model._orig_mod.`
prefix (left behind by `torch.compile`-wrapped checkpoints) so compiled and
uncompiled checkpoints are interchangeable.

### 2.3 `evaluate`

```
gridfm_graphkit evaluate --config <path> [--model_path P] [options]
```

Adds `--model_path` (optional, unlike finetune — a config-embedded path is
implied but there is in fact no such config key read anywhere in
`main_cli`; if omitted, `state_dict = torch.load(args.model_path, ...)` will
raise `TypeError`/`FileNotFoundError` since `args.model_path` is `None`),
`--normalizer_stats` (path to a `normalizer_stats.pt` saved by a prior
`on_fit_start`; restores `fit_on_train`-strategy normalizer statistics — see
§8.2), `--batch_size` (overrides `training.batch_size` for eval only), and
`--save_output` (flag; writes predictions as `<grid_name>[_<table>]_predictions.parquet`
under `.../artifacts/test/`, same code path as `predict`).

`evaluate` never calls `trainer.fit`; it always builds a **second**,
single-device `Trainer` (`devices=1, num_nodes=1`) for `trainer.test(...)`
rather than reusing the training trainer (that reuse only happens for
`train`/`finetune`, to keep `torch.compile` kernel caches warm).

### 2.4 `predict`

```
gridfm_graphkit predict --config <path> --model_path <path> [options]
```

Adds `--output_path` (default `data`; predictions parquet is written here
directly, not under MLflow artifacts) and `--normalizer_stats`. `predict`
implicitly forces `save_output = True` regardless of any flag
(`save_output = getattr(args, "save_output", False) or args.command == "predict"`
in `cli.py`) and only calls `trainer.predict`, never `trainer.test`.

**Multi-grid limitation:** if `len(config_args.data.networks) > 1`, both
`predict` and `evaluate --save_output` raise
`NotImplementedError("Predict/save_output with multiple grids is not yet supported.")`
— this only applies to the *prediction-saving* path; multi-grid training and
plain `evaluate` (without `--save_output`) work fine.

### 2.5 `benchmark`

```
gridfm_graphkit benchmark --config <path> [--epochs N] [options]
```

Not documented at all as a subcommand purpose beyond one line in the README's
command list, though the README does have a full flag table for it. Times
`--epochs` (default `3`) full iterations of the **train** dataloader only
(`dm.setup(stage="fit")`, `dm.train_dataloader()`) — it never touches the
model or GPU compute, purely measures data-loading throughput (setup time,
batches/epoch, seconds/epoch, seconds/batch, and — for `epochs > 1` — an
average across epochs). Accepts `--dataset_wrapper`,
`--dataset_wrapper_cache_dir`, `--num_workers`, `--plugins`, `--mp_context`,
but **no** `--data_path` override docs mismatch — it does accept
`--data_path` (default `data`), just like the others.

### 2.6 Device/environment banner (all commands except `benchmark`)

`main_cli` always prints a `[device]` banner before running: hostname, CUDA
availability + GPU names + `CUDA_HOME` + resolved `nvcc` path (falls back to
probing `$CUDA_HOME/bin/nvcc` if `which nvcc` finds nothing), or MPS
detection, or an explicit `WARNING: No GPU found, running on CPU only`.

---

## 3. Config file structure

Configs are flat YAML, loaded via `yaml.safe_load` and wrapped in
`NestedNamespace` (`gridfm_graphkit/io/param_handler.py`) — a recursive
`argparse.Namespace` subclass that turns nested dicts into nested namespaces
(attribute access, `args.data.networks`) and nested lists-of-dicts into
lists of namespaces. It also exposes `.to_dict()` and `.flatten(sep=".")`.
Missing keys raise `AttributeError` unless the reading code uses
`getattr(args.x, "y", default)`; several config keys documented below are
read this way and are therefore genuinely optional, while others (e.g.
`args.callbacks.tol`, `args.callbacks.patience`) are read directly and will
crash the run at `get_training_callbacks()` if omitted.

Top-level sections: `task`, `data`, `model`, `optimizer`, `training`, `seed`,
`verbose`, `callbacks`. (`docs/quick_start/yaml_config.md` documents this
schema reasonably well for the `PowerFlow`/`OptimalPowerFlow` + `GNS_heterogeneous`
combination; it does **not** cover `StateEstimation`, the `GRIT` model, or
the positional-encoding blocks — all covered below.)

### 3.1 `task`

- `task.task_name` — one of the three `TASK_REGISTRY` keys: `PowerFlow`,
  `OptimalPowerFlow`, `StateEstimation`. This value is used **three separate
  times** for three different registries (`TASK_REGISTRY`,
  `TRANSFORM_REGISTRY`, `PHYSICS_DECODER_REGISTRY`) — see §8.1.
- `task.grid_path` (`StateEstimation` only, optional) — path to a saved
  `HeteroData.to_dict()` template used by `LoadGridParamsFromPath` to inject
  static admittance/generator-status columns into every sample (see §10).
- `task.noise_type`, `task.measurements.{power_inj,power_flow,vm}`,
  `task.relative_measurement` — `StateEstimation` measurement-simulation
  config; see §10.

### 3.2 `data`

| Key | Meaning / default |
| --- | --- |
| `data.networks` | List of dataset directory names under `--data_path`. |
| `data.scenarios` | List of scenario counts, one per network, **positionally aligned** with `data.networks`. If a count exceeds the dataset size, `UserWarning` + use full dataset. |
| `data.normalization` | `NORMALIZERS_REGISTRY` key: `HeteroDataMVANormalizer` or `HeteroDataPerSampleMVANormalizer` — see §8. |
| `data.baseMVA` | Casefile base MVA, default `100` (`getattr(args.data, "baseMVA", 100)`); used to rescale admittances back to physical units. |
| `data.mask_value` | Fill value substituted into masked feature slots (`ApplyMasking`). Examples use `0` or `0.0`. |
| `data.mask_type` | `"rnd"` for `AddRandomHeteroMask` (self-supervised random masking); any other value (or omitted) uses the deterministic `AddPFHeteroMask`/`AddOPFHeteroMask`. PF/OPF only — ignored by `StateEstimation` transforms. |
| `data.mask_ratio` | Only consulted when `mask_type: rnd`. |
| `data.test_ratio`, `data.val_ratio` | Split fractions; `val_ratio + test_ratio >= 1` raises `ValueError`. |
| `data.workers` | DataLoader worker count (also `--num_workers` CLI override). |
| `data.split_by_load_scenario_idx` | bool, default `False`. See §9.2. Mutually exclusive with `split_from_existing_files` (asserted in `LitGridHeteroDataModule.__init__`). |
| `data.split_from_existing_files` | Path to a folder with `train.pt`/`val.pt`/`test.pt` index tensors. See §9.3. |
| `data.posenc_RRWP.{enable,ksteps,cache,topk}` | GRIT-only RRWP positional encoding — see §10. |
| `data.posenc_RWSE.{enable,cache,kernel.times}` | GRIT-only RWSE positional encoding — see §10. |

### 3.3 `model`

`model.type` selects the `MODELS_REGISTRY` entry: `GNS_heterogeneous` or
`GRIT` — see §6 for the full per-model key list.

### 3.4 `optimizer`

`learning_rate`, `beta1`, `beta2` (AdamW), `lr_decay`, `lr_patience`
(`ReduceLROnPlateau(mode="min", factor=lr_decay, patience=lr_patience)`).
Not configurable: optimizer type is hardcoded to `AdamW`, scheduler type is
hardcoded to `ReduceLROnPlateau`.

### 3.5 `training`

`batch_size`, `epochs`, `accelerator`/`devices`/`strategy` (passed straight
through to `lightning.Trainer`), and the loss composition:
`training.losses` (list of `LOSS_REGISTRY` names), `training.loss_weights`
(parallel list of floats), `training.loss_args` (parallel list of
dicts/namespaces, one per loss — can be `{}`). All three lists **must** be
the same length; `MixedLoss.__init__` raises `ValueError` if
`len(loss_functions) != len(weights)`, but a `loss_args` length mismatch
instead fails with an opaque `zip`-truncation or `AttributeError` at
`get_loss_function`'s `zip(args.training.losses, args.training.loss_args)`
(silently truncates to the shorter list rather than raising — **not
validated**).

### 3.6 `callbacks`

`patience`, `tol` (both required, no defaults — see §3 note above),
`early_stopping_monitor`, `checkpoint_monitor`, `lr_scheduler_monitor` (all
three optional, default `DEFAULT_MONITOR = "Validation loss"` from
`gridfm_graphkit/training/callbacks.py`). All three metrics are minimized;
`mode="min"` is hardcoded and not configurable. The monitored name must
match a metric actually produced during validation (e.g.
`Validation layer_11_residual`, only emitted by `LayeredWeightedPhysics`
with a 12-layer model, index 0–11) — pointing at a non-existent metric
raises `KeyError` from `SaveBestModelStateDict.on_validation_end` (with the
list of available metrics attached) once training genuinely starts (the
Lightning sanity-check validation pass is explicitly skipped).

### 3.7 `seed`, `verbose`

`seed` seeds `lightning.seed_everything(seed, workers=True)` and is also
used directly for `random.seed`/`np.random.seed` calls during dataset
subsetting/splitting (§9). `verbose: true` enables extra per-bus-type
residual plots and CSVs during `test`/`predict` — see §5.

---

## 4. Tasks (`gridfm_graphkit/tasks/`)

All three tasks are `TASK_REGISTRY`-registered subclasses of
`ReconstructionTask` (`gridfm_graphkit/tasks/reconstruction_tasks.py`),
itself a subclass of the abstract `BaseTask` (`base_task.py`), a
`lightning.LightningModule`.

### 4.1 `BaseTask` (shared machinery)

- `transfer_batch_to_device`: pre-casts `float64` tensors to `float32`
  before an MPS move (MPS doesn't support float64; some PyG metadata gets
  collated as float64 even with float32 model inputs).
- `on_after_batch_transfer`: casts all floating-point tensors in the
  `HeteroData` batch to the model's parameter dtype — needed because
  Lightning's automatic mixed-precision casting doesn't understand PyG
  `HeteroData`/`Batch` objects, so `--bfloat16` would silently produce
  dtype-mismatch errors without this.
- `on_fit_start` (`@rank_zero_only`): writes
  `.../artifacts/stats/normalization_stats.txt` (human-readable) and
  `normalizer_stats.pt` (machine-loadable, keyed by network name) — this is
  the file consumed by `--normalizer_stats`.
- `configure_optimizers`: hardcoded `AdamW` + `ReduceLROnPlateau`, as in §3.4.

### 4.2 `ReconstructionTask` (`reconstruction_tasks.py`)

Defines `shared_step` (forward pass + `loss_fn(output, batch.y_dict,
batch.edge_index_dict, batch.edge_attr_dict, batch.mask_dict, model=self.model,
x_dict=batch.x_dict)`), `training_step`, `validation_step`, and
`on_test_end` (clears `self.test_outputs`). It is still abstract: subclasses
must supply `test_step` and `predict_step`.

### 4.3 `PowerFlowTask` (`pf_task.py`, `@TASK_REGISTRY.register("PowerFlow")`)

- **Predicts** (via `PhysicsDecoderPF`, see §6.3): `[Vm, Va, Pg, Qg]` per bus.
  `Vm`/`Va` come directly from the model at masked positions (clamped to
  ground truth elsewhere); `Qg` at PV/REF buses and `Pg` at the REF bus are
  computed analytically from the nodal reactive/active power balance
  (`Qg = Q_in + Qd − q_shunt`, `Pg_ref = P_in + Pd − p_shunt`); `Pg` at PV
  buses and PQ-bus `Pg`/`Qg` are `0`.
- **Masks** (`AddPFHeteroMask`, `datasets/masking.py`): PQ buses mask
  `Vm, Va`; PV buses mask `Va, Qg`; REF buses mask `Vm, Qg`; generators at
  REF buses mask `Pg`. Static feature columns (`min_vm_pu`, `max_vm_pu`,
  `min_q_mvar`, `max_q_mvar`, `vn_kv`, `min_p_mw`, `max_p_mw`, cost
  coefficients) are **always masked** (fed as zeros, i.e. treated as
  "unknown to reconstruct" for self-supervised purposes) regardless of bus
  type. Branch `P_E, Q_E, ANG_MIN, ANG_MAX, RATE_A` are always masked too.
  If `data.mask_type: rnd`, `AddRandomHeteroMask` is used instead (uniform
  i.i.d. masking at `data.mask_ratio` over `PD, QD, VM, VA, QG` / `PG` /
  `P_E, Q_E`, for self-supervised pretraining rather than PF-consistent
  masking).
- **`test_step` metrics**: per-bus-type (`PQ`/`PV`/`REF`) MSE→RMSE for
  `Pg, Qg, Vm, Va`; active/reactive power-balance residuals
  (`ComputeNodeResiduals`); Power Balance Error (PBE) mean and max
  (`sqrt(P_residual² + Q_residual²)`, mean-pooled per graph then averaged,
  vs. global max). Verbose mode (`args.verbose`) additionally computes
  per-bus-type mean/max residuals and gathers full predictions/targets for
  plotting; in DDP this is gathered to rank 0 via `dist.gather_object`.
- **Output artifacts** (`on_test_end`, rank 0 only): `{dataset}_RMSE.csv`
  (RMSE by bus-type × quantity) and `{dataset}_metrics.csv` (avg active/
  reactive residual, PBE mean/max) under `.../artifacts/test/`. If verbose:
  `mean_residual_P.png` / `_Q.png` (mean/max histograms by bus type) and
  `{dataset}_correlation_{PQ,PV,REF}.png` (predicted-vs-target scatter per
  quantity) under `.../artifacts/test_plots/{dataset}/`.
- **`predict_step`** returns a flat dict (parquet-serializable via
  `_predictions_to_dataframe`) with per-bus scenario id, local bus index,
  `Pd, Qd, Vm_min/max, Qg_min/max`, targets, bus-type one-hot flags,
  predictions, and residuals — saved as `<grid>_predictions.parquet`.

### 4.4 `OptimalPowerFlowTask` (`opf_task.py`, `@TASK_REGISTRY.register("OptimalPowerFlow")`)

- **Predicts** (via `PhysicsDecoderOPF`): `[Vm, Va, Pg, Qg]` where `Vm, Va`
  come from the model everywhere (all buses are "unknown" in OPF — the
  model must find the optimal setpoint, not just reconstruct known values),
  `Pg` is the aggregated generator prediction, and `Qg` at PV/REF is derived
  analytically the same way as PF; PQ-bus `Qg` is `0`.
- **Masks** (`AddOPFHeteroMask`): `Vm, Va` masked at PQ and PV buses (PV
  `Vm` is masked too, unlike PF, since OPF must *choose* the voltage
  setpoint); `Qg` masked at PV/REF; `Vm` masked at REF; **all** generator
  `Pg` masked (`mask_gen[:, PG_H] = True` unconditionally); branch `P_E, Q_E`
  masked.
- **`GNS_heterogeneous`-specific behavior**: when `task.task_name ==
  "OptimalPowerFlow"`, the model additionally squashes `Vm` and `Pg`
  predictions into `[min, max]` via `bound_with_sigmoid` (a sigmoid map, not
  a hard clip) before computing branch flows — this is model-internal, not
  task-level, and does **not** apply to `GRIT` (see §6.2).
- **`test_step` metrics**: everything PF computes, plus: `Opt gap` (mean
  `|cost_pred − cost_gt| / cost_gt × 100`, generator cost polynomial
  evaluated at predicted vs. target `Pg`, **assumes all branches are
  in-service** per an inline code comment); `MSE PG`; branch thermal
  violations (`relu(|S| − rate_a)`, forward/reverse split by edge-index
  half, since `("bus","connects","bus")` stores each branch twice); branch
  voltage-angle-limit violations (wrapped to `[−π, π]`); `Qg` limit
  violations at PV/REF buses (`relu` over/under amounts and boolean mask).
- **Output artifacts**: same `{dataset}_RMSE.csv` as PF, plus
  `{dataset}_metrics.csv` with `RMSE PG generators`, `Mean optimality gap
  (%)`, thermal/angle/Qg violation means. Verbose mode adds
  `{dataset}_objective.png` (predicted vs. ground-truth cost scatter with
  R), plus the same residual histograms and correlation plots as PF (with
  `Qg` violations highlighted in red).
- **`predict_step`** returns a **nested dict** `{"bus": {...}, "gen": {...}}`
  (unlike PF's flat dict) — `cli.py`'s `_predictions_to_dataframe` detects
  this (`any(isinstance(value, dict) ...)`) and writes two parquet files:
  `<grid>_predictions.parquet` (bus table) and
  `<grid>_gen_predictions.parquet` (gen table, with `min_p_mw`, `max_p_mw`,
  and cost coefficients per generator).

### 4.5 `StateEstimationTask` (`se_task.py`, `@TASK_REGISTRY.register("StateEstimation")`)

- **Predicts** (via `PhysicsDecoderSE`): `[Vm, Va, P_in − p_shunt, Q_in −
  q_shunt]` — i.e. voltage directly from the model, and net nodal power
  injection derived from the predicted voltages via the analytic branch-flow
  layer (no clamping to ground truth anywhere — SE has no "known" quantities
  in the PF/OPF sense, only noisy/missing measurements; see §10).
- **Masking**: uses `SimulateMeasurements` + `ApplyMasking` (not
  `AddPFHeteroMask`/`AddOPFHeteroMask`) — see §10 for the full mechanism.
- **`test_step`**: builds `target = [Vm, Va, Pg_agg − Pd, Qg − Qd]` (true net
  injection) and `measurements = [Vm_meas, Va_meas, Pg_meas_agg − Pd_meas,
  Qg_meas − Qd_meas]` (as-measured, i.e. noisy/possibly-masked-to-zero
  input). Three disjoint bus subsets are tracked: `outliers_bus` (measurement
  was corrupted by an injected outlier), `mask_bus` (measurement was dropped
  entirely — the model never saw it), `non_outliers_bus` (clean measurement).
  Only stores outputs for plotting (`self.test_outputs`); logs no scalar
  metrics at all (no `self.log(...)` calls in `test_step` — this is the only
  task that doesn't log per-batch test metrics).
- **`on_test_end`** (rank 0, non-DDP-gathered — unlike PF/OPF this one has
  **no** `dist.gather_object` call, so in multi-GPU DDP the SE test plots
  only reflect rank 0's shard of the test set, not the full test set — a
  real inconsistency vs. the PF/OPF tasks): three families of correlation
  scatter plots (`pred_vs_target`, `pred_vs_measured`, `measured_vs_target`),
  each split by the three subsets above, under
  `.../artifacts/test_plots/{dataset}/`. No CSV metrics files are written at
  all (contrast with PF/OPF's `_RMSE.csv`/`_metrics.csv`).
- **`predict_step`** is a no-op stub (`pass`, returns `None`) — **`gridfm_graphkit
  predict` and `evaluate --save_output` are non-functional for
  `StateEstimation`**: `_predictions_to_dataframe` would receive a list of
  `None`s and crash on `predictions[0].keys()`. This is marked `# TODO: add
  custom test and predict steps` in the source.

### 4.6 AC/DC ground-truth baselines (`pf_ac_dc_baseline.py`, `opf_ac_dc_baseline.py`)

Triggered by `--compute_dc_ac_metrics`, run once after the test loop on rank
0 only. These do **not** touch the trained model at all — they load the raw
parquet test-split rows (scenario IDs read back from
`.../artifacts/stats/{grid_name}_scenario_splits.json`, itself written by
`LitGridHeteroDataModule.save_scenario_splits` during `setup()`) and compute
AC/DC ground-truth power-balance residuals via
`gridfm_datakit.utils.power_balance.compute_bus_balance` — i.e. a sanity
baseline: "how much does the underlying dataset's own AC vs DC solution
disagree" independent of any model prediction. If no splits JSON exists yet
(e.g. `setup()` wasn't reached, or logger isn't `MLFlowLogger`), the function
prints a message and returns `False` without raising. OPF's version adds
optimality-gap (DC vs AC dispatch cost), branch thermal/angle violations, and
Pg/Qg bound-violation statistics computed straight from parquet columns.
Runtime stats divide the stored per-scenario solve time by a hardcoded
`NUM_PROCESSES = 64` constant (`pf_ac_dc_baseline.py`) — **this assumes the
dataset was generated with exactly 64 worker processes**; if it wasn't, the
reported "ms with 64 cores" figures are meaningless for that dataset (no
validation or config surface for the actual generation process count).

### 4.7 `tasks/utils.py` — shared plotting helpers

`residual_stats_by_type` (per-graph mean/max via `torch_scatter`, with an
explicit CPU fallback for `scatter_max` on MPS — a known
torch_scatter/MPS gap), `plot_residuals_histograms`, `plot_correlation_by_node_type`
(shared scatter-plot helper used by PF/OPF/SE, supports highlighting `Qg`
violations in red for OPF).

---

## 5. Models (`gridfm_graphkit/models/`)

`MODELS_REGISTRY` (`io/registries.py`) has exactly **two** entries, found by
grepping for `@MODELS_REGISTRY.register(...)`:

| Registry key | Class | File |
| --- | --- | --- |
| `GNS_heterogeneous` | `GNS_heterogeneous` | `gnn_heterogeneous_gns.py` |
| `GRIT` | `GritHeteroAdapter` | `grit_transformer.py` |

**"HGNS" is not a third model.** Every `examples/config/HGNS_*.yaml` file
sets `model.type: GNS_heterogeneous` — "HGNS" is only a filename convention
in this repo (short for "heterogeneous GNS"), not a distinct registered
architecture. Anyone looking for an `"HGNS"` registry key or class will not
find one.

### 5.1 `GNS_heterogeneous` (`models/gnn_heterogeneous_gns.py`)

A heterogeneous transformer-style message-passing GNN with an **analytic
physics decoder baked into every layer's forward pass** (not just the loss):

- Separate input projections (`Linear → LeakyReLU → Linear → LayerNorm`) for
  bus features (`input_bus_dim`), gen features (`input_gen_dim`), and edge
  features (`edge_dim`), all into `hidden_size`.
- `num_layers` stacked `HeteroConv` blocks, each a dict of
  `torch_geometric.nn.TransformerConv` (with `beta=True` gating) over the
  three relations `("bus","connects","bus")`, `("gen","connected_to","bus")`,
  `("bus","connected_to","gen")` — bus-bus convolution uses edge attributes;
  the gen-bus relations do not. Multi-head attention (`attention_head`
  heads); output width per node type after each layer is `hidden_size *
  attention_head`, with a residual/skip connection only when shapes match
  (i.e. from layer 2 onward).
- After every layer, `mlp_bus`/`mlp_gen` decode a `[Vm, Va]` / `[Pg]`
  estimate, which is passed through the **task-specific physics decoder**
  (`PhysicsDecoderPF`/`OPF`/`SE`, `PHYSICS_DECODER_REGISTRY`, `models/utils.py`
  — selected by `task.task_name` again, a third use of that config key) to
  derive `Pg`/`Qg` analytically from the branch-flow/injection physics
  (`ComputeBranchFlow` → `ComputeNodeInjection` → decoder). The resulting
  bus-power residual is **fed back into the bus hidden state** via a small
  `physics_mlp` (`Linear(2, hidden_size*heads) → LeakyReLU`) as a per-layer
  correction — this residual-injection loop is what `LayeredWeightedPhysics`
  loss (§13.1) supervises via `model.layer_residuals[i]`, one entry per
  layer, populated fresh on every forward call.
- `StateEstimation` special-case: `mlp_gen`, `physics_mlp`, the last layer's
  gen-branch `norms_gen`, and the last layer's `("bus","connected_to","gen")`
  conv have `requires_grad = False` — those code paths genuinely never
  contribute to the SE loss (SE uses `PhysicsDecoderSE`, which doesn't use
  `agg_bus`/generator predictions at all), and freezing them avoids DDP's
  unused-parameter error while keeping the modules present so PF/OPF-trained
  checkpoints still load into an SE model (and vice versa) without key
  mismatches.
- `task.task_name == "OptimalPowerFlow"` additionally squashes `Vm`, `Pg`
  through `bound_with_sigmoid(pred, min, max)` (a sigmoid-based soft clamp,
  **not** a hard `torch.clamp`) using the bus's `min_vm_pu`/`max_vm_pu` and
  the generator's `min_p_mw`/`max_p_mw` feature columns.

Required config keys: `num_layers, hidden_size, input_bus_dim, input_gen_dim,
output_bus_dim, output_gen_dim, edge_dim, attention_head`; optional
`dropout` (default `0.0`, `getattr`).

### 5.2 `GritHeteroAdapter` / `GritTransformer` (`grit_transformer.py`, `grit_layer.py`)

Adapts the homogeneous GRIT architecture (Ma et al., *Graph Inductive Biases
in Transformers without Message Passing*, 2023) to the heterogeneous
power-grid graph. **Unlike `GNS_heterogeneous`, GRIT has no analytic physics
decoder anywhere in its forward pass** — it is a pure learned transformer
whose only physics awareness comes from the `PBE`/`MaskedReconstructionMSE`
loss terms at training time, not from any architectural inductive bias
beyond the graph structure and (optional) RRWP/RWSE positional encodings.

- **Bus-only subgraph extraction**: generator `Pg` is aggregated onto buses
  via `scatter_add` (`aggregate_pg`, masked generators excluded from the
  sum; buses with *all* generators masked get the model's mask-value
  sentinel instead of `0`, to preserve an "unknown" signal distinct from
  "known zero"), concatenated onto the bus feature vector (e.g. 15 → 16
  dims), and the resulting homogeneous `Data` graph (bus nodes,
  `("bus","connects","bus")` edges only — generators are otherwise invisible
  to the transformer) is run through `FeatureEncoder` → optional
  `RRWPLinearNodeEncoder`/`RRWPLinearEdgeEncoder` → `num_layers` ×
  `GritTransformerLayer` (sparse multi-head attention with learned edge
  bias, degree-scaled aggregation, optional LayerNorm/BatchNorm, ReZero).
- **Output heads are separate from the transformer**: `bus_head` (a small
  MLP off the final transformer node embedding, width `output_bus_dim`) and
  an *optional* `gen_head` (MLP directly off raw, unmasked-by-transformer
  `batch["gen"].x`, width `output_gen_dim`). **If `output_gen_dim` is `0` or
  unset, `gen_head` is `None` and the "gen prediction" returned is literally
  the untouched (masked, per `ApplyMasking`) generator input tensor** — not
  a model prediction at all. Every `GRIT_PF*.yaml` example sets
  `output_gen_dim: 0` for exactly this reason (PG is predicted at the bus
  level via `output_bus_dim: 6 = [VM, VA, PG, QG, PD, QD]` instead). Using a
  per-generator loss like `MaskedGenMSE` against this "gen" output would be
  silently comparing masked input to masked input — no example config does
  this, but nothing in the code prevents it.
- Config keys largely mirror `GNS_heterogeneous`'s hetero-head dims
  (`input_bus_dim, input_gen_dim, output_bus_dim, output_gen_dim, edge_dim,
  hidden_size, attention_head`) plus GRIT-specific `dropout, num_layers, act,
  gt.*` (transformer-layer hyperparameters: `layer_norm, batch_norm,
  update_e, attn_dropout, attn.{clamp,act,full_attn,edge_enhance,O_e,norm_e,
  signed_sqrt,bn_momentum,bn_no_runner}`) and `encoder.*` (`node_encoder,
  node_encoder_name, node_encoder_bn, edge_encoder, edge_encoder_bn`).
  `GritHeteroAdapter.__init__` auto-derives `model.input_dim` from
  `input_bus_dim` and `model.output_dim` from `output_bus_dim` if absent,
  and auto-syncs `model.gt.dim_hidden` from `model.hidden_size` and
  `model.encoder.posenc_RWSE.kernel.times` from `data.posenc_RWSE.kernel.times`
  — deep-copies `args` first so multiple models built from the same shared
  config namespace don't mutate each other.
- **`GritTransformer.__init__`** asserts
  `model.hidden_size == model.gt.dim_hidden == model.input_dim` (post
  feature-encoding) — a real, deliberate constraint (not a bug), enforced
  with a plain `assert`.

---

## 6. Datasets (`gridfm_graphkit/datasets/`)

### 6.1 On-disk processing (`powergrid_hetero_dataset.py`)

`HeteroGridDatasetDisk` is a `torch_geometric.data.Dataset` over
`{root}/raw/{bus,gen,branch}_data.parquet`. `process()` runs once (guarded by
a `processed_raw_files.done` sentinel file) and, per scenario, writes one
`data_index_{scenario}.pt` file to `{root}/processed/` containing a
`HeteroData.to_dict()`. Feature columns are hardcoded lists (`bus_features`,
`gen_features`, `forward_branch_features`, `reverse_branch_features`) — see
§7 for the exact index mapping.

Key invariants **asserted, not just assumed**:
- `bus_data["scenario"]` must be `0..N-1` contiguous (`process()`'s first
  assert) — a dataset with gaps in scenario IDs will fail hard, not
  silently skip.
- Buses within a scenario must be indexed `0..n_buses-1` in increasing order
  (asserted per-scenario during processing) — this ordering assumption is
  relied on later by `predict_step`'s `local_bus_idx` reconstruction via
  `torch.arange`, with an inline `# todo: we should remove this assert` note
  acknowledging the fragility.
- If `load_scenario_idx` is present in `bus_data`, a `load_scenarios.pt`
  tensor is written (first `load_scenario_idx` per scenario, groupby-sorted)
  — this is what `split_by_load_scenario_idx` requires; **absent this
  column, that split mode raises `ValueError` at datamodule setup, not at
  processing time.**
- Generator reactive limits (`min_q_mvar`, `max_q_mvar`) are pre-aggregated
  per-bus (`groupby(["scenario","bus"]).sum()`, left-joined, NaN→0) into the
  bus table **before** any graph is built — this happens for every scenario
  read, not cached, so it re-runs on every `HeteroGridDatasetDisk`
  construction (not just the first).
- `data["gen","connected_to","bus"]` and its reverse are built from
  `gen_index` = the generator's row-order position *within that scenario*
  (`gen_df.reset_index()`), not any persistent generator ID from the source
  data — generator identity is not stable/traceable across scenarios beyond
  positional order.

`get(idx)` loads the `.pt` file, rebuilds `HeteroData`, and calls
`self.data_normalizer.transform(data)` — **normalization happens at
`__getitem__` time, every access, not once at process time**; the normalizer
itself must already be fitted externally (raises `ValueError("BaseMVA not
properly set")` otherwise).

### 6.2 Transform pipeline (`task_transforms.py`, `masking.py`, `transforms.py`)

`TRANSFORM_REGISTRY` composes a `torch_geometric.transforms.Compose` per
task, applied at `__getitem__` time (after normalization, since `transform`
is passed to `HeteroGridDatasetDisk`, and PyG applies `transform` after
`get()`):

- **`PowerFlowTransforms`** / **`OptimalPowerFlowTransforms`**:
  `RemoveInactiveBranches` → `RemoveInactiveGenerators` →
  (`AddRandomHeteroMask` if `mask_type: rnd` else `AddPFHeteroMask`/
  `AddOPFHeteroMask`) → `ApplyMasking`.
- **`StateEstimationTransforms`**: (`LoadGridParamsFromPath` if
  `task.grid_path` set) → `RemoveInactiveBranches` →
  `RemoveInactiveGenerators` → `SimulateMeasurements` → `ApplyMasking`.

`RemoveInactiveGenerators`/`RemoveInactiveBranches` filter on the raw
`G_ON`/`B_ON` flag columns (index 6 and 10 respectively — the *last* column
of the raw feature vectors) and then **drop that column entirely** from
`x`/`edge_attr` (`x[:, :G_ON]`, `edge_attr[:, :B_ON]`) — this is why
`input_gen_dim=6` (not 7) and `edge_dim=10` (not 11) in every example
config: the on/off flag is consumed structurally (by filtering) rather than
passed to the model as a feature. `ApplyMasking` writes `data.mask_value`
(config `data.mask_value`) into every masked slot of `x_dict["bus"]`,
`x_dict["gen"]`, and `edge_attr_dict[("bus","connects","bus")]` — this is an
in-place overwrite of the *input*, meaning the model literally never sees
the true value of a masked feature (as opposed to only being told not to
supervise on it).

### 6.3 Normalization (`normalizers.py`)

Both normalizers implement a common per-unit (p.u.) scheme keyed by a fitted
`baseMVA` (95th percentile of non-zero `{Pd, Qd, Pg, Qg}` values over the
fitting scenario set) and `vn_kv_max` (max nominal voltage). Power
quantities divide by `baseMVA`; voltage angle converts degrees↔radians;
admittances/shunts scale by `baseMVA_orig / baseMVA`; generator cost
coefficients go through a signed `log1p`/`expm1` transform
(`sign(x) * log1p(|x|)`) rather than linear scaling, since cost coefficients
can span many orders of magnitude.

- **`HeteroDataMVANormalizer`** (`fit_strategy = "fit_on_train"`): one global
  `baseMVA`/`vn_kv_max` for the whole dataset, fit once from the **training
  split only** (`data_normalizer.fit(raw_data_path, train_scenario_ids)`).
  This is the one restorable from `--normalizer_stats` (checked via
  `fit_strategy == "fit_on_train"` in `LitGridHeteroDataModule.setup`).
- **`HeteroDataPerSampleMVANormalizer`** (`fit_strategy = "fit_on_dataset"`):
  a per-scenario `baseMVA`/`vn_kv_max` lookup table, fit over **all** splits
  combined (`train + val + test` scenario IDs — `assert
  np.unique(all_ids).shape[0] == num_scenarios`), indexed by `scenario_id`
  at both train and inference time. **`--normalizer_stats` explicitly does
  not apply to this normalizer** (README documents this correctly): it
  always recomputes from whatever scenarios are present in the current run,
  because a per-sample scale by definition depends on which specific
  scenarios are loaded.

Both raise `ValueError` from `inverse_transform` if `data.is_normalized` is
false, and from `inverse_transform`/`transform` if the normalizer's own
`baseMVA` doesn't match the batch's stored `baseMVA` tensor —
cross-normalizer-instance batch reuse fails loudly rather than silently
producing wrong numbers.

**Documented `WARNING` in the source** (both normalizers, verbatim in
code comments): admittance/shunt values (`GS`, `BS`, `Yff/Yft/Ytf/Ytt`) are
**not** restored to their original per-unit values by `inverse_transform` —
the forward transform scales by `baseMVA_orig/baseMVA` but the inverse
multiplies by `baseMVA` (not `baseMVA/baseMVA_orig`), landing in
"`original * baseMVA_orig`" physical SI-like units instead. This is stated
to be intentional (the physics layers `ComputeBranchFlow` etc. expect this
convention), but it means "denormalized" admittance values are **not**
directly comparable to the source parquet's raw `r`/`x`/`Yff` columns.

### 6.4 Feature index constants (`datasets/globals.py`)

Fixed column-index constants shared across the whole codebase (not derived
from config), reproduced here since they're load-bearing for anyone writing
a custom loss/task/model:

- **Bus `x`** (15 cols post-transform): `PD_H=0, QD_H=1, QG_H=2, VM_H=3,
  VA_H=4, PQ_H=5, PV_H=6, REF_H=7, MIN_VM_H=8, MAX_VM_H=9, MIN_QG_H=10,
  MAX_QG_H=11, GS=12, BS=13, VN_KV=14`. Bus `y` = first 5 columns
  (`PD..VA`).
- **Bus output** (model prediction columns): `VM_OUT=0, VA_OUT=1, PG_OUT=2,
  QG_OUT=3, PD_OUT=4, QD_OUT=5` (the last two only populated for
  `output_bus_dim ≥ 6`, i.e. GRIT-style reconstruction heads;
  `PG_OUT_GEN=0` is a separate constant for the *generator*-head output).
- **Gen `x`** (6 cols post-transform): `PG_H=0, MIN_PG=1, MAX_PG=2, C0_H=3,
  C1_H=4, C2_H=5` (raw 7th column `G_ON=6` is stripped, §6.2). Gen `y` = 1
  column (`PG`).
- **Edge `attr`** (10 cols post-transform): `P_E=0, Q_E=1, YFF_TT_R=2,
  YFF_TT_I=3, YFT_TF_R=4, YFT_TF_I=5, TAP=6, ANG_MIN=7, ANG_MAX=8,
  RATE_A=9` (raw 11th column `B_ON=10` stripped).

### 6.5 Positional encodings (GRIT only) (`posenc_stats.py`, `rrwp.py`, `cached_transform.py`)

Two PE types, each independently toggled via `data.posenc_{RRWP,RWSE}.enable`:

- **RWSE** (Random Walk Structural Encoding): `get_rw_landing_probs`
  computes, per node, the diagonal of `P^k` (`P = D⁻¹A`, dense
  matrix-power) for `k = 1..data.posenc_RWSE.kernel.times`, stored as
  `pestat_RWSE` on the bus store. Encoded by `RWSENodeEncoder` (in
  `models/kernel_pos_encoder.py`) which shrinks the linear node-feature
  projection by `pe_dim` and concatenates the encoded RW landing
  probabilities. **Dense `to_dense_adj`/matrix-power over the bus subgraph
  — O(N²) memory and O(N³·k) compute per graph** — not viable for very
  large grids (e.g. case2000/caseTexas) without RWSE disabled; no
  automatic guard against this exists.
- **RRWP** (Relative Random Walk Probabilities, the GRIT paper's own
  encoding): `add_full_rrwp` (dense, all-pairs) or, if
  `data.posenc_RRWP.topk > 0`, `add_topk_rrwp` (keeps only each node's
  top-K structurally-nearest neighbors by RRWP L2 norm, plus original graph
  edges and self-loops — a genuine sparsification knob, documented well
  in-line in `GRIT_PF_datakit_case14.yaml`'s comments). Encoded via
  `RRWPLinearNodeEncoder` (absolute, added into node features) and
  `RRWPLinearEdgeEncoder` (relative, merged with edge attributes; can pad
  to a fully-connected graph if `model.gt.attn.full_attn: true`, using
  `torch_sparse.coalesce`).
- **Disk caching** (`CachedPosencTransform`, `cached_transform.py`):
  optional per-PE-type (`posenc_*.cache: true`), keyed by a SHA-256 hash of
  graph topology (edge_index + node count, optionally quantized admittance
  weights) — **not** by scenario ID by default (`key_attr="topology"`), so
  all scenarios sharing the same topology (e.g. no topology perturbation in
  the source dataset) reuse a single cached file. Cache directory name
  encodes PE hyperparameters (`pe_cache_rwse_k21`, `pe_cache_rrwp_k21_topk5`)
  so changing `kernel.times`/`ksteps`/`topk` invalidates automatically.
  Writes are atomic (`tempfile.mkstemp` + `os.replace`) for multi-worker/
  multi-job safety.

### 6.6 Splitting (`datasets/utils.py`, `hetero_powergrid_datamodule.py`)

Three mutually-exclusive strategies, selected in `LitGridHeteroDataModule.setup`:

1. **Default (`split_dataset`)**: after subsetting to `data.scenarios[i]`
   scenarios (`random.seed(args.seed)` + `random.shuffle` over all dataset
   indices, then `[:num_scenarios]`), a **second**, independent
   `np.random.seed(args.seed)` + `np.random.permutation` splits that subset
   into train/val/test by `val_ratio`/`test_ratio` fractions (`val_size =
   int(val_ratio*N)`, `test_size = int(test_ratio*N)`, remainder to train).
   Reusing the same `seed` value for two different RNGs (`random` and
   `numpy.random`) is deliberate for reproducibility but means the two
   shuffles are *not* independent samples in a cryptographic sense — fine
   for ML reproducibility, just worth knowing if auditing randomness.
2. **`split_by_load_scenario_idx: true`** (`split_dataset_by_load_scenario_idx`):
   splits by **unique `load_scenario_idx` values**, not by individual
   dataset rows — i.e. if a dataset has multiple *topology* variants of the
   same underlying load scenario (as `gridfm-datakit`'s topology
   perturbation produces), all variants of a given load scenario land in
   the same split, preventing the model from seeing near-duplicate
   topologies of the same load across train and test. Requires
   `load_scenario_idx` to exist in the raw bus data (§6.1) or raises
   `ValueError` with an explicit, actionable message.
3. **`split_from_existing_files`**: loads `train.pt`/`val.pt`/`test.pt`
   (lists/tensors of dataset indices) from a folder — `data.scenarios` is
   **ignored** for split purposes (warned), and `num_scenarios` is
   recomputed as the union of all three splits' unique scenario IDs.

In all cases, scenario IDs used for each split are saved to
`.../artifacts/stats/{network}_scenario_splits.json` (rank 0 only, only if
`trainer.logger` is set) — this is the file consumed by
`--compute_dc_ac_metrics` (§4.6) to know which scenarios are "test."

In distributed (DDP) settings, dataset processing and normalizer fitting
happen on rank 0 only, with `dist.barrier()` synchronization and
`dist.broadcast_object_list` to distribute fitted normalizer stats to other
ranks — other ranks never read the raw parquet directly for fitting.

---

## 7. Training infrastructure (`gridfm_graphkit/training/`)

### 7.1 Losses (`loss.py`)

`LOSS_REGISTRY` entries (all subclass `BaseLoss`/`nn.Module`, all take
`(loss_args, args)` in `__init__` and
`(pred, target, edge_index, edge_attr, mask, model, x_dict)` in `forward`,
returning a dict with a `"loss"` key plus named metrics):

| Registry key | What it computes |
| --- | --- |
| `MaskedMSE` | Plain MSE over `pred[mask]` vs `target[mask]` (homogeneous-style, generic). |
| `MaskedGenMSE` | MSE over generator `Pg` restricted to `mask_dict["gen"][:, :PG_H+1]`. |
| `MaskedBusMSE` | MSE over bus `[Vm, Va]` (or `[Vm, Va, Qg]` if `self.args.task == "OptimalPowerFlow"`) — **see the bug note below**. |
| `MaskedReconstructionMSE` | Unified MSE over `[Vm, Va, Pg_agg, Qg, Pd, Qd]` (requires `output_bus_dim ≥ 6`); this is the loss GRIT configs use in place of `MaskedGenMSE + MaskedBusMSE`. |
| `MSE` | Plain unmasked MSE. |
| `LayeredWeightedPhysics` | Geometrically-weighted sum of `model.layer_residuals[i]` (per-layer power-balance residual norms — `GNS_heterogeneous`-only; requires the model to populate this dict, i.e. incompatible with `GRIT`). Weight for the last layer is highest (`base_weight^0`), decaying by `base_weight^(L-idx-1)` toward earlier layers, then normalized to sum to 1. |
| `LossPerDim` | MAE or MSE (`loss_args.loss_str`) on exactly one of `VM, VA, P_in, Q_in` (`loss_args.dim`) — `P_in`/`Q_in` are net nodal injections re-derived from generator/load targets, used by the `SE_simulate_measurements.yaml` test config (4 `LossPerDim` instances, one per dim). |
| `PBE` | Power Balance Equation loss: reconstructs the full complex Y-bus (off-diagonal from `Yft/Ytf` edge columns, diagonal by scatter-summing `Yff/Ytt` per source bus + bus shunt `Gs+jBs`) as a sparse COO tensor, computes `S_injection = V ⊙ conj(Y_bus) @ conj(V)`, compares against net `S = (Pg−Pd) + j(Qg−Qd)` (masked positions clamped to prediction, unmasked to ground truth) via mean absolute complex error. Optional `loss_args.visualization` adds per-node residual tensors to the output dict (not just scalars) for downstream plotting. |
| `QgViolationPenalty` | `relu(Qg_pred − Qg_max) + relu(Qg_min − Qg_pred)`, mean over violating entries only (NaN from an empty mask is replaced with `0`). |

`MixedLoss` (not registry-registered, constructed directly by
`get_loss_function`) is a weighted sum: `total = Σ weights[i] * loss_i`,
with all named sub-metrics from each component merged into one flat dict
(later-registered losses' metric names silently overwrite earlier ones on
key collision — no namespacing).

**Confirmed bug**: `MaskedBusMSE.forward` (line 143 of `loss.py`) checks
`if self.args.task == "OptimalPowerFlow":`, but `self.args` is the **full**
top-level `NestedNamespace` (passed in by `get_loss_function`), so
`self.args.task` is itself a `NestedNamespace` (with a `.task_name`
attribute) — comparing a `NestedNamespace` object to the string
`"OptimalPowerFlow"` with `==` is always `False` (verified: `NestedNamespace.__eq__`
inherits `argparse.Namespace`'s dict-comparison, which returns
`NotImplemented` against a plain string, and Python then falls back to
identity comparison → `False`). **The intended comparison is
`self.args.task.task_name == "OptimalPowerFlow"`.** As written,
`MaskedBusMSE` *always* takes the PF branch (`[Vm, Va]` only, never `Qg`),
even in every `HGNS_OPF_*.yaml` example config, which lists `MaskedBusMSE`
as one of its four loss terms. `Qg` is not left completely unsupervised in
those configs (`QgViolationPenalty` penalizes limit violations, and the PBE
component of the physics decoder loop still shapes it indirectly), but the
direct `Qg`-reconstruction MSE term that the OPF branch of this loss is
clearly meant to add is silently never applied.

### 7.2 Callbacks (`callbacks.py`)

- `DEFAULT_MONITOR = "Validation loss"` — module-level constant, the
  fallback for all three `callbacks.*_monitor` config keys (§3.6).
- `EpochTimerCallback` — only instantiated when `--report-performance` is
  passed; records wall-clock time and batch count per epoch, exposes
  `last_epoch_time`/`last_epoch_iters_per_sec`.
- `SaveBestModelStateDict(monitor, mode="min", filename="best_model_state_dict.pt")`
  — on every `on_validation_end` (rank 0 only via `@rank_zero_only`), reads
  `trainer.callback_metrics[monitor]`; if it's the best so far, saves a
  **state-dict-only** checkpoint (compile-prefix-stripped via the same
  `model._orig_mod.` → `model.` rename as `_normalize_loaded_state_dict_keys`)
  to `.../artifacts/model/best_model_state_dict.pt`. Explicitly skips the
  Lightning sanity-check validation pass (`trainer.sanity_checking`); once
  real training has started, a missing/misnamed monitor key raises
  `KeyError` listing all available metrics (this is the failure mode
  referenced in §3.6).
- `ModelCheckpoint(save_last=True, save_top_k=0)` — always added by
  `get_training_callbacks` in `cli.py` alongside the two above. This is a
  **second, independent** checkpointing mechanism: a full Lightning
  checkpoint (weights + optimizer + scheduler + epoch state, not just
  weights) is written by Lightning's own machinery to
  `default_root_dir`/… (i.e. under `--log_dir`), separate from and not read
  by any other part of this codebase — only `SaveBestModelStateDict`'s
  `best_model_state_dict.pt` is what `--model_path`/`finetune`/`evaluate`/
  `predict` actually load.
- `EarlyStopping(monitor, min_delta=callbacks.tol, patience=callbacks.patience,
  mode="min", strict=True)` — from `lightning.pytorch.callbacks`, wired up
  in `cli.py`'s `get_training_callbacks`, not in `training/callbacks.py`
  itself.

---

## 8. IO / config handling (`gridfm_graphkit/io/`)

### 8.1 `registries.py`

A minimal `Registry` class: `register(name)` decorator (raises `KeyError` on
duplicate name), `get(name)` (raises `KeyError` if absent), `create(name,
*args, **kwargs)` (get + instantiate). Seven module-level registry
instances:

| Registry | Populated entries (grep-verified) |
| --- | --- |
| `MODELS_REGISTRY` | `GNS_heterogeneous`, `GRIT` |
| `TASK_REGISTRY` | `PowerFlow`, `OptimalPowerFlow`, `StateEstimation` |
| `TRANSFORM_REGISTRY` | `PowerFlow`, `OptimalPowerFlow`, `StateEstimation` |
| `PHYSICS_DECODER_REGISTRY` | `PowerFlow`, `OptimalPowerFlow`, `StateEstimation` |
| `LOSS_REGISTRY` | `MaskedMSE`, `MaskedGenMSE`, `MaskedBusMSE`, `MaskedReconstructionMSE`, `MSE`, `LayeredWeightedPhysics`, `LossPerDim`, `PBE`, `QgViolationPenalty` |
| `NORMALIZERS_REGISTRY` | `HeteroDataMVANormalizer`, `HeteroDataPerSampleMVANormalizer` |
| `DATASET_WRAPPER_REGISTRY` | **empty in this package** — no `@DATASET_WRAPPER_REGISTRY.register(...)` call exists anywhere in `gridfm_graphkit/`. It exists purely as an extension point for the `--plugins` mechanism (e.g. an internal `gridfm_graphkit_ee` package, referenced only as an example in every `--help` string, is presumably where `SharedMemoryCacheDataset` actually lives — it ships in neither this repo nor on PyPI as part of this package). Passing `--dataset_wrapper` without first loading a plugin that registers it raises `KeyError` from `_validate_dataset_wrapper` in `cli.py`, with the (empty) list of available wrappers in the message. |

### 8.2 `param_handler.py`

- `NestedNamespace` — see §3.
- `load_normalizer(args)` → `NORMALIZERS_REGISTRY.create(args.data.normalization, args)`.
- `get_loss_function(args)` → builds one `LOSS_REGISTRY` instance per
  `(loss_name, loss_args)` pair, wraps in `MixedLoss(loss_functions,
  weights=args.training.loss_weights)`.
- `load_model(args)` → `MODELS_REGISTRY.create(args.model.type, args)`.
- `get_task_transforms(args)` → `TRANSFORM_REGISTRY.create(args.task.task_name, args)`.
- `get_task(args, data_normalizers)` → `TASK_REGISTRY.create(args.task.task_name, args, data_normalizers)`.
- `get_physics_decoder(args)` → `PHYSICS_DECODER_REGISTRY.create(args.task.task_name)`.

Every `.create(...)` call wraps the registry's `KeyError` in a `ValueError`
with a clearer message (`f"Unknown {kind}: {name}"`) — consistent error
handling across all six lookup points.

---

## 9. Example config catalog (`examples/config/*.yaml`)

20 files, all `PowerFlow`/`OptimalPowerFlow` × `GNS_heterogeneous` except
one `GRIT_PF_datakit_case14.yaml`. **No `StateEstimation` example config
exists under `examples/config/` at all** — the only `StateEstimation`
config in the repository is `tests/config/SE_simulate_measurements.yaml`
(used by `tests/test_simulate_measurements.py`), not documented or
discoverable from the `examples/` tree a user would naturally look in.

| Axis | Values observed |
| --- | --- |
| Task | `PowerFlow` (`HGNS_PF_*`), `OptimalPowerFlow` (`HGNS_OPF_*`) |
| Model | `GNS_heterogeneous` (19 configs), `GRIT` (1 config) |
| Grid size | `case14_ieee`, `case30_ieee`, `case57_ieee`, `case118_ieee`, `case500_goc` (implied by filename `case500`), `case2000_goc`, `Texas2k_case1_2016summerpeak` |
| Split strategy | `HGNS_OPF_Ola_*` variants use `split_from_existing_files` (pointing at an absolute, environment-specific path — e.g. `/dccstor/gridfm/march_opf_exp/opfdata_olay_splits/` — these configs are **not directly runnable** outside the original authors' filesystem without editing that path) instead of `split_by_load_scenario_idx: true`. |
| Scale-driven overrides | `case2000`/`caseTexas` configs drop `workers` and `training.batch_size` (e.g. `batch_size: 4`, `workers: 16`, commented `# 4 gpus`) vs. `case14`'s `batch_size: 64, workers: 32, # 1 gpu` — a manual, per-grid-size tuning convention, not automatic. |

### 9.1 `HGNS_OPF_datakit_case14.yaml` (verbatim, representative OPF config)

```yaml
task:
  task_name: OptimalPowerFlow
data:
  baseMVA: 100
  mask_value: 0.0
  normalization: HeteroDataMVANormalizer
  networks:
  - case14_ieee
  scenarios:
  - 250000
  test_ratio: 0.1
  val_ratio: 0.1
  workers: 32 # 1 gpu
  split_by_load_scenario_idx: true
model:
  attention_head: 8
  edge_dim: 10
  hidden_size: 48
  input_bus_dim: 15
  input_gen_dim: 6
  output_bus_dim: 2
  output_gen_dim: 1
  num_layers: 12
  type: GNS_heterogeneous
optimizer:
  beta1: 0.9
  beta2: 0.999
  learning_rate: 0.0005
  lr_decay: 0.7
  lr_patience: 5
training:
  batch_size: 64 # 1 gpu
  epochs: 200
  loss_weights:
  - 0.1
  - 0.1
  - 0.75
  - 0.001
  losses:
  - LayeredWeightedPhysics
  - MaskedGenMSE
  - MaskedBusMSE
  - QgViolationPenalty
  loss_args:
  - base_weight: 0.5
  - {}
  - {}
  - {}
  accelerator: auto
  devices: auto
  strategy: auto
seed: 0
verbose: true
callbacks:
  patience: 100
  tol: 0
  early_stopping_monitor: Validation loss
  checkpoint_monitor: Validation loss
  lr_scheduler_monitor: Validation loss
```

### 9.2 `GRIT_PF_datakit_case14.yaml` (verbatim, only GRIT example — also the only example config using RRWP/RWSE)

```yaml
callbacks:
  patience: 100
  tol: 0
  early_stopping_monitor: Validation loss
  checkpoint_monitor: Validation loss
  lr_scheduler_monitor: Validation loss
task:
  task_name: PowerFlow
data:
  baseMVA: 100
  mask_type: rnd        # or determinstic
  mask_ratio: 0.5       # for random masking only
  mask_value: 0
  normalization: HeteroDataMVANormalizer
  networks:
  - case14_ieee
  scenarios:
  - 5000
  test_ratio: 0.1
  val_ratio: 0.1
  workers: 4
  posenc_RRWP:
    enable: false
    ksteps: 21
    cache: true
    topk: 5
  posenc_RWSE:
    enable: true
    cache: true
    kernel:
      times: 21
model:
  attention_head: 8
  dropout: 0.1
  edge_dim: 10
  hidden_size: 496
  input_dim: 16
  input_bus_dim: 16
  input_gen_dim: 6
  output_bus_dim: 6    # [VM, VA, PG, QG, PD, QD]
  output_gen_dim: 0    # PG predicted at bus level; no per-generator head needed
  num_layers: 7
  type: GRIT
  act: relu
  encoder:
    node_encoder: true
    edge_encoder: true
    node_encoder_name: RWSE
    node_encoder_bn: true
    edge_encoder_bn: true
    posenc_RWSE:
      pe_dim: 20
      raw_norm_type: batchnorm
  gt:
    layer_type: GritTransformer
    layer_norm: false
    batch_norm: true
    update_e: true
    attn_dropout: 0.2
    attn:
      clamp: 5.
      act: relu
      full_attn: false
      edge_enhance: true
      O_e: true
      norm_e: true
      signed_sqrt: true
      bn_momentum: 0.1
      bn_no_runner: false
optimizer:
  beta1: 0.9
  beta2: 0.999
  learning_rate: 0.0001
  lr_decay: 0.7
  lr_patience: 10
seed: 0
training:
  batch_size: 8
  epochs: 500
  loss_weights:
  - 0.99
  - 0.01
  losses:
  - PBE
  - MaskedReconstructionMSE
  loss_args:
  - {}
  - {}
  accelerator: auto
  devices: auto
  strategy: auto
verbose: true
```

Note `input_bus_dim: 16` here vs. `15` in the GNS configs above —
`GritHeteroAdapter` concatenates the aggregated `Pg` onto the bus feature
vector (§5.2), so the *bus store's* raw dim (15) plus 1 aggregated feature
= 16 is what actually reaches `FeatureEncoder`. `model.input_dim: 16` is set
redundantly here even though `GritHeteroAdapter.__init__` would auto-derive
it from `input_bus_dim` if omitted.

### 9.3 `tests/config/SE_simulate_measurements.yaml` (verbatim, the only `StateEstimation` config anywhere in the repo)

```yaml
callbacks:
  patience: 100
  tol: 0
  early_stopping_monitor: Validation loss
  checkpoint_monitor: Validation loss
  lr_scheduler_monitor: Validation loss
task:
  task_name: StateEstimation
  noise_type: Gaussian
  measurements:
    power_inj:
      mask_ratio: 0.2
      outlier_ratio: 0.1
      std: 0.02
    power_flow:
      mask_ratio: 0.2
      outlier_ratio: 0.1
      std: 0.02
    vm:
      mask_ratio: 0.2
      outlier_ratio: 0.1
      std: 0.02
  relative_measurement: true
data:
  baseMVA: 100
  mask_value: 0.0
  normalization: HeteroDataMVANormalizer
  networks:
  - case14_ieee
  scenarios:
  - 100
  split_by_load_scenario_idx: true
  test_ratio: 0.1
  val_ratio: 0.1
  workers: 1
model:
  attention_head: 8
  edge_dim: 10
  hidden_size: 48
  input_bus_dim: 15
  input_gen_dim: 6
  output_bus_dim: 2
  output_gen_dim: 1
  num_layers: 12
  type: GNS_heterogeneous
optimizer:
  beta1: 0.9
  beta2: 0.999
  learning_rate: 0.0005
  lr_decay: 0.7
  lr_patience: 5
seed: 0
training:
  batch_size: 32
  epochs: 1
  losses:
  - LossPerDim
  - LossPerDim
  - LossPerDim
  - LossPerDim
  loss_weights:
  - 0.35
  - 0.35
  - 0.1
  - 0.1
  loss_args:
  - dim: VM
    loss_str: MAE
  - dim: VA
    loss_str: MAE
  - dim: P_in
    loss_str: MAE
  - dim: Q_in
    loss_str: MAE
  accelerator: auto
  devices: auto
  strategy: auto
verbose: true
```

---

## 10. State estimation measurement simulation (`datasets/masking.py::SimulateMeasurements`)

This is the mechanism behind `StateEstimationTransforms`, and is worth
documenting precisely since it's a real simulated-measurement pipeline, not
a simple mask:

1. For each of three measurement families — `vm` (bus `VM_H` only),
   `power_inj` (bus `PD_H, QD_H, QG_H`), `power_flow` (branch `P_E, Q_E`) —
   `place_measurement_std_and_outliers` independently draws, **per bus/edge
   row** (not per feature):
   - `measurement_mask ~ Bernoulli(mask_ratio)` — this row's measurement is
     **dropped entirely** (treated as unavailable; std stays `inf`, marking
     it for `ApplyMasking` to overwrite with `mask_value` later).
   - `outliers_mask ~ Bernoulli(outlier_ratio)`, restricted to rows *not*
     already dropped (`logical_and(outliers_mask, ~measurement_mask)`) — this
     row gets a large, deliberate corruption.
   - For all other rows, `std` is set to the configured `measurement.std`
     (a **fraction**, since it's later multiplied by the true value under
     `relative_measurement: true`, or by `data.baseMVA` under `false`).
2. Generator active-power measurement noise (`std_gen`) is **not**
   independently configured — it's derived by broadcasting the bus-level
   `Pd` measurement std down to connected generators via
   `BusToGenBroadcaster` (a `MessagePassing` module dividing by
   `sqrt(degree)` per bus, i.e. assumes each connected generator shares an
   equal fraction of the bus's aggregate measurement uncertainty — a
   modeling simplification, not a separate configurable
   `measurements.pg`/generator-specific block).
3. `relative_measurement: true` (the example config's setting) scales `std`
   by `|true_value|` per-element (so a fixed 0.02 `std` means "2% of the
   true magnitude"); `false` scales by the global `data.baseMVA` instead
   (fixed-magnitude noise in physical units, independent of local value
   size).
4. **Noise** (`add_noise`): `Gaussian` (`~N(0, std)`), `Laplace`
   (`b = std/√2`, standard Laplace scaled by `b`), or `Uniform`
   (`~U(-1,1) · std·√3`, so the resulting distribution has standard
   deviation exactly `std` regardless of which of the three is chosen —
   moment-matched). Selected via `task.noise_type` (default `"Gaussian"`
   if omitted). Applied only to non-dropped rows (`torch.where(mask, ...,
   noisy)`).
5. **Outliers** (`add_outliers`): a fixed `±3·std` offset with a random
   sign, applied only to rows flagged as outliers (independent of the
   Gaussian/Laplace/Uniform noise already applied — outliers are an
   *additional* corruption on top of ordinary measurement noise, not an
   alternative to it).
6. `ApplyMasking` (run after `SimulateMeasurements` in the transform
   Compose) then overwrites the **dropped** rows' feature values with
   `data.mask_value` — so the final input tensor distinguishes three
   states per feature: normal (small noise), outlier (noise + large
   ±3σ jump), and dropped (hard-set to `mask_value`, `std=inf` in the
   mask/std tensors carried in `mask_dict`).
7. `mask_dict` (consumed by `se_task.py`'s `test_step`, §4.5) carries not
   just boolean masks but also the per-feature `std_bus`/`std_gen`/
   `std_branch` tensors and separate `outliers_bus`/`outliers_branch`
   boolean tensors — enough information to reconstruct exactly which rows
   were dropped vs. corrupted vs. clean, for the three-way plot split in
   `on_test_end`.

`grid_path` / `LoadGridParamsFromPath` (only invoked if `task.grid_path` is
set) exists to let SE experiments hold the **topology and generator
in-service status fixed** across all samples (loaded once from a saved
`HeteroData` template) while only load/generation/voltage vary per scenario
— useful for simulating repeated measurement snapshots of a single known
grid rather than a distribution of different grids/topologies.

---

## 11. Multi-GPU / distributed training

Distributed training is Lightning-native, not custom: `training.strategy:
ddp` (or `auto`, which resolves to `DDPStrategy(find_unused_parameters=False)`
whenever the accelerator is not `cpu`/`mps` — `cli.py`'s `main_cli`
explicitly special-cases `mps` to avoid DDP there even if `strategy: ddp` or
`auto` is set, since MPS doesn't support NCCL). `find_unused_parameters` is
hardcoded to `False` — any config/model combination that leaves a parameter
truly unused in the forward graph (not just frozen via `requires_grad =
False`, which DDP tolerates) will hang or error under DDP; this is why
`GNS_heterogeneous` explicitly freezes (rather than removes) the SE-unused
`mlp_gen`/`physics_mlp`/last-layer gen params, and why `GritHeteroAdapter`
skips building `gen_head`/`O_e`/`norm_e` modules entirely when they'd be
unused rather than leaving them present-but-unused.

LSF-cluster environment-variable derivation (`MASTER_ADDR`, `MASTER_PORT`,
`NODE_RANK`, `NCCL_SOCKET_IFNAME=ib,bond`, `NCCL_IB_CUDA_SUPPORT=1`) and
InfiniBand HCA-port exclusion (`fix_infiniband`, probes `ibv_devinfo` for
Ethernet-only ports and sets `NCCL_IB_HCA` to exclude them) are automatic
and unconditional whenever LSF env vars are detected — no opt-out flag
exists; on a non-LSF SLURM/bare-metal multi-node setup, none of this fires
and Lightning's own DDP environment detection takes over unassisted.

Rank-0-only guards recur throughout: dataset processing/normalizer fitting
(`hetero_powergrid_datamodule.py`), scenario-split JSON writing, all
`on_test_end` CSV/plot writing (PF/OPF gather other ranks' verbose outputs
via `dist.gather_object` first; **SE does not gather**, §4.5), AC/DC
baseline computation, and predictions-parquet saving. `--compute_dc_ac_metrics`
and `--save_output`/`predict` are explicitly gated to
`is_rank0` in `cli.py`.

---

## 12. Checkpointing, MLflow logging, resuming

- **Logger**: always `lightning.pytorch.loggers.MLFlowLogger(save_dir=
  args.log_dir, experiment_name=args.exp_name, run_name=args.run_name)` —
  no alternative logger backend is wired up anywhere (no TensorBoard, no
  W&B option). `save_dir` defaults to `mlruns` (a local file-based MLflow
  tracking store, not a remote tracking server — no `MLFLOW_TRACKING_URI`
  handling in this codebase).
- **Artifacts layout** (all under `{log_dir}/{experiment_id}/{run_id}/artifacts/`):
  `stats/normalization_stats.txt` + `normalizer_stats.pt` (§4.1),
  `stats/{network}_scenario_splits.json` (§6.6), `model/best_model_state_dict.pt`
  (§7.2), `test/{dataset}_RMSE.csv` + `{dataset}_metrics.csv` (+ AC/DC
  variants, §4.6), `test_plots/{dataset}/*.png` (verbose only),
  `test/<grid>_predictions.parquet` (`evaluate --save_output`).
- **Two independent checkpoint mechanisms** exist simultaneously (see §7.2):
  Lightning's own `last.ckpt` (full training state, under `--log_dir`, not
  MLflow-tracked) and `SaveBestModelStateDict`'s `best_model_state_dict.pt`
  (weights only, MLflow-artifact-tracked, compile-prefix-normalized). Only
  the second is what `--model_path` on `finetune`/`evaluate`/`predict`
  actually loads — **there is no "resume interrupted training from
  last.ckpt" CLI flag**; `train`/`finetune` always start a fresh
  `Trainer.fit` from epoch 0 with freshly-initialized (or, for `finetune`,
  `--model_path`-loaded) weights, never calling `trainer.fit(...,
  ckpt_path=...)`.
- `finetune` is architecturally just `train` with a mandatory
  `--model_path` state-dict preload (§2.2) — no separate learning-rate
  schedule, layer-freezing, or discriminative fine-tuning logic exists; it
  is a full re-training run starting from different initial weights, using
  whatever `optimizer`/`training` config is in the (possibly different)
  YAML passed to `finetune`.
- `--normalizer_stats` restoration (§6.3) is the mechanism for keeping
  normalization consistent between a training run and a later
  `evaluate`/`predict` run against **different** data (e.g. a different
  grid or scenario subset) — without it, `evaluate`/`predict` re-fit
  `fit_on_train`-type normalizers from whatever scenarios are visible in
  the current run, which will silently diverge from the training-time
  scale if the scenario set differs.

---

## 13. Test suite (`tests/`, `integrationtests/`)

### 13.1 Unit tests (`tests/`)

- `test_yaml_configs.py` — parametrized over every file in
  `examples/config/*.yaml` (20 configs); for each, asserts that
  `load_normalizer`, `get_task`, `load_model`, `get_loss_function`,
  `get_task_transforms`, `get_physics_decoder` all construct without
  raising. This is a **construction-only** smoke test — it does not run a
  forward pass, so a config that constructs successfully but produces
  shape-mismatched tensors at `forward()` time (e.g. a wrong `input_dim`)
  would not be caught here.
- `test_data_module.py`, `test_edge_flows.py`, `test_node_rank.py` — dataset/
  datamodule and physics-layer (`ComputeBranchFlow`/injection) correctness
  against `tests/data/case14_ieee/raw/*.parquet` (a small bundled fixture
  dataset, present in the repo under `tests/data/`).
- `test_losses.py` — loss registry construction/forward correctness.
- `test_simulate_measurements.py` — exercises `SimulateMeasurements`
  directly (mask/outlier/noise placement logic, §10).
- `util/test_util_implementation_equivalence.py` — cross-checks alternate
  implementations of some utility for numerical equivalence (title implies
  this guards against a prior refactor regressing results, but the
  specific comparison wasn't read in depth for this document — check the
  file directly if auditing that guarantee).

### 13.2 Integration tests (`integrationtests/`)

`test_base_set.py` runs **actual end-to-end training** (`case14_ieee` only,
via `gridfm_graphkit train` subprocess calls) for both `PowerFlow`
(`HGNS_PF_datakit_case14.yaml`) and `OptimalPowerFlow`
(`HGNS_OPF_datakit_case14.yaml`), overriding `epochs: 20` and
`hidden_size: 12` for speed, and asserts the resulting `PBE Mean` test
metric falls within **calibrated numeric bounds** captured from prior runs
on the reference CI hardware (`environment_fingerprint()` records
hostname/CPU/GPU/CUDA/cuDNN/package versions so a mismatched environment can
be flagged rather than silently compared against inapplicable bounds).
Supports a `--calibrate [N]` pytest flag (default 5 runs) to regenerate
bounds instead of asserting, with `--ci`/`--pad` controlling the confidence
interval and safety margin. Test fixture data is either found locally in
`data_out/` or downloaded from a fixed Google Drive file ID via `gdown` —
**this makes the integration test suite dependent on an external,
unversioned Google Drive artifact** (`case14_ieee.10000_scenarios_2_variants.zip`)
rather than a repo-tracked or regeneratable-on-CI fixture. **Neither `GRIT`
nor `StateEstimation` is covered by the integration suite** — only the two
`GNS_heterogeneous` × `{PowerFlow, OptimalPowerFlow}` combinations on
`case14_ieee` are exercised end-to-end.

`generate_test_data.py` (a helper, not a test file itself) downloads
`gridfm-datakit`'s default config from GitHub and regenerates PF/OPF
`case14_ieee` data via `gridfm_datakit` directly — used to produce fresh
fixture data rather than relying solely on the Google Drive download, though
`test_base_set.py`'s actual data-loading path only exercises the Drive
download branch unless `data_out/` is pre-populated.

---

## 14. Standalone scripts and notebooks (outside `gridfm_graphkit/`)

Two more trees exist alongside the installed package and are not reachable
from the CLI, `--help`, or anywhere in the rest of this document:

### 14.1 `scripts/` (top-level, sibling to `gridfm_graphkit/`)

- `scripts/benchmark_model_inference.py` — standalone `argparse` script
  (`--model {hetero,grit}`, `--config`, `--num_nodes`, `--num_edges`,
  `--num_gens`, `--iterations`, `--output_csv`, `--num_workers`, …) that
  builds a model directly from a YAML config and times forward-pass
  inference latency (`avg_time_per_sample_ms`) across batch sizes — distinct
  from `gridfm_graphkit benchmark`, which only times **dataloader**
  throughput and never touches the model (§2.5). This script is the one
  that actually benchmarks compute.
- `scripts/run_benchmark.sh` — shell driver that sweeps
  `benchmark_model_inference.py` over `GRIT_PF_datakit_case14.yaml` at
  synthetic graph sizes up to 30000 nodes / 100784 edges (`--num_nodes`/
  `--num_edges` overrides, not real grids) to test GRIT's scaling behavior.

Neither script is installed as a console entry point, packaged, or covered
by any test — they are dev tooling only, run manually from within `scripts/`.

### 14.2 `examples/notebooks/`

Two Jupyter notebooks, plus fixture data under `examples/data/contingency_texas/`:

- **`Tutorial_contingency_analisys.ipynb`** — loads ground-truth and
  `gridfm_graphkit predict`-produced CSVs for a Texas grid contingency
  study, and imports `compute_branch_currents_kA`, `compute_loading`,
  `create_admittance_matrix` from `gridfm_graphkit.datasets.postprocessing`,
  plus `compute_cm_metrics` from `gridfm_graphkit.utils.utils` and several
  plotting helpers (`plot_mass_correlation_density`, `plot_cm`,
  `plot_loading_predictions`, `plot_mass_correlation_density_voltage`) from
  `gridfm_graphkit.utils.visualization`. **This is the only place in the
  entire repository that imports `gridfm_graphkit/datasets/postprocessing.py`
  or `gridfm_graphkit/utils/{utils,visualization}.py`** — none of these three
  modules is imported by any task, CLI command, model, or test. They are
  real, working code, just reachable only from this one notebook, not from
  the package's own training/inference pipeline.
- **`Tutorial_reconstruction_visualization.ipynb`** — **broken against the
  current codebase.** It imports
  `gridfm_graphkit.datasets.powergrid_datamodule.LitGridDataModule` and
  `gridfm_graphkit.tasks.feature_reconstruction_task.FeatureReconstructionTask`
  — neither module exists anywhere in `gridfm_graphkit/` (verified by grep;
  the current dataset/task classes are `HeteroGridDatasetDisk`/
  `LitGridHeteroDataModule` and `PowerFlowTask`/`OptimalPowerFlowTask`/
  `StateEstimationTask`). It also references a homogeneous-graph API
  (`batch.x`, `batch.pe`, `batch.edge_index`, a single flat
  `node_normalizers` list, feature constants `PD, QD, PG, QG, VM, VA`
  imported from an unspecified module) that predates the current
  `HeteroData`-based bus/gen graph representation entirely. This notebook is
  a leftover from an earlier, homogeneous-graph version of the package and
  will fail on the first import cell as the code stands today. (The stale
  `from gridfm_graphkit.datasets.powergrid_datamodule import LitGridDataModule`
  line also appears, unexecuted, inside a docstring `Example:` block in
  `hetero_powergrid_datamodule.py`'s class docstring — same leftover, just
  harmless there since docstrings aren't executed.)

---

## 15. Known limitations / what this tool cannot do

- **No third model architecture.** "HGNS" is a config-naming convention for
  `GNS_heterogeneous`, not a separate model (§5).
- **`StateEstimation` has no `predict_step`** — `gridfm_graphkit predict`
  and `evaluate --save_output` are non-functional for SE (§4.5). No example
  SE config exists under `examples/config/` either (§9).
- **`MaskedBusMSE` never applies its OPF branch** due to a comparison bug
  (`self.args.task == "OptimalPowerFlow"` compares a namespace to a string
  and is always `False`) — every `HGNS_OPF_*.yaml` example config that uses
  this loss is silently missing its intended `Qg` reconstruction term (§7.1).
- **`DATASET_WRAPPER_REGISTRY` ships empty.** `--dataset_wrapper
  SharedMemoryCacheDataset`, referenced in every CLI `--help` string and in
  the README, does not exist anywhere in this package or its declared
  dependencies — it requires an unlisted external plugin (§8.1).
- **No resume-from-checkpoint.** `train`/`finetune` always start
  `Trainer.fit` fresh; the Lightning `last.ckpt` that gets written is never
  read back by this CLI (§12).
- **Multi-grid prediction is unimplemented.** `predict` and `evaluate
  --save_output` raise `NotImplementedError` if `len(data.networks) > 1`
  (§2.4) — multi-grid *training* and plain `evaluate` work fine, only
  saving predictions across multiple grids in one run does not.
- **Single logger backend.** Only `MLFlowLogger` (local file store) is
  wired up; no TensorBoard/W&B/remote-tracking-server option (§12).
- **Only `AdamW` + `ReduceLROnPlateau`.** No other optimizer or LR schedule
  is configurable (§3.4).
- **RWSE positional encoding is dense** (`to_dense_adj` + repeated dense
  matrix power) — memory/compute scales poorly (O(N²)/O(N³)) with grid
  size; no automatic disable or sparse fallback for large grids (§6.5).
- **SE test-time plotting is not DDP-gathered**, unlike PF/OPF — multi-GPU
  SE evaluation plots only reflect rank 0's shard of the test set (§4.5).
- **`data.mask_value`, once written into an input tensor, is
  indistinguishable from a genuine zero-valued feature** to the model —
  masking is destructive overwrite, not a separate signal channel (beyond
  whatever the model architecture chooses to also condition on
  `mask_dict`, which `GNS_heterogeneous`/`GRIT` do use internally, but the
  raw feature tensor itself carries no "this was masked" marker on its own).
- **`gridfm_graphkit/datasets/postprocessing.py`** (branch-current/loading
  computation helpers, `compute_branch_currents_kA`, `compute_loading`,
  `create_admittance_matrix`) and **`gridfm_graphkit/utils/{utils,visualization}.py`**
  (confusion-matrix and plotting helpers) have **zero references from any
  task, CLI command, model, or test** (verified by grep) — the only code in
  the repository that imports them is `examples/notebooks/Tutorial_contingency_analisys.ipynb`
  (§14.2). They are real, working modules, just entirely disconnected from
  the package's own training/inference pipeline.
- **`examples/notebooks/Tutorial_reconstruction_visualization.ipynb` is
  broken.** It imports classes (`LitGridDataModule` from a
  `datasets.powergrid_datamodule` module, `FeatureReconstructionTask` from a
  `tasks.feature_reconstruction_task` module) that do not exist anywhere in
  the current package — leftovers from a pre-`HeteroData` homogeneous-graph
  version of the codebase (§14.2).
- **`gridfm_graphkit benchmark` does not benchmark model compute** — it only
  times dataloader iteration speed (§2.5). Actual forward-pass/inference
  latency benchmarking exists only as the separate, unpackaged
  `scripts/benchmark_model_inference.py` (§14.1).
- **`loss_args` list-length mismatches are not validated.** If
  `training.losses`/`training.loss_weights`/`training.loss_args` have
  different lengths, `MixedLoss` catches the weights/losses mismatch, but a
  short `loss_args` list is silently truncated by `zip()` in
  `get_loss_function` rather than raising (§3.5).
- **Integration tests depend on an external, unversioned Google Drive
  file** rather than a repo-tracked fixture, and only cover
  `GNS_heterogeneous` × `{PowerFlow, OptimalPowerFlow}` on `case14_ieee`
  — `GRIT` and `StateEstimation` have no end-to-end CI coverage at all
  (§13.2).
- **AC/DC baseline runtime stats assume exactly 64 generation worker
  processes** (`NUM_PROCESSES = 64` hardcoded in `pf_ac_dc_baseline.py`)
  when normalizing reported per-scenario solve times — inaccurate for
  datasets generated with a different `gridfm-datakit`
  `settings.num_processes` (§4.6).
- **No built-in hyperparameter search / sweep tooling** — nothing beyond a
  single YAML config per run; any HPO (e.g. the LSF `hpo_trial_190` example
  in the README) is external shell scripting, not part of this package.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MD-ViSCo is a PyTorch + Hydra framework for multi-directional vital sign waveform conversion (ECG ↔ PPG ↔ ABP, plus IMP and BP scalar prediction). All entry points are thin Hydra wrappers; runtime orchestration lives in trainers and evaluators. Python 3.12 (the source uses PEP 701 multi-line f-strings), package name `mdvisco`.

## Common Commands

### Lint, format, type-check
The project uses Ruff (replaces Black + Flake8 + isort) and Pyright. Configuration is in `pyproject.toml`.

```bash
ruff format src/
ruff check src/ --fix
pyright src/
```

There is no test runner. `src/test.py` is the evaluation entry point, not a pytest suite — the deprecated `--run-pytest` and `--include-performance` flags are explicitly rejected.

### Train (always use `torchrun`, even single GPU)

```bash
# Single GPU (also valid for CPU when CUDA_VISIBLE_DEVICES='')
torchrun --standalone --nproc_per_node=1 --module src.train -m \
    train_dataset=train_pulsedb test_dataset=test_pulsedb \
    model=patchtst trainer=approximation_trainer_patchtst

# Multi-GPU
torchrun --standalone --nproc_per_node=2 --module src.train -m ...
```

### Evaluate

```bash
torchrun --standalone --nproc_per_node=1 --module src.test -m \
    evaluator=waveform_reconstruction_evaluator \
    test_dataset=test_pulsedb model=patchtst \
    evaluator.checkpoint_epoch=100
```

### Preprocess raw datasets to HDF5

```bash
python -m src.script.preprocess.preprocess \
    preprocessor=pulsedb_preprocessing \
    preprocessor.input_file=path/to/PulseDB_data.mat \
    preprocessor.output_file=path/to/processed_pulsedb.h5
```

### Feature extraction analysis

```bash
python -m src.script.features.feature_analysis \
    feature_analysis.gt_features_file=... \
    feature_analysis.model_name=mdvisco \
    feature_analysis.seed=42 \
    feature_analysis.direction=PPG2ECG \
    feature_analysis.dataset_name=PulseDB
```

## Architecture (the parts that span multiple files)

### Bootstrap order — must not be reshuffled
Every Hydra entry point (`train.py`, `test.py`, preprocess script) follows the same startup contract:

1. `register_resolvers()` — registers OmegaConf resolvers (e.g. `${encoder_output_size:1024}`). Must run before `@hydra.main` composes.
2. `register_core()` — registers `directions`, vital metadata, and single-direction configs in `ConfigStore`.
3. A series of `import_*()` calls (`import_datasets`, `import_models`, `import_criterions`, `import_optimizers`, `import_schedulers`, `import_trainers`, `import_processors`, `import_extractors`, `import_preprocessors`, `import_evaluators`). These walk submodules to trigger side-effect ConfigStore registrations; they are **not** imports for runtime use.
4. Only then can `@hydra.main(config_path="conf", config_name="config")` (or `"test_config"`) compose.

If a new model / dataset / trainer / criterion / processor isn't visible to Hydra, it almost always means its module wasn't picked up by the matching `import_*()` walker — check that the file lives directly under the corresponding `src/<group>/` package and isn't `_underscore_prefixed`.

### Config composition
- Root schema: `Config` dataclass in [src/conf/config.py](src/conf/config.py), registered as `base_config`.
- YAML entry points: [src/conf/config.yaml](src/conf/config.yaml) (train), `test_config` (eval), [src/conf/preprocessing_config.yaml](src/conf/preprocessing_config.yaml).
- Config groups under [src/conf/](src/conf/): `model/`, `train_dataset/`, `test_dataset/`, `trainer/`, `evaluator/`, `criterion/`, `optimizer/`, `scheduler/`, `early_stopping/`, `directions/`, `processor/`, `preprocessor/`, `extractor/`, `input_preprocessor/`, `checkpoint_io/`, `checkpoint_manager/`, `progress_bar/`, `csv_wrapper/`, `wandb_wrapper/`.
- Many shape values (`input_length`, `num_targets`, BP bounds) flow via Hydra interpolation from dataset configs into model configs. **Do not manually override them** — the README's "Common Configuration Issues" section calls this out as a frequent shape-mismatch trap.

### Trainer owns runtime, entry point stays thin
[src/train.py](src/train.py) does config validation, dataset construction, then calls `trainer.run_training(dataset_tuple)`. The trainer (subclasses of `BaseTrainer` in [src/trainers/trainer.py](src/trainers/trainer.py)) handles:
- Hardware/seed/threads/device binding via `LOCAL_RANK`
- DDP init / wrap / barrier / cleanup
- DataLoader + `DistributedSampler` construction
- Optimizer & scheduler creation **after** DDP wrap
- Training loops, metric sync, early stopping
- Checkpoint save/load with the canonical PyTorch DDP-aware pattern

The same pattern applies to evaluators in [src/evaluators/](src/evaluators/) — `test.py` only validates and instantiates; `evaluator.run_evaluation()` does everything.

### Checkpoint loading is DDP-aware
Pattern used throughout `BaseTrainer`:
1. Rank-0 preloads the checkpoint via `CheckpointIO`
2. Rank-0 loads model weights **before** DDP wrap (DDP then auto-broadcasts)
3. Optimizer / scheduler are created **after** DDP wrap
4. Trainer state (optimizer, scheduler, early-stopping state) is loaded on rank-0 and broadcast as a payload

Path construction is handled by [src/utils/checkpoint_manager.py](src/utils/checkpoint_manager.py); load/save bytes by [src/utils/checkpoint_io.py](src/utils/checkpoint_io.py). Checkpoint paths embed training hyperparameters (`batch_size`, `num_epochs`, `learning_rate`, `seed`) — when evaluating, **these must match the values used during training** or `CheckpointManager.build_path` will look in the wrong directory.

### Two-stage vs single-stage models
- Two-stage (MD-ViSCo, PatchTST, NABNet): stage 1 produces a normalized ABP waveform; stage 2 (`refinement_*`) predicts SBP/DBP scalars used to unscale the waveform to mmHg. Two checkpoint managers are configured (one per stage).
- Single-stage (PPG2ABP, WaveNet, P2E-WGAN): direct ABP output, single checkpoint manager. Cascade variants chain stage1 approximation → stage2 refinement.

### Datasets, directions, lazy normalization
- `BaseDataset` (in [src/dataset/base_dataset.py](src/dataset/base_dataset.py)) returns `Sample` dataclasses with raw waveform fields (`ecg_raw`, `ppg_raw`, `abp_raw`, `imp_raw`, `bp_raw`) and optional demographics. Demographics are PulseDB-only.
- Normalization, padding, and trimming all happen at batch time inside the collate function ([src/utils/collate_utils.py](src/utils/collate_utils.py)) — datasets stay raw on purpose.
- `Directions` ([src/core/direction.py](src/core/direction.py)) is the dataset-agnostic container of allowed source→target flows. `train.py:_validate_directions_with_preprocessing` enforces that every direction's source vital is present in both trainer-side and dataset-side `input_preprocessing` mappings before any tensor is touched.
- Splits supported: ratio split, patient-aware split (`use_patient_split`), and the NABNet-vanilla split (`use_nabnet_vanilla_split`). Train/val/test datasets may be `torch.utils.data.Subset` wrappers — use the helpers in [src/utils/dataset_utils.py](src/utils/dataset_utils.py) (`get_dataset_attribute`, etc.) when reading attributes that the wrapper doesn't expose.

### Demographic / patient information
Some models (notably MD-ViSCo BPModel, PatchTST refinement) accept demographics. Wired via `model.use_demographics=true model.num_demographic_channels=5`, but only when the dataset is PulseDB-derived.

### NABNet on MIMIC PERform Large needs overrides
`model.model_depth=5 model.model_width=32 model.attention_type=lstm` (called out in the README — easy to miss when swapping datasets).

### MIMIC PERform Large lacks ABP
Refinement / BP-prediction trainers will not work with `train_mimic_perform_large` — only approximation across ECG/PPG/IMP. AF classification has its own path: `trainer=classification_trainer` + `train_dataset=train_mimic_perform_af_*`.

## Code Conventions Specific to This Repo

- All model `__init__` accept a Hydra-instantiated config dataclass (subclass of `BaseModelConfig`). Don't add positional hyperparameter arguments.
- Configs are dataclasses (not pydantic). `MISSING` from omegaconf is the standard sentinel for "must be set in YAML / overrides."
- Logging: `logger = logging.getLogger(__name__)` per module. Avoid `print` for anything other than truly user-facing CLI scripts.
- New plugin modules belong in the package matching their type (`src/model/foo.py`, `src/criterions/foo.py`, etc.). The `import_*()` walker picks them up automatically; no central registry edit needed.

## Non-Obvious Gotchas

### `WandBWrapper` is now a SwanLab adapter, not wandb
[src/loggings/wandb_wrapper.py](src/loggings/wandb_wrapper.py) keeps the historical `WandBWrapper` / `WandBWrapperConfig` class names and the `wandb_enabled` / `progress_bar.wandb` attribute paths so existing yamls and ~30 call sites in trainers/evaluators continue to work — but **inside the wrapper every call now goes through `swanlab.init / log / finish` natively**. wandb the library is no longer a dependency. When extending logging:
- Don't reintroduce `import wandb`. Use `swanlab` directly inside the wrapper.
- `swanlab.init` types its `workspace / project / experiment_name` as required `str` (not `Optional[str]`); pass them via a kwargs dict that only includes non-None values, otherwise pyright complains.
- `swanlab.watch` does not exist; the wrapper's `watch()` method is a no-op kept for backward compat.

### Hydra defaults lists require `override` for re-specified groups
When a child trainer/evaluator yaml re-specifies a default already set by its base (e.g. `base_approximation_trainer.yaml` sets `/processor@processor: waveform_processor`, and `approximation_trainer_mdvisco.yaml` wants `waveform_processor_ref_test`), the child entry **must** be prefixed with `override`:

```yaml
defaults:
  - base_approximation_trainer
  - override /processor@processor: waveform_processor_ref_test  # not just "/processor@processor: ..."
```

Without `override`, Hydra raises `Multiple values for processor@trainer.processor`. Existing files already affected: `approximation_trainer_mdvisco.yaml`, `approximation_trainer_gan.yaml`, `refinement_trainer_mdvisco.yaml`, `refinement_trainer_gan.yaml`, `refinement_trainer_ppg2abp.yaml`, `blood_pressure_p2ewgan_evaluator.yaml`, `waveform_reconstruction_p2ewgan_evaluator.yaml`. Don't strip `override` when reformatting.

### `pyppg==1.0.73` lives in a `--no-deps` pip block
[environment.yml](environment.yml) installs pyppg via `pip install --no-deps pyppg==1.0.73` because pyppg's lockfile-style hard pins (`scipy==1.9.1`, `numpy==1.23.2`, ...) conflict with the modern py3.12 stack. The actually-imported APIs (`pyPPG.preproc`, `Fiducials`, `pack_ppg._ErrorHandler`) work fine against newer libs. **Don't add other packages to that pip block** — the `--no-deps` flag is global to the pip install command and would strip their transitive deps too. SwanLab in particular must be installed separately (`pip install swanlab` post-env-creation, or via the macOS env file's separate pip block which has no `--no-deps`).

## Helper Scripts

[scripts/pulsedb/](scripts/pulsedb/) holds 4 launcher scripts for the canonical PulseDB train/eval flow (stage 1 approximation → stage 2 refinement → eval each stage). Parameters live as plain `VAR=value` assignments in an "Edit here" block at the top of each script — change the file, don't pass env vars. Direction defaults assume `ecg_ppg_abp_clinically_meaningful` (multi) for waveform stage and `ppg2bp_ecg2bp` (multi) for BP stage; 01 ↔ 03 must use the same direction (eval loads the trained checkpoint by direction tag), as must 02 ↔ 04.

[scripts/sync_from_github.sh](scripts/sync_from_github.sh) force-aligns a server-side checkout to `origin/main` (`git fetch + reset --hard + clean -fd`). Tracked files get overwritten; untracked training products under `outputs/ checkpoints/ logs/ results/ data/` are preserved because they're already gitignored.

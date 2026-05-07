# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MD-ViSCo is a PyTorch + Hydra framework for multi-directional vital sign waveform conversion (ECG ↔ PPG ↔ ABP, plus IMP and BP scalar prediction). All entry points are thin Hydra wrappers; runtime orchestration lives in trainers and evaluators. Python 3.10, package name `mdvisco`.

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

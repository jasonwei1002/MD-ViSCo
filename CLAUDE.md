# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A standalone working copy of **MD-ViSCo** (Meyer, Hur, Choi — *A Unified Model for Multi-Directional Vital Sign Waveform Conversion*, IEEE JBHI 2026; upstream `github.com/fr-meyer/MD-ViSCo`). The intent here is to actually **run** preprocessing / training / evaluation locally against the `.mat` data present in this directory.

### What is present vs. missing (verify before assuming runnable)

Present at repo root: `README.md`, `src/`, and two raw PulseDB `.mat` files:
- `Train_Subset.mat` (~26 GB) — PulseDB **training** subset.
- `CalFree_Test_Subset.mat` (~3.3 GB) — PulseDB **calibration-free test** subset.

**Not present** (must be obtained/produced before a full run):
- `environment.yml` — the upstream conda spec is **not** vendored here. Get it from upstream (`github.com/fr-meyer/MD-ViSCo`) or build an equivalent env (Python 3.10, PyTorch w/ CUDA 12.1+, Hydra, OmegaConf, h5py, transformers, neurokit2, pyPPG). README "Installation" assumes `conda env create -f environment.yml && conda activate mdvisco`.
- Preprocessed `.h5` files — training reads HDF5, **not** the raw `.mat`. You must run the preprocessing step first (below).
- Checkpoints — none here; evaluation needs a checkpoint whose path is reconstructed from training hyperparameters (see "Checkpoints").

This directory is **not a git repo**. The huge `.mat` files are slow to open and must never be read/parsed as text — go through the preprocessing pipeline, which is HDF5-streaming aware.

## How to read this codebase (architecture)

MD-ViSCo is a **Hydra + ConfigStore, fully config-driven** framework. Nothing is wired by hand: entry points discover and register every component, then `hydra.utils.instantiate` builds the object graph from `_target_` strings. Understanding the registration flow is the key to navigating it.

### Entry points and the registration handshake

- `src/train.py` and `src/test.py` are **thin entry points**. At import time each one calls `register_resolvers()`, `register_core()`, and a series of `import_*()` calls (`import_models`, `import_datasets`, `import_trainers`, `import_criterions`, `import_optimizers`, `import_schedulers`, `import_processors`, `import_extractors`, `import_preprocessors`).
- Those `import_*()` helpers all funnel through `src/utils/module_utils.py::import_modules`, which **walks every submodule of a package and imports it purely for its side effect**: each module registers its config dataclass with Hydra's `ConfigStore`. This is the registry pattern — there is no explicit `REGISTRY` dict; the ConfigStore *is* the registry, keyed by `(group, name)`.
- `src/conf/config.py::Config` is the **root schema** (a dataclass with `train_dataset / test_dataset / trainer / evaluator` fields), registered as `base_config` in `src/conf/__init__.py`. `src/conf/config.yaml` composes it via Hydra `defaults`.
- After composition, `main()` calls `instantiate(cfg.trainer)` — Hydra recursively instantiates the trainer *and its nested model/criterion/optimizer/scheduler* from their `_target_`s. **The trainer owns all runtime orchestration** (device binding, DDP lifecycle, DataLoader + DistributedSampler, optimizer/scheduler creation *after* DDP wrap, training loop, metric sync, early stopping, checkpointing). `train.py` only validates config, sets up logging, builds datasets, and calls `trainer.run_training(dataset_tuple)`.

So the trace for any run is: **`config.yaml` defaults → CLI overrides → ConfigStore-registered dataclasses → `instantiate` → `trainer.run_training`**. To find what a flag does, find the dataclass field (in the relevant `src/.../*.py`), not just the YAML.

### Core domain model (`src/core/`)

- `domain.py::Vital` enum = `{ECG, PPG, ABP, IMP, BP}` (BP = scalar SBP/DBP/MAP; the rest are waveforms).
- `domain.py::Direction` = one `source(list[Vital]) → target(Vital)` mapping with a string `key()`: single-source `"PPG2ABP"`, multi-source `"PPG+ECG2ABP"`. `Direction.parse()` is the inverse.
- `direction.py::Directions` = an immutable container of active directions. `trainer.direction_mode` is `single` (exactly one direction) or `multi` (>1). `train.py::validate_config` enforces the count matches the mode.
- Configs select directions via the `directions` Hydra group (`src/conf/directions/*.yaml`, e.g. `ppg2abp_ecg2abp.yaml`). `train.py::_validate_directions_with_preprocessing` cross-checks that every direction's source vital actually exists in `input_preprocessing` — a common failure point.

### Two-stage scaling/refinement (the central design)

BP models are **two-stage**, and the two stages are **separate models, trained independently, combined only at evaluation**:

- **Stage 1 — approximation** (`*_approximation` model + `approximation_trainer_*`): source waveform(s) → *normalized* ABP waveform (shape only, unitless). e.g. `mdvisco_approximation` = `UNetSwinUnet` (U-Net + Swin).
- **Stage 2 — refinement** (`*_refinement` model + `refinement_trainer_*`): predicts SBP/DBP scalars used to **unscale** the normalized waveform back to mmHg. e.g. `mdvisco_refinement` = `BPModel` (a `SingleStageModel`), which reads **raw PPG/ECG (+demographics) directly** and emits scalars — it does **not** consume stage-1 output during its own training, and it is **not** weight-initialized from the approximation (different architectures).

`src/model/two_stage_model.py` is the base orchestrator (composition over inheritance — wraps two independent `BaseModel`s in a `ModuleDict`). Two concrete subclasses:
- `TwoStageCascadeModel` — stage1 output passes straight into stage2.
- `TwoStageScalingModel` — stage2 BP scalars rescale stage1's normalized waveform. At forward, stage1 is forced to `eval()` (frozen feature extractor, `auto_eval_in_forward=True`); stage2 is controlled externally.

The composition lives at **eval time**: e.g. `model=mdvisco_inference_scale` with the `scaling_two_stage_*_evaluator`, which loads stage1 and stage2 checkpoints from **separate** dirs (`weights/pretrained/stage1/`, `weights/pretrained/stage2/`). Single-stage models (`ppg2abp`, `wavenet`, `p2e_wgan`) emit ABP directly and use the standard single-model checkpoint path.

`trainer.is_finetuning: true` on refinement trainers selects the **"finetuning" data-split scenario** (train+val+test must sum to 1.0, enabling a held-out test split) — see `src/utils/dataset_utils.py`. It is **not** a weight warm-start flag (`is_pretraining` defaults to `False` everywhere).

### Models (`src/model/`)

All instantiated via Hydra `_target_`, configured in `src/conf/model/*.yaml`:
- `mdvisco.py` — the proposed model: 1D U-Net + Swin Transformer encoder/decoder + **AdaIN** style injection (target domain as style indicator) + demographic text embedding (PulseDB only). Also defines `BPModel` (refinement) and `VitalEncoder`.
- Baselines: `nabnet.py`, `patchtst.py`, `ppg2abp.py` (UNetDS64 / MultiResUNet1D), `p2e_wgan.py`, `wavenet.py`. `af_classifier.py` for atrial-fibrillation classification.
- `base_model.py::BaseModel` is the shared abstract base (checkpoint load/save, layout). `cnn_encoder.py` / `scaling_model.py` / `single_stage_model.py` / `cascade_model.py` are shared building blocks.

### Data path (`src/dataset/`, `src/script/preprocess/`, `src/preprocessors/`)

- **Preprocessing is a separate offline step**: `python -m src.script.preprocess.preprocess preprocessor=<name> ...` converts raw `.mat`/`.h5` into the **HDF5** format the training pipeline expects. Datasets: PulseDB (MIMIC-III + VitalDB), UCI, MIMIC PERform AF, MIMIC PERform Large.
- At train time, datasets return **raw waveforms with lazy normalization** — padding/trimming and per-batch normalization happen in the **collate function** (`src/utils/collate_utils.py`, direction-aware), driven by `trainer.input_preprocessing`.
- A dataset config resolves its file as **`dataset_path` (runtime `???`) + `dataset_folder` (e.g. `PulseDB/mdvisco_processed/PulseDB`) + `file_name`** (`Train_Subset.h5`, `CalFree_Test_Subset.h5`). Set `dataset_path` to wherever you wrote the preprocessed `.h5`.
- **Split semantics matter**: `use_patient_split=true` → patient-level / subject-disjoint (calibration-free); sample-level re-split → calibration-based.
- Demographics (`age/gender/height/weight/bmi`) exist only for PulseDB, encoded via `src/preprocessors/demographics_text_encoder.py` (DistilBERT text embedding — chosen for cross-dataset schema agnosticism).

### Which `.mat` feeds which stage (non-obvious — required reading several configs)

- `Train_Subset.mat` → `Train_Subset.h5` → `train_dataset=train_pulsedb` → **stage-1 approximation** training (targets ECG/PPG/ABP waveforms, 80/20 train/val, no test).
- `CalFree_Test_Subset.mat` → `CalFree_Test_Subset.h5` → `train_dataset=train_pulsedb_refinement_bp` (BP) or `train_pulsedb_refinement_abp` (ABP) → **stage-2 refinement**, with an internal **81/9/10** finetuning split.
- The refinement stage does **not** train on `Train_Subset`. It only benefits from stage 1 at eval, where the frozen approximation supplies the normalized waveform that stage-2's SBP/DBP rescale.

### Hydra interpolation rule (do not fight it)

Dataset configs are the single source of truth for `input_length`/`input_size`, `num_targets`, and BP normalization bounds (`dbp_min`, `sbp_max`). Model configs pull these via interpolation (`${train_dataset.input_size}`, `${oc.select:train_dataset.input_size,${test_dataset.input_size}}`, etc.). **Never manually override `model.input_length` / `model.num_targets`** — that causes shape-mismatch errors. Always specify both `train_dataset` and `test_dataset` or you get `InterpolationKeyError`. (README "Common Configuration Issues" documents both.)

## Commands

Everything runs through **`torchrun` and Hydra `-m`**, even single-GPU (`--nproc_per_node=1`). Requires the conda env active (`conda activate mdvisco`) — see "What is present vs. missing".

```bash
# 1) Preprocess the local .mat files once → HDF5 (run for each subset)
python -m src.script.preprocess.preprocess \
    preprocessor=pulsedb_preprocessing \
    preprocessor.input_file=Train_Subset.mat \
    preprocessor.output_file=<dataset_path>/PulseDB/mdvisco_processed/PulseDB/Train_Subset.h5
python -m src.script.preprocess.preprocess \
    preprocessor=pulsedb_preprocessing \
    preprocessor.input_file=CalFree_Test_Subset.mat \
    preprocessor.output_file=<dataset_path>/PulseDB/mdvisco_processed/PulseDB/CalFree_Test_Subset.h5

# 2) Train stage-1 approximation (swap model / trainer via config groups)
torchrun --standalone --nproc_per_node=1 --module src.train -m \
    train_dataset=train_pulsedb test_dataset=test_pulsedb \
    train_dataset.dataset_path=<dataset_path> test_dataset.dataset_path=<dataset_path> \
    model=mdvisco_approximation trainer=approximation_trainer_mdvisco

# 3) Train stage-2 BP refinement; direction selects source modality
torchrun --standalone --nproc_per_node=1 --module src.train -m \
    trainer=refinement_trainer_mdvisco trainer.directions=ppg2bp_ecg2bp \
    train_dataset=train_pulsedb_refinement_bp \
    train_dataset.dataset_path=<dataset_path>

# 4) Evaluate — checkpoint_epoch + training hyperparameters must match the trained ckpt
torchrun --standalone --nproc_per_node=1 --module src.test -m \
    evaluator=waveform_reconstruction_evaluator \
    test_dataset=test_pulsedb model=patchtst evaluator.checkpoint_epoch=100

# CPU-only: prefix with CUDA_VISIBLE_DEVICES='' and keep --nproc_per_node=1
```

### Checkpoints

Checkpoint paths embed `batch_size / num_epochs / learning_rate / seed` (and scheduler/early-stopping patience). Evaluation overrides must **reproduce the training values** or the path won't resolve. Two-stage eval loads stage1/stage2 from separate `weights/pretrained/{stage1,stage2}/` dirs.

### PulseDB full-feature refinement (demographics / PI mode)

The stock `mdvisco_refinement` ships in **UCI mode** (`pi: false`, `text_encoder_pipeline: null`) — demographics are present in the batch but the model does not consume them. The paper's full PulseDB variant (§6.2.2, DistilBERT demographic-text-embedding fusion) uses two **added** configs:

- `model=mdvisco_refinement_pulsedb` (`src/conf/model/mdvisco_refinement_pulsedb.yaml`) — adds `text_encoder_pipeline` (DistilBERT, 768→512 to match `projection_dim`) and sets `vital_encoders.{ppg,ecg}.pi: true`. `BPModel` derives PI purely from `text_encoder_pipeline is not None` (`mdvisco.py:3847`), so setting the pipeline is what turns PI on.
- `trainer=refinement_trainer_mdvisco_pulsedb` (`src/conf/trainer/refinement_trainer_mdvisco_pulsedb.yaml`) — same as `refinement_trainer_mdvisco` but swaps in the PI model. Defined standalone off `base_refinement_scalar` (not by inheriting the other trainer), with `override /directions@directions:` placed **after** the append entries to avoid Hydra "Multiple values" errors.

Verified-runnable command (composes with 0 missing fields; PPG+ECG→SBP/DBP, multi-WCL, 60 epochs):

```bash
torchrun --standalone --nproc_per_node=1 --module src.train -m \
    trainer=refinement_trainer_mdvisco_pulsedb \
    trainer.use_patient_information=true \
    train_dataset=train_pulsedb_refinement_bp \
    test_dataset=test_pulsedb_refinement_bp \
    train_dataset.dataset_path=<dataset_path> test_dataset.dataset_path=<dataset_path> \
    trainer.progress_bar.wandb_wrapper.project_name=<proj> \
    trainer.progress_bar.wandb_wrapper.entity=<entity>
```

- `trainer.use_patient_information=true` is **mandatory** (`config.yaml` marks it `???`); it gates demographics and tags the checkpoint path `_PI_True`.
- W&B `project_name` / `entity` are the only other mandatory `???` — or disable logging with `trainer.progress_bar.wandb_wrapper=null`.
- `ppg2bp_ecg2bp` resolves to **one multi-source** direction (`[PPG,ECG]→BP`), so the auto `direction_mode=single` is correct — do **not** force `multi`.
- DistilBERT (`distilbert-base-uncased`) loads via `from_pretrained` **without** `from_tf=True`, so a PyTorch/safetensors weight cache is required (a TF-only cache raises `OSError`).
- Pre-flight on any new env: `python -m src.train --cfg job --resolve <same overrides>` composes only (no GPU/data) and surfaces missing fields / wiring.

### Local modifications to this snapshot (diverges from upstream)

Three changes were made so training runs on **Python 3.12 + omegaconf 2.3.0 + hydra 1.3.2** (the upstream entry point crashes at import on this stack):

- `src/trainers/trainer.py` — moved 7 `*Config` imports (`CriterionBaseConfig`, `CheckpointManagerConfig`, `CheckpointIOConfig`, `ProgressBarConfig`, `EarlyStoppingConfig`, `DirectionsConfig`, `BasePreprocessorConfig`) out of the `TYPE_CHECKING` block to runtime. Without it, `cs.store()` → `OmegaConf.structured()` → `get_type_hints()` raises `NameError: CriterionBaseConfig` at import and **every** trainer fails to register.
- Added the two `*_pulsedb` configs documented above.
- **Known unfixed upstream bug**: `trainer/refinement_trainer_mdvisco.yaml` (and other trainers) select `directions` without the `override` keyword while `base_refinement_scalar` already set `ppg2abp`, so they fail to compose on Hydra 1.3.x with "Multiple values for directions@trainer.directions". Use the `_pulsedb` trainer, or add `override` to the directions default before running the non-PI variant.

### Code quality (upstream gate — no config files vendored here)

```bash
ruff format src/
ruff check src/ --fix
pyright src/
```

There is no test suite in this snapshot.

## Conventions to respect when reading

- **No central registry dict** — registration is "import the module, it stores its dataclass in `ConfigStore`". If a config group seems unregistered, check that its `import_*()` runs in the entry point.
- **Config groups live in `src/conf/<group>/`** (`model`, `trainer`, `train_dataset`, `test_dataset`, `criterion`, `optimizer`, `scheduler`, `directions`, `evaluator`, `processor`, `preprocessor`, `extractor`, `input_preprocessor`, `early_stopping`, `checkpoint_manager`, `checkpoint_io`, `progress_bar`). The matching dataclass schema lives next to the code it configures.
- This is a third-party snapshot — keep edits faithful to the published baseline; prefer minimal, reversible changes over refactors.
- Architecture/organization is adapted from the **GenHPF** framework; feature extraction uses **pyPPG** (PPG) and **NeuroKit2** (ECG).

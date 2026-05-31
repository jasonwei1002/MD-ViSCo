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

`trainer.is_finetuning: true` on refinement trainers selects the **"finetuning" data-split scenario** (train+val+test must sum to 1.0, enabling a held-out test split) — see `src/utils/dataset_utils.py`. `is_finetuning` only picks the **split**, not the weights. The paper's separate **pretraining** phase uses `is_pretraining: true` instead (its scenario is train+val only, sums to 1.0, **no** test split), run on `Train_Subset`; the finetune phase then warm-starts the refinement encoders from that pretrain checkpoint via `checkpoint_managers.load`. This snapshot ships **only** the finetune dataset config, so `is_pretraining` is `False` in every shipped config — but it is a real scenario, not a dead flag (see the corrected stage-2 data flow below).

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
- `CalFree_Test_Subset.mat` → `CalFree_Test_Subset.h5` → `train_dataset=train_pulsedb_refinement_bp` (BP) or `train_pulsedb_refinement_abp` (ABP) → **stage-2 refinement finetune**, with an internal **81/9/10** finetuning split.
- **Stage-2 refinement is a two-phase procedure in the paper** (§III.E "Data split" + Table II), and it **does** use `Train_Subset`:
  1. **Pretrain** — self-supervised **weighted-contrastive (WCL)** pretraining on the PulseDB *original train/val* patients = `Train_Subset` (2,494 patients, **80/20**, `is_pretraining: true`, **no** test split → 721,728 / 180,432 samples). This initializes the refinement's own waveform encoder (`E_W`) and patient-info encoder (`E_PI`) — it is **not** initialized from the stage-1 approximation (different architectures); the warm-start is from this contrastive pretrain checkpoint.
  2. **Finetune** — on the held-out *original test* patients = `CalFree_Test_Subset` (279 patients, re-split **81/9/10**, `is_finetuning: true` → 90,342 / 10,038 / 11,155 samples), a calibration-based setting where the same patient may appear in finetune-train and finetune-test. The finetune step warm-starts from the pretrain checkpoint via `checkpoint_managers.load`.
- **This snapshot only ships the finetune dataset config** (`train_pulsedb_refinement_bp` → `CalFree_Test_Subset.h5`); there is **no** `Train_Subset` refinement config. So the documented `train.sh` runs **finetune-only**, from-scratch, with WCL **off** (`WCL_False` in the checkpoint path) — i.e. it does **not** reproduce the paper's full pretrain→finetune+WCL recipe. To match the paper you must add a Train_Subset refinement-pretrain config (`file_name: Train_Subset.h5`, 80/20, `is_pretraining=true`, WCL on), run it first, then point the finetune's `checkpoint_managers.load.base_dir` at its output.
- At **eval** time, the frozen stage-1 approximation supplies the normalized waveform that the (finetuned) stage-2 SBP/DBP rescale to mmHg.

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
    trainer.overwrite_checkpoint=true \
    train_dataset=train_pulsedb_refinement_bp \
    test_dataset=test_pulsedb_refinement_bp \
    train_dataset.dataset_path=<dataset_path> test_dataset.dataset_path=<dataset_path> \
    trainer.progress_bar.wandb_wrapper.project_name=<proj> \
    trainer.progress_bar.wandb_wrapper.entity=<entity>
```

- `trainer.use_patient_information=true` is **mandatory** (`config.yaml` marks it `???`); it gates demographics and tags the checkpoint path `_PI_True`.
- `trainer.overwrite_checkpoint=true` is **required for any multi-epoch run**. The best-model checkpoint is a single rolling file (`epoch=None` → no `_epoch_N` suffix, e.g. `PPG+ECG2BP_checkpoint_S_42.pt`), re-saved every time val improves (`scalar_regression_trainer.py:642-667`). With the resolved default `overwrite_checkpoint=false`, `CheckpointManager.build_path(key="save", overwrite=False)` raises `ValueError: Checkpoint exists ... overwrite is False` on the **second** best-improvement (or first, if a stale ckpt from a prior run exists) — so training crashes mid-run. `config.yaml` marks `overwrite_checkpoint: ???`, i.e. it is a user decision; the schema default fills it to `false`, which only survives a compose check, not real training. Deleting the stale file alone is **not** a fix (next improvement re-crashes).
- W&B `project_name` / `entity` are the only other mandatory `???` — or disable logging with `trainer.progress_bar.wandb_wrapper=null`.
- `ppg2bp_ecg2bp` resolves to **one multi-source** direction (`[PPG,ECG]→BP`; BPModel runs a per-vital encoder and **averages** the two SBP/DBP predictions), so the auto `direction_mode=single` is correct — do **not** force `multi`.
- ⚠️ **Paper-vs-config mismatch (affects what you reproduce).** This `ppg2bp_ecg2bp` (both PPG+ECG required, aggregated) matches the paper's **multi-input ablation** (Appendix E / Table XIII: `PPG+ECG→ABP`), **not** its headline refinement. The paper's main BP result (§III.D, the AAMI/BHS numbers) trains **two single-source branches** `PPG→ABP` and `ECG→ABP` jointly (MAE summed over `i∈{ECG,PPG}`) and **infers from one modality**. The naming convention is inconsistent: the sibling `ppg2abp_ecg2abp.yaml` correctly ships **two** single-source directions (`[PPG]→ABP`, `[ECG]→ABP` → `direction_mode=multi`), but `ppg2bp_ecg2bp.yaml` ships **one** multi-source direction. To reproduce the paper's headline BP refinement, use a directions config with **two** single-source BP entries (`[PPG]→BP`, `[ECG]→BP`) — i.e. the BP analog of `ppg2abp_ecg2abp` (singles `ppg2bp.yaml` / `ecg2bp.yaml` exist; a two-entry combo does not).
- **Paper hyperparameters differ from `train.sh`.** Paper (§III.E.2): batch size **2048**, LR 1e-3, scheduler patience 3, ES patience 5, **30K steps**, WCL **on** (App. C: `λ_MAE=0.001`, `λ1=λ2=0.01`). The verified-runnable command above uses **BS 32 / 60 epochs / WCL off** — composes and trains, but is not the paper's recipe.
- Refinement encoders (App. B.2): waveform `E_W` = **PatchTSMixer** (hidden 64, 15 layers, expansion 5, patch length 4); `E_PI` = DistilBERT base-uncased (6L/12H/768) → 2-layer MLP to 512. Approximation = U-Net + SwinT (patch-embed 256, window 4, 32 heads, AdaIN style dim 64).
- DistilBERT (`distilbert-base-uncased`) loads via `from_pretrained` **without** `from_tf=True`, so a PyTorch/safetensors weight cache is required (a TF-only cache raises `OSError`).
- Pre-flight on any new env: `python -m src.train --cfg job --resolve <same overrides>` composes only (no GPU/data) and surfaces missing fields / wiring.

### BP refinement recipe — pretrain → finetune (two-step)

The `pretrain.sh`/`finetune.sh` scripts default to the **multi-source** direction `ppg2bp_ecg2bp` (`[PPG,ECG]→BP`, both signals in at once, predictions aggregated; `direction_mode=single` since it is one direction). This is the paper's **Appendix-E multi-input** setting — best MAE *when both signals are present at inference*, but it cannot run single-modality. Trainer: the **stock** `refinement_trainer_mdvisco_pulsedb`.

To instead reproduce the paper's **headline** single-source result (§III.D + Table II: two single-source branches `PPG→BP`/`ECG→BP` jointly trained, single-modality inference, the AAMI/BHS numbers), swap the trainer to `refinement_trainer_mdvisco_pulsedb_dual` (added). Pick the variant by deployment: multi-source if PPG **and** ECG are always co-present; `_dual` if any single signal must work.

Added configs:
- `src/conf/directions/ppg2bp_ecg2bp_dual.yaml` — **two** single-source directions (`[PPG]→BP`, `[ECG]→BP`) ⇒ `direction_mode=multi` (the BP analog of `ppg2abp_ecg2abp`).
- `src/conf/train_dataset/train_pulsedb_refinement_pretrain_bp.yaml` — refinement **pretrain** split: `file_name: Train_Subset.h5`, **80/20**, `test_ratio: 0.0` (the "pretraining" scenario). Same preprocessing/targets as `train_pulsedb_refinement_bp`. Used by **both** variants.
- `src/conf/trainer/refinement_trainer_mdvisco_pulsedb_dual.yaml` — clone of `refinement_trainer_mdvisco_pulsedb` but `override /directions@directions: ppg2bp_ecg2bp_dual` and **`direction_mode: multi`** (without the latter, `train.py::validate_config` raises "direction_mode is 'single', but multiple directions are provided").

Recipe flags (both steps add `trainer.use_wcl=true` to turn WCL on — default is `False`, which only labels the ckpt `_WCL_False` and withholds the `bp_raw/age_raw/gender_raw` fields WCL needs):
- **Pretrain**: `trainer.is_pretraining=true trainer.is_finetuning=false` + `train_dataset=train_pulsedb_refinement_pretrain_bp`. `_determine_training_scenario` checks `is_finetuning` **first**, so `is_finetuning=false` is required for the "pretraining" (80/20, no-test) scenario.
- **Finetune**: `train_dataset=train_pulsedb_refinement_bp` (trainer keeps `is_finetuning=true`), warm-started from the pretrain checkpoint via `trainer.load_weights_from=<path>` (see below).
- Both compose **and** pass `validate_config` (verified locally; runtime needs GPU/data on the server).

**Warm-start handoff — `trainer.load_weights_from` (added field, see "Local modifications").** The stock trainer can only *resume* from its own save path (default `checkpoint_mapping` maps every component to the `"save"` manager, and `_load_checkpoint_from_disk` raises if a component's file is missing — so `load_model_weights=true` on a fresh finetune dir crashes). Cross-run warm-start is therefore done with the **added** `load_weights_from` field: give it an explicit checkpoint *file* path and it loads **only** the model weights (reusing `prepare_model_weights`/`load_from_checkpoint_dict`, rank-0 + DDP broadcast), without touching optimizer/scheduler/early-stopping state or the save path. Pretrain & finetune share the same architecture/directions, so `load_state_dict(strict=True)` matches.

**Three scripts** (run from repo root): `pretrain.sh` (Step 1, auto-discovers the produced checkpoint and writes its path to `./weights/.last_pretrain_ckpt`), `finetune.sh [ckpt]` (Step 2, warm-starts from `$1` or `.last_pretrain_ckpt`; `COLD_START=1` to skip), and `train.sh` (orchestrator: `pretrain.sh` then `finetune.sh`). Checkpoint filename embeds the direction: multi-source (default) ⇒ `PPG+ECG2BP_checkpoint_S_42.pt`; `_dual` (multi-direction) ⇒ no prefix (`checkpoint_S_42.pt`). `pretrain.sh`'s discovery glob `*checkpoint_S_*.pt` matches both.

### Local modifications to this snapshot (diverges from upstream)

These changes were made so training runs on **Python 3.12 + omegaconf 2.3.0 + hydra 1.3.2** (the upstream entry point crashes at import on this stack):

- `src/trainers/trainer.py` — moved 7 `*Config` imports (`CriterionBaseConfig`, `CheckpointManagerConfig`, `CheckpointIOConfig`, `ProgressBarConfig`, `EarlyStoppingConfig`, `DirectionsConfig`, `BasePreprocessorConfig`) out of the `TYPE_CHECKING` block to runtime. Without it, `cs.store()` → `OmegaConf.structured()` → `get_type_hints()` raises `NameError: CriterionBaseConfig` at import and **every** trainer fails to register.
- `src/conf/train_dataset/base_{pulsedb,uci,mimicperformaf,mimicperformlarge}.yaml` — fixed `_target_` from the **config** class (`*Config`) to the **dataset** class (`*Dataset`). The schema dataclass already declares the correct `_target_: *Dataset`, but each base YAML re-declared `_target_: *Config` (matches upstream `main`). On Hydra 1.3.x the same-named YAML value **overrides** the schema default, so `instantiate(cfg.train_dataset)` built a `PulseDBConfig` dataclass instead of the dataset → `create_single_dataset` crashes with `TypeError: object of type 'PulseDBConfig' has no len()`. The `test_dataset` group has no YAML (only the registered schema), so it was unaffected. Setting the YAML `_target_` to `*Dataset` makes both merge orders agree.
- Added the two `*_pulsedb` configs documented above, plus the paper-recipe configs (`ppg2bp_ecg2bp_dual`, `train_pulsedb_refinement_pretrain_bp`, `refinement_trainer_mdvisco_pulsedb_dual`) and the `pretrain.sh`/`finetune.sh`/`train.sh` scripts.
- `src/trainers/trainer.py` — added a `load_weights_from: str | None = None` trainer field (dataclass + `__init__` + a branch in `run_training`) for cross-run **warm-start**: when set, rank-0 loads that explicit checkpoint file and applies **only** model weights via the existing `prepare_model_weights`/`load_from_checkpoint_dict` path (DDP broadcasts), leaving optimizer/scheduler/early-stopping fresh and the save path untouched. Needed because the stock load path (`checkpoint_mapping` → `"save"` manager) only supports same-run *resume*, not pretrain→finetune handoff. Composes/compiles + ruff-clean locally; the actual weight load is runtime-verified on the server (no local GPU).
- `src/trainers/trainer.py` — added **memory-optimization** trainer fields (dataclass + `__init__` + `_run_epoch`/`run_training` branches), all defaulting OFF so upstream behavior is unchanged:
  - `use_amp: bool` + `amp_dtype: str` ("bfloat16"|"float16") — wraps **only** the forward+loss (`_step_core`) of every stage in `torch.autocast(device_type="cuda", ...)`; CUDA-only (`amp_enabled = use_amp and device.type=="cuda"`, no-op on CPU/Mac). bf16 needs no loss scaling; fp16 routes backward through a `torch.amp.GradScaler` (built in `__init__`, `is_enabled()` only for fp16+CUDA, so bf16/off take the plain `loss.backward()` path). Near-lossless; big activation-memory saving + faster.
  - `use_gradient_checkpointing: bool` — new `_enable_gradient_checkpointing()` walks `self.model.modules()` and calls HF `gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})` (False required for DDP `find_unused_parameters=True`) on every submodule that supports it, **before** DDP wrap. Mathematically lossless; ~20-30% slower. ⚠️ In transformers 4.39.3 **only `DistilBertModel` supports it** (`supports_gradient_checkpointing=True`); `PatchTSMixerModel` does **not** (`False` → raises `ValueError`, caught + warned, skipped). So for refinement, checkpointing covers only the text branch; the PatchTSMixer **waveform** branch relies on bf16 AMP for its memory saving. The three `pretrain.sh`/`finetune.sh` scripts now pass `trainer.use_amp=true trainer.amp_dtype=bfloat16 trainer.use_gradient_checkpointing=true`. Logic verified in isolation (scaler/dtype/backward decision) + ruff-clean; full compose currently blocked on this Mac by a **numpy 2.4.4 ABI mismatch** in the `mdvisco` env (a C-extension built for numpy 1.x breaks `src.model.{mdvisco,patchtst}` import → `base_bp_model` registration; baseline w/o these flags fails identically) — runtime-verify on the server.
- `src/criterions/{l1_loss,mse_loss}.py` — added `**kwargs` forwarding to accept the inherited `enabled` field; `src/trainers/trainer.py::on_checkpoint_loaded` made a concrete no-op (was `@abstractmethod`, blocking `ScalarRegressionTrainer` instantiation).
- `src/dataset/base_dataset.py::BaseDataset` — derive `vitals_dataset` from `input_preprocessing` when none is configured (new `_derive_vitals_dataset`, called in `__init__`). No config or code anywhere (incl. upstream `main`) ever populates `vitals_dataset`, yet `train.py::create_datasets` **raises** `"<split>_dataset does not have vitals_dataset attribute"` whenever it is `None` — so the direction-capability gate was unpassable for every dataset. The derivation builds a `VitalsDataset` channel map from the union of `input_preprocessing.{source,target}` vitals; only vital **membership** is consumed (`supports_directions`), so indices are arbitrary-but-unique. Strictly additive (returns `None` when `input_preprocessing` is absent → original behaviour), matching the "channel layout derived from preprocessing-driven mapping" intent the dataset YAMLs already document.
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

#!/usr/bin/env bash
# Stage 2: BP scalar refinement on PulseDB, loading the stage-1 checkpoint.
# Edit the variables below to point at the stage-1 checkpoint and tweak the run.
set -euo pipefail

cd "$(dirname "$0")/../.."

# ----- Edit here -----------------------------------------------------------
# Path to the directory holding stage-1 checkpoints (output of script 01).
STAGE1_CKPT_DIR=/path/to/stage1/checkpoints

TRAINER=refinement_trainer_mdvisco
TRAIN_DATASET=train_pulsedb_refinement_bp

# Available DIRECTION values for BP scalar refinement (src/conf/directions/*.yaml):
#   Single-direction (DIRECTION_MODE=single):
#     ppg2bp                             # PPG -> SBP/DBP
#     ecg2bp                             # ECG -> SBP/DBP
#   Multi-direction (DIRECTION_MODE=multi):
#     ppg2bp_ecg2bp                      # PPG -> BP and ECG -> BP (default)
#
# Note: stage-2 targets BP scalars, so waveform directions like
#   ecg_ppg_abp_clinically_meaningful do NOT apply here.
DIRECTION=ppg2bp_ecg2bp
DIRECTION_MODE=multi
# ---------------------------------------------------------------------------

torchrun --standalone --nproc_per_node=1 --module src.train -m \
    trainer="${TRAINER}" \
    train_dataset="${TRAIN_DATASET}" \
    trainer.direction_mode="${DIRECTION_MODE}" \
    directions@trainer.directions="${DIRECTION}" \
    checkpoint_managers.load.base_dir="${STAGE1_CKPT_DIR}"

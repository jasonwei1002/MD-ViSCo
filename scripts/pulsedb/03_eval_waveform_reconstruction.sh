#!/usr/bin/env bash
# Evaluate stage-1 waveform reconstruction on PulseDB.
# Edit the variables below to match the training run you want to evaluate.
set -euo pipefail

cd "$(dirname "$0")/../.."

# ----- Edit here -----------------------------------------------------------
# CHECKPOINT_EPOCH must match the epoch you want to load.
# MODEL must match the architecture used at training time. Default evaluator
# yaml pins mdvisco_approximation_uci (UCI variant); override to match your
# stage-1 training run.
# DIRECTION / DIRECTION_MODE must match script 01's training config -- the
# checkpoint path embeds the direction tag, so a mismatch will fail to load.
CHECKPOINT_EPOCH=100
MODEL=mdvisco_approximation
EVALUATOR=waveform_reconstruction_evaluator
TEST_DATASET=test_pulsedb
DIRECTION=ecg_ppg_abp_clinically_meaningful
DIRECTION_MODE=multi
# ---------------------------------------------------------------------------

torchrun --standalone --nproc_per_node=1 --module src.test -m \
    evaluator="${EVALUATOR}" \
    test_dataset="${TEST_DATASET}" \
    model@evaluator.model="${MODEL}" \
    evaluator.load_model_weights=true \
    evaluator.checkpoint_epoch="${CHECKPOINT_EPOCH}" \
    evaluator.direction_mode="${DIRECTION_MODE}" \
    evaluator.directions="${DIRECTION}"

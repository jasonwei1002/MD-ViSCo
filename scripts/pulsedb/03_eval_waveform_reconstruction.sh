#!/usr/bin/env bash
# Evaluate stage-1 waveform reconstruction on PulseDB.
# Edit the variables below to match the training run you want to evaluate.
set -euo pipefail

cd "$(dirname "$0")/../.."

# ----- Edit here -----------------------------------------------------------
# CHECKPOINT_EPOCH:
#   - leave empty -> load the BEST checkpoint (suffix-free file saved when
#     val metric improves; see _handle_epoch_checkpointing in
#     waveform_reconstruction_trainer.py)
#   - integer (e.g. 100) -> load that periodic checkpoint
#     (requires save_checkpoint_frequency to have produced it)
# MODEL must match the architecture used at training time. Default evaluator
# yaml pins mdvisco_approximation_uci (UCI variant); override to match your
# stage-1 training run.
# DIRECTION / DIRECTION_MODE must match script 01's training config -- the
# checkpoint path embeds the direction tag, so a mismatch will fail to load.
CHECKPOINT_EPOCH=
MODEL=mdvisco_approximation
EVALUATOR=waveform_reconstruction_evaluator
TEST_DATASET=test_pulsedb
DIRECTION=ppg_ecg_multi_source
DIRECTION_MODE=single
# ---------------------------------------------------------------------------

EXTRA_OVERRIDES=()
if [[ -n "${CHECKPOINT_EPOCH}" ]]; then
    EXTRA_OVERRIDES+=("evaluator.checkpoint_epoch=${CHECKPOINT_EPOCH}")
fi

torchrun --standalone --nproc_per_node=1 --module src.test -m \
    evaluator="${EVALUATOR}" \
    test_dataset="${TEST_DATASET}" \
    model@evaluator.model="${MODEL}" \
    evaluator.load_model_weights=true \
    evaluator.direction_mode="${DIRECTION_MODE}" \
    directions@evaluator.directions="${DIRECTION}" \
    "${EXTRA_OVERRIDES[@]}"

#!/usr/bin/env bash
# Evaluate stage-2 BP scalar prediction on PulseDB.
# Edit the variables below to match the refinement training run.
set -euo pipefail

cd "$(dirname "$0")/../.."

# ----- Edit here -----------------------------------------------------------
# CHECKPOINT_EPOCH:
#   - leave empty -> load the BEST checkpoint (suffix-free file saved when
#     val metric improves; see _handle_epoch_checkpointing in
#     scalar_regression_trainer.py)
#   - integer (e.g. 60) -> load that periodic checkpoint
#     (requires save_checkpoint_frequency to have produced it)
# DIRECTION / DIRECTION_MODE must match script 02's training config.
CHECKPOINT_EPOCH=
EVALUATOR=blood_pressure_evaluator
TEST_DATASET=test_pulsedb_refinement_bp
DIRECTION=ppg2bp_ecg2bp
DIRECTION_MODE=multi
# ---------------------------------------------------------------------------

EXTRA_OVERRIDES=()
if [[ -n "${CHECKPOINT_EPOCH}" ]]; then
    EXTRA_OVERRIDES+=("evaluator.checkpoint_epoch=${CHECKPOINT_EPOCH}")
fi

torchrun --standalone --nproc_per_node=1 --module src.test -m \
    evaluator="${EVALUATOR}" \
    test_dataset="${TEST_DATASET}" \
    evaluator.load_model_weights=true \
    evaluator.direction_mode="${DIRECTION_MODE}" \
    directions@evaluator.directions="${DIRECTION}" \
    "${EXTRA_OVERRIDES[@]}"

#!/usr/bin/env bash
# Evaluate stage-2 BP scalar prediction on PulseDB.
# Edit the variables below to match the refinement training run.
set -euo pipefail

cd "$(dirname "$0")/../.."

# ----- Edit here -----------------------------------------------------------
# CHECKPOINT_EPOCH must match the epoch you want to load.
# DIRECTION / DIRECTION_MODE must match script 02's training config.
CHECKPOINT_EPOCH=60
EVALUATOR=blood_pressure_evaluator
TEST_DATASET=test_pulsedb_refinement_bp
DIRECTION=ppg2bp_ecg2bp
DIRECTION_MODE=multi
# ---------------------------------------------------------------------------

torchrun --standalone --nproc_per_node=1 --module src.test -m \
    evaluator="${EVALUATOR}" \
    test_dataset="${TEST_DATASET}" \
    evaluator.load_model_weights=true \
    evaluator.checkpoint_epoch="${CHECKPOINT_EPOCH}" \
    evaluator.direction_mode="${DIRECTION_MODE}" \
    evaluator.directions="${DIRECTION}"

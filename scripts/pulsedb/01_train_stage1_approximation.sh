#!/usr/bin/env bash
# Stage 1: Train multi-direction waveform reconstruction (approximation) on PulseDB.
# Edit the variables below to switch direction / mode / model.
set -euo pipefail

cd "$(dirname "$0")/../.."

# ----- Edit here -----------------------------------------------------------
# Available DIRECTION values on PulseDB (src/conf/directions/*.yaml):
#
#   Single-direction (set DIRECTION_MODE=single):
#     ppg2abp                            # PPG -> ABP
#     ecg2abp                            # ECG -> ABP
#     abp2ppg                            # ABP -> PPG
#     abp2ecg                            # ABP -> ECG
#     ppg2ecg                            # PPG -> ECG
#     ecg2ppg                            # ECG -> PPG
#     ppg_ecg_multi_source               # multi-source single-direction: [PPG,ECG] -> ABP
#
#   Multi-direction (set DIRECTION_MODE=multi):
#     ecg_ppg_abp                        # all 6 pairwise among ECG / PPG / ABP
#     ecg_ppg_abp_clinically_meaningful  # PPG->ECG, PPG->ABP, ECG->ABP (default)
#     ppg2abp_ecg2abp                    # PPG -> ABP and ECG -> ABP
#     ppg2ecg_ecg2ppg                    # PPG <-> ECG
#
#   Not supported on PulseDB:
#     *imp* directions (IMP channel only exists in UCI)
#     ppg2bp / ecg2bp / ppg2bp_ecg2bp    (BP scalar -> use stage-2 refinement)

DIRECTION=ppg_ecg_multi_source
DIRECTION_MODE=single
# Number of source vitals fed to the model in one forward pass. Must equal the
# largest `len(direction.source)` in the chosen DIRECTION (e.g. 1 for ppg2abp,
# 2 for ppg_ecg_multi_source). Models default to in_channels=1, so multi-source
# directions need an explicit override.
SOURCE_CHANNELS=2
# Per-rank batch size. Global batch = BATCH_SIZE * nproc_per_node.
# Checkpoint path embeds this value, so script 03 must use the same number.
BATCH_SIZE=256
# Optimizer learning rate. Checkpoint path embeds this value, so script 03
# must use the same number.
LEARNING_RATE=3e-3
# Trainer yaml already pins the matching model in its defaults list, so we do
# not pass a separate `model=` override here (Hydra rejects top-level model=).
# Switch trainer to swap models, e.g. approximation_trainer_patchtst.
TRAINER=approximation_trainer_mdvisco
# ---------------------------------------------------------------------------

torchrun --standalone --nproc_per_node=1 --module src.train -m \
    train_dataset=train_pulsedb \
    test_dataset=test_pulsedb \
    trainer="${TRAINER}" \
    trainer.direction_mode="${DIRECTION_MODE}" \
    trainer.model.in_channels="${SOURCE_CHANNELS}" \
    trainer.batch_size="${BATCH_SIZE}" \
    trainer.learning_rate="${LEARNING_RATE}" \
    directions@trainer.directions="${DIRECTION}"

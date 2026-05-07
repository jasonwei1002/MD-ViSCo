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
#
#   Multi-direction (set DIRECTION_MODE=multi):
#     ecg_ppg_abp                        # all 6 pairwise among ECG / PPG / ABP
#     ecg_ppg_abp_clinically_meaningful  # PPG->ECG, PPG->ABP, ECG->ABP (default)
#     ppg2abp_ecg2abp                    # PPG -> ABP and ECG -> ABP
#     ppg2ecg_ecg2ppg                    # PPG <-> ECG
#     ppg_ecg_multi_source               # multi-source -> single-target variants
#
#   Not supported on PulseDB:
#     *imp* directions (IMP channel only exists in UCI)
#     ppg2bp / ecg2bp / ppg2bp_ecg2bp    (BP scalar -> use stage-2 refinement)

DIRECTION=ppg_ecg_multi_source
DIRECTION_MODE=multi
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
    directions@trainer.directions="${DIRECTION}"

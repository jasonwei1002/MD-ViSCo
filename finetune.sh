#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Step 2/2 — FINETUNE BP refinement on CalFree_Test, warm-started from pretrain
# ============================================================================
# Finetunes on the PulseDB held-out test patients (CalFree_Test_Subset, 81/9/10),
# initializing the model from the Step-1 pretrain checkpoint via
# `trainer.load_weights_from` (loads ONLY model weights; optimizer/scheduler
# start fresh). The modality is chosen by SOURCE (see the SOURCE block below) and
# MUST match the SOURCE used in pretrain.sh — warm-start strict-loads the weights, so
# a mismatched architecture/direction would fail load_state_dict. Same WCL as pretrain.
# Same memory flags as pretrain: use_amp(bf16) + gradient_checkpointing.
#
# Usage (the pretrain checkpoint path is REQUIRED — no auto-discovery):
#   bash finetune.sh /path/to/PPG+ECG2BP_checkpoint_S_42_best.pt         # default SOURCE=joint
#   SOURCE=ppg bash finetune.sh /path/to/PPG2BP_checkpoint_S_42_best.pt  # single-source PPG
#   COLD_START=1 SOURCE=joint bash finetune.sh                          # no warm-start (from scratch)
# --- Prereqs:  conda activate mdvisco   (run from repo root)
# ============================================================================

DATA=/public/home/hs_mmcd_5/project/jasonwei/MD-ViSCo/data
WANDB_PROJECT=mdvisco-refinement
WANDB_ENTITY=jasonwei
LR=2.5e-4  # drives both the live optimizer.lr and the learning_rate path label

# SOURCE selects the training modality — MUST match the SOURCE used in pretrain.sh,
# because warm-start strict-loads the weights (same architecture/directions required):
#   joint -> PAPER HEADLINE: both PPG+ECG per sample, each branch supervised independently,
#            per-branch MAE summed; single-modality inference (trainer=..._pulsedb_joint).  [default]
#   ppg   -> single-source PPG only    (trainer=..._pulsedb_ppg,  direction [PPG]->BP,    single)
#   ecg   -> single-source ECG only    (trainer=..._pulsedb_ecg,  direction [ECG]->BP,    single)
#   multi -> multi-INPUT ablation (App. E): [PPG,ECG]->BP, predictions AVERAGED (trainer=..._pulsedb).
#   dual  -> ⚠ BROKEN with BPModel (mixed-batch mis-routing); use `joint`, not `dual`.
SOURCE="${SOURCE:-joint}"
case "$SOURCE" in
  joint) TRAINER=refinement_trainer_mdvisco_pulsedb_joint ;;
  ppg)   TRAINER=refinement_trainer_mdvisco_pulsedb_ppg   ;;
  ecg)   TRAINER=refinement_trainer_mdvisco_pulsedb_ecg   ;;
  multi) TRAINER=refinement_trainer_mdvisco_pulsedb       ;;
  dual)  TRAINER=refinement_trainer_mdvisco_pulsedb_dual  ;;
  *) echo "ERROR: invalid SOURCE='$SOURCE' (use joint | ppg | ecg | multi | dual)" >&2; exit 1 ;;
esac

WARMSTART_OVERRIDE=()
if [[ "${COLD_START:-0}" == "1" ]]; then
    echo "=== Step 2/2: FINETUNE (SOURCE=$SOURCE, trainer=$TRAINER, COLD START — no pretrain weights) ==="
else
    PRETRAIN_CKPT="${1:-}"
    if [[ -z "${PRETRAIN_CKPT}" ]]; then
        echo "ERROR: no pretrain checkpoint path given (first argument is required)." >&2
        echo "       Usage: bash finetune.sh /path/to/checkpoint_S_42_best.pt" >&2
        echo "       (or COLD_START=1 bash finetune.sh to train from scratch)" >&2
        exit 1
    fi
    if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
        echo "ERROR: pretrain checkpoint not found: ${PRETRAIN_CKPT}" >&2
        exit 1
    fi
    echo "=== Step 2/2: FINETUNE (SOURCE=$SOURCE, trainer=$TRAINER) warm-started from: ${PRETRAIN_CKPT} ==="
    WARMSTART_OVERRIDE=(trainer.load_weights_from="${PRETRAIN_CKPT}")
fi

torchrun --standalone --nproc_per_node=1 --module src.train -m \
    trainer="$TRAINER" \
    trainer.use_wcl=true \
    trainer.batch_size=512 \
    trainer.learning_rate="$LR" \
    trainer.optimizer.lr="$LR" \
    trainer.scheduler.patience=3 \
    trainer.early_stopping.patience=10 \
    trainer.use_amp=true \
    trainer.amp_dtype=bfloat16 \
    trainer.use_gradient_checkpointing=false \
    trainer.use_patient_information=true \
    trainer.overwrite_checkpoint=true \
    trainer.save_checkpoint_frequency=null \
    "${WARMSTART_OVERRIDE[@]}" \
    train_dataset=train_pulsedb_refinement_bp \
    test_dataset=test_pulsedb_refinement_bp \
    train_dataset.dataset_path="$DATA" \
    test_dataset.dataset_path="$DATA" \
    trainer.progress_bar.wandb_wrapper.project_name="$WANDB_PROJECT" \
    trainer.progress_bar.wandb_wrapper.entity="$WANDB_ENTITY"

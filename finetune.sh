#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Step 2/2 — FINETUNE BP refinement on CalFree_Test, warm-started from pretrain
# ============================================================================
# Finetunes on the PulseDB held-out test patients (CalFree_Test_Subset, 81/9/10),
# initializing the model from the Step-1 pretrain checkpoint via
# `trainer.load_weights_from` (loads ONLY model weights; optimizer/scheduler
# start fresh). Same SINGLE-SOURCE-JOINT directions (PPG->BP + ECG->BP,
# direction_mode=multi) + WCL as pretrain.
# Same memory flags as pretrain: use_amp(bf16) + gradient_checkpointing.
# (Multi-source [PPG,ECG]->BP ablation: set TRAINER=refinement_trainer_mdvisco_pulsedb — see CLAUDE.md.)
#
# Usage (the pretrain checkpoint path is REQUIRED — no auto-discovery):
#   bash finetune.sh /path/to/checkpoint_S_42_best.pt   # warm-start from this checkpoint
#   COLD_START=1 bash finetune.sh                        # no warm-start (train from scratch)
# --- Prereqs:  conda activate mdvisco   (run from repo root)
# ============================================================================

DATA=/public/home/hs_mmcd_5/project/jasonwei/MD-ViSCo/data
WANDB_PROJECT=mdvisco-refinement
WANDB_ENTITY=jasonwei
# MUST match the trainer used in pretrain.sh — warm-start strict-loads the weights.
# Single-source-joint (paper headline): refinement_trainer_mdvisco_pulsedb_dual
# Multi-source ablation ([PPG,ECG]->BP):  refinement_trainer_mdvisco_pulsedb
TRAINER=refinement_trainer_mdvisco_pulsedb_dual
LR=2.5e-4  # drives both the live optimizer.lr and the learning_rate path label

WARMSTART_OVERRIDE=()
if [[ "${COLD_START:-0}" == "1" ]]; then
    echo "=== Step 2/2: FINETUNE (COLD START — no pretrain weights) ==="
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
    echo "=== Step 2/2: FINETUNE warm-started from: ${PRETRAIN_CKPT} ==="
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

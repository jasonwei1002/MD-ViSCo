#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Step 2/2 — FINETUNE BP refinement on CalFree_Test, warm-started from pretrain
# ============================================================================
# Finetunes on the PulseDB held-out test patients (CalFree_Test_Subset, 81/9/10),
# initializing the model from the Step-1 pretrain checkpoint via
# `trainer.load_weights_from` (loads ONLY model weights; optimizer/scheduler
# start fresh). Same MULTI-SOURCE direction [PPG,ECG]->BP + WCL as pretrain.
# Same memory flags as pretrain: use_amp(bf16) + gradient_checkpointing.
# (Single-source-joint variant: swap trainer to *_pulsedb_dual — see CLAUDE.md.)
#
# Usage:
#   bash finetune.sh                       # uses ./weights/.last_pretrain_ckpt
#   bash finetune.sh /path/to/checkpoint_S_42.pt   # explicit pretrain checkpoint
#   COLD_START=1 bash finetune.sh          # no warm-start (train from scratch)
# --- Prereqs:  conda activate mdvisco   (run from repo root)
# ============================================================================

DATA=/public/home/hs_mmcd_5/project/jasonwei/MD-ViSCo/data
WANDB_PROJECT=mdvisco-refinement
WANDB_ENTITY=jasonwei

PRETRAIN_CKPT="${1:-$(cat ./weights/.last_pretrain_ckpt 2>/dev/null || true)}"

WARMSTART_OVERRIDE=()
if [[ "${COLD_START:-0}" == "1" ]]; then
    echo "=== Step 2/2: FINETUNE (COLD START — no pretrain weights) ==="
elif [[ -n "${PRETRAIN_CKPT}" ]]; then
    if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
        echo "ERROR: pretrain checkpoint not found: ${PRETRAIN_CKPT}" >&2
        echo "       run pretrain.sh first, or pass a valid path, or COLD_START=1." >&2
        exit 1
    fi
    echo "=== Step 2/2: FINETUNE warm-started from: ${PRETRAIN_CKPT} ==="
    WARMSTART_OVERRIDE=(trainer.load_weights_from="${PRETRAIN_CKPT}")
else
    echo "ERROR: no pretrain checkpoint. Usage: bash finetune.sh /path/to/checkpoint_S_42.pt" >&2
    echo "       (or run pretrain.sh first; or COLD_START=1 bash finetune.sh)" >&2
    exit 1
fi

torchrun --standalone --nproc_per_node=1 --module src.train -m \
    trainer=refinement_trainer_mdvisco_pulsedb \
    trainer.use_wcl=true \
    trainer.batch_size=1024 \
    trainer.use_amp=true \
    trainer.amp_dtype=bfloat16 \
    trainer.use_gradient_checkpointing=true \
    trainer.use_patient_information=true \
    trainer.overwrite_checkpoint=true \
    "${WARMSTART_OVERRIDE[@]}" \
    train_dataset=train_pulsedb_refinement_bp \
    test_dataset=test_pulsedb_refinement_bp \
    train_dataset.dataset_path="$DATA" \
    test_dataset.dataset_path="$DATA" \
    trainer.progress_bar.wandb_wrapper.project_name="$WANDB_PROJECT" \
    trainer.progress_bar.wandb_wrapper.entity="$WANDB_ENTITY"

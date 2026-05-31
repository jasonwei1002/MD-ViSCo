#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Step 1/2 — PRETRAIN BP refinement on Train_Subset (paper §III.E)
# ============================================================================
# Self-supervised + WCL pretraining of the refinement encoders on the PulseDB
# original train/val partition (Train_Subset, 80/20, no test split).
#   - MULTI-SOURCE direction [PPG,ECG]->BP (both signals in at once; direction_mode=single)
#   - is_pretraining=true + is_finetuning=false  => "pretraining" scenario
#   - use_wcl=true                                => weighted contrastive loss ON
# (For the single-source-joint variant instead, swap the trainer to
#  refinement_trainer_mdvisco_pulsedb_dual — see CLAUDE.md.)
#
# On success, writes the produced checkpoint path to ./weights/.last_pretrain_ckpt
# so finetune.sh can warm-start from it automatically.
# --- Prereqs:  conda activate mdvisco   (run from repo root)
# ============================================================================

DATA=/public/home/hs_mmcd_5/project/jasonwei/MD-ViSCo/data
WANDB_PROJECT=mdvisco-refinement
WANDB_ENTITY=jasonwei

echo "=== Step 1/2: PRETRAIN on Train_Subset ==="
torchrun --standalone --nproc_per_node=1 --module src.train -m \
    trainer=refinement_trainer_mdvisco_pulsedb \
    trainer.is_pretraining=true \
    trainer.is_finetuning=false \
    trainer.use_wcl=true \
    trainer.batch_size=1024 \
    trainer.use_patient_information=true \
    trainer.overwrite_checkpoint=true \
    train_dataset=train_pulsedb_refinement_pretrain_bp \
    test_dataset=test_pulsedb_refinement_bp \
    train_dataset.dataset_path="$DATA" \
    test_dataset.dataset_path="$DATA" \
    trainer.progress_bar.wandb_wrapper.project_name="$WANDB_PROJECT" \
    trainer.progress_bar.wandb_wrapper.entity="$WANDB_ENTITY"

# Discover the pretrain checkpoint (pretrain dir ends in _BP_NORM, NOT _finetuning).
# Leading wildcard catches the direction prefix: multi-source => PPG+ECG2BP_checkpoint_S_*.pt,
# single-source-joint (_dual) => checkpoint_S_*.pt.
CKPT=$(ls -t ./weights/MDViSCoRef/PulseDB/*_BP_NORM/*checkpoint_S_*.pt 2>/dev/null \
         | grep -v _finetuning | head -1 || true)
if [[ -n "${CKPT}" ]]; then
    echo "${CKPT}" > ./weights/.last_pretrain_ckpt
    echo "=== Pretrain checkpoint: ${CKPT}"
    echo "    (saved to ./weights/.last_pretrain_ckpt — finetune.sh will use it)"
else
    echo "WARNING: could not locate the pretrain checkpoint under ./weights/MDViSCoRef/PulseDB/*_BP_NORM/" >&2
    echo "         pass the path explicitly to finetune.sh." >&2
fi

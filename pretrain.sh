#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Step 1/2 — PRETRAIN BP refinement on Train_Subset (paper §III.E)
# ============================================================================
# Self-supervised + WCL pretraining of the refinement encoders on the PulseDB
# original train/val partition (Train_Subset, 80/20, no test split).
#   - SINGLE-SOURCE-JOINT directions PPG->BP + ECG->BP (two single-source branches
#       jointly trained, single-modality inference; direction_mode=multi) — paper §III.D headline
#   - is_pretraining=true + is_finetuning=false  => "pretraining" scenario
#   - use_wcl=true                                => weighted contrastive loss ON
#   - optimizer.lr 2.5e-4 + scheduler.patience=3 + early_stopping.patience=10
#       LR linearly scaled to BS 512 (paper: 1e-3 @ BS 2048; 1e-3 here diverged ~epoch 2).
#       LIVE knobs are trainer.optimizer.lr / trainer.scheduler.patience, NOT
#       trainer.learning_rate / trainer.scheduler_patience (those are checkpoint-path
#       labels only — kept equal below for an honest path). See CLAUDE.md "labels vs live knobs".
#   - use_amp(bf16) + gradient_checkpointing      => memory savings, near-lossless
#       (bf16 covers the PatchTSMixer waveform branch; checkpointing only the
#        DistilBERT text branch — PatchTSMixer lacks HF checkpointing support.)
# (To run the MULTI-SOURCE [PPG,ECG]->BP ablation instead, set
#  TRAINER=refinement_trainer_mdvisco_pulsedb below — see CLAUDE.md.)
#
# On success, prints the produced checkpoint path and the ready-to-run finetune
# command — pass that path explicitly to finetune.sh (no file handoff).
# --- Prereqs:  conda activate mdvisco   (run from repo root)
# ============================================================================

DATA=/public/home/hs_mmcd_5/project/jasonwei/MD-ViSCo/data
WANDB_PROJECT=mdvisco-refinement
WANDB_ENTITY=jasonwei
# Single-source-joint (paper headline): refinement_trainer_mdvisco_pulsedb_dual
# Multi-source ablation ([PPG,ECG]->BP):  refinement_trainer_mdvisco_pulsedb
TRAINER=refinement_trainer_mdvisco_pulsedb_dual
LR=2.5e-4  # drives both the live optimizer.lr and the learning_rate path label

echo "=== Step 1/2: PRETRAIN on Train_Subset (trainer=$TRAINER) ==="
torchrun --standalone --nproc_per_node=1 --module src.train -m \
    trainer="$TRAINER" \
    trainer.is_pretraining=true \
    trainer.is_finetuning=false \
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
    train_dataset=train_pulsedb_refinement_pretrain_bp \
    test_dataset=test_pulsedb_refinement_bp \
    train_dataset.dataset_path="$DATA" \
    test_dataset.dataset_path="$DATA" \
    trainer.progress_bar.wandb_wrapper.project_name="$WANDB_PROJECT" \
    trainer.progress_bar.wandb_wrapper.entity="$WANDB_ENTITY"

# Discover the pretrain checkpoint (pretrain dir ends in _BP_NORM, NOT _finetuning).
# Best-model file ends in _best.pt (periodic saving is off); leading wildcard catches
# the direction prefix: multi-source => PPG+ECG2BP_checkpoint_S_*_best.pt,
# single-source-joint (_dual) => checkpoint_S_*_best.pt.
CKPT=$(ls -t ./weights/MDViSCoRef/PulseDB/*_BP_NORM/*checkpoint_S_*_best.pt 2>/dev/null \
         | grep -v _finetuning | head -1 || true)
if [[ -n "${CKPT}" ]]; then
    echo "=== Pretrain checkpoint: ${CKPT}"
    echo "    Next: bash finetune.sh \"${CKPT}\""
else
    echo "WARNING: could not locate the pretrain checkpoint under ./weights/MDViSCoRef/PulseDB/*_BP_NORM/" >&2
    echo "         pass the path explicitly to finetune.sh." >&2
fi

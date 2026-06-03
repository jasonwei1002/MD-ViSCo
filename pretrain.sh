#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Step 1/2 — PRETRAIN BP refinement on Train_Subset (paper §III.E)
# ============================================================================
# Self-supervised + WCL pretraining of the refinement encoders on the PulseDB
# original train/val partition (Train_Subset, 80/20, no test split).
#   - modality chosen by SOURCE (see the SOURCE block below): joint | ppg | ecg | multi | dual.
#       joint (default, PAPER HEADLINE) = both PPG+ECG per sample, each branch supervised
#         INDEPENDENTLY, per-branch MAE SUMMED (L_MAE over i in {ECG,PPG}); single-modality infer.
#       ppg/ecg = TRUE single-source (one encoder, that signal only).
#       multi = App.-E ablation: [PPG,ECG]->BP, predictions AVERAGED (both required at infer).
#       dual  = ⚠ BROKEN with BPModel (mixed-batch mis-routing) — kept only for reference.
#   - is_pretraining=true + is_finetuning=false  => "pretraining" scenario
#   - use_wcl=true                                => weighted contrastive loss ON
#       (single-source keeps only its own `<vital>_*` + `text_*` WCL terms; the
#        absent modality's terms self-skip on the missing embedding key.)
#   - optimizer.lr 2.5e-4 + scheduler.patience=3 + early_stopping.patience=10
#       LR linearly scaled to BS 512 (paper: 1e-3 @ BS 2048; 1e-3 here diverged ~epoch 2).
#       LIVE knobs are trainer.optimizer.lr / trainer.scheduler.patience, NOT
#       trainer.learning_rate / trainer.scheduler_patience (those are checkpoint-path
#       labels only — kept equal below for an honest path). See CLAUDE.md "labels vs live knobs".
#   - use_amp(bf16) + gradient_checkpointing      => memory savings, near-lossless
#       (bf16 covers the PatchTSMixer waveform branch; checkpointing only the
#        DistilBERT text branch — PatchTSMixer lacks HF checkpointing support.)
#
# Usage:
#   bash pretrain.sh                  # default SOURCE=joint (paper headline)
#   SOURCE=ppg bash pretrain.sh       # single-source PPG only
#   SOURCE=ecg bash pretrain.sh       # single-source ECG only
# On success, prints the produced checkpoint path and the ready-to-run finetune
# command (with the SAME SOURCE) — pass that path explicitly to finetune.sh.
# --- Prereqs:  conda activate mdvisco   (run from repo root)
# ============================================================================

DATA=/public/home/hs_mmcd_5/project/jasonwei/MD-ViSCo/data
WANDB_PROJECT=mdvisco-refinement
WANDB_ENTITY=jasonwei
LR=2.5e-4  # drives both the live optimizer.lr and the learning_rate path label

# SOURCE selects the training modality (export SOURCE=... or pass inline):
#   joint -> PAPER HEADLINE (§III.D): both PPG+ECG per sample, each branch supervised
#            INDEPENDENTLY, per-branch MAE SUMMED; single-modality inference.   [default]
#            (trainer=..._pulsedb_joint: direction [PPG,ECG]->BP + per-vital summed-MAE loss)
#   ppg   -> single-source PPG only    (trainer=..._pulsedb_ppg,  direction [PPG]->BP,    single)
#   ecg   -> single-source ECG only    (trainer=..._pulsedb_ecg,  direction [ECG]->BP,    single)
#   multi -> multi-INPUT ablation (App. E): [PPG,ECG]->BP, predictions AVERAGED, both
#            signals required at inference (trainer=..._pulsedb).
#   dual  -> ⚠ BROKEN with BPModel: direction_mode=multi makes mixed batches that
#            BPModel mis-routes (keys whole batch off directions[0] -> cross-modality
#            contamination; supports_multi_directional=False). Use `joint`, not `dual`.
# joint/multi keep BOTH PPG+ECG encoders; ppg/ecg are TRUE single-encoder models.
# CKPT_PREFIX is the checkpoint filename prefix (the direction key) used to disambiguate
# the discovery glob below. NOTE: joint and multi share the SAME prefix AND subfolder
# (same model/hyperparams/direction) -> do NOT run both into the same weights dir.
SOURCE="${SOURCE:-joint}"
case "$SOURCE" in
  joint) TRAINER=refinement_trainer_mdvisco_pulsedb_joint; CKPT_PREFIX="PPG+ECG2BP_" ;;
  ppg)   TRAINER=refinement_trainer_mdvisco_pulsedb_ppg;   CKPT_PREFIX="PPG2BP_" ;;
  ecg)   TRAINER=refinement_trainer_mdvisco_pulsedb_ecg;   CKPT_PREFIX="ECG2BP_" ;;
  multi) TRAINER=refinement_trainer_mdvisco_pulsedb;       CKPT_PREFIX="PPG+ECG2BP_" ;;
  dual)  TRAINER=refinement_trainer_mdvisco_pulsedb_dual;  CKPT_PREFIX="" ;;
  *) echo "ERROR: invalid SOURCE='$SOURCE' (use joint | ppg | ecg | multi | dual)" >&2; exit 1 ;;
esac

echo "=== Step 1/2: PRETRAIN on Train_Subset (SOURCE=$SOURCE, trainer=$TRAINER) ==="
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
# Best-model file ends in _best.pt (periodic saving is off). The filename prefix IS the
# direction key, so match the SOURCE-specific ${CKPT_PREFIX} to avoid picking another
# modality's checkpoint (all variants share the same _BP_NORM dir; only the prefix
# differs): ppg => PPG2BP_, ecg => ECG2BP_, multi => PPG+ECG2BP_, dual => "" (no prefix).
CKPT=$(ls -t ./weights/MDViSCoRef/PulseDB/*_BP_NORM/${CKPT_PREFIX}checkpoint_S_*_best.pt 2>/dev/null \
         | grep -v _finetuning | head -1 || true)
if [[ -n "${CKPT}" ]]; then
    echo "=== Pretrain checkpoint: ${CKPT}"
    echo "    Next: SOURCE=$SOURCE bash finetune.sh \"${CKPT}\""
else
    echo "WARNING: could not locate '${CKPT_PREFIX}checkpoint_S_*_best.pt' under ./weights/MDViSCoRef/PulseDB/*_BP_NORM/" >&2
    echo "         pass the path explicitly: SOURCE=$SOURCE bash finetune.sh <ckpt>" >&2
fi

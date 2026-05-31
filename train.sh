#!/usr/bin/env bash
set -euo pipefail

# --- Prereqs ---
# conda activate mdvisco

DATA=/public/home/hs_mmcd_5/project/jasonwei/MD-ViSCo/data

torchrun --standalone --nproc_per_node=1 --module src.train -m \
    trainer=refinement_trainer_mdvisco_pulsedb \
    trainer.use_patient_information=true \
    train_dataset=train_pulsedb_refinement_bp \
    test_dataset=test_pulsedb_refinement_bp \
    train_dataset.dataset_path="$DATA" \
    test_dataset.dataset_path="$DATA" \
    trainer.progress_bar.wandb_wrapper.project_name=mdvisco-refinement \
    trainer.progress_bar.wandb_wrapper.entity=jasonwei

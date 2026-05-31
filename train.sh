#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# MD-ViSCo stage-2 BP refinement — FULL paper recipe (pretrain -> finetune)
# ============================================================================
# Convenience orchestrator: runs pretrain.sh then finetune.sh. The finetune step
# warm-starts from the pretrain checkpoint (pretrain.sh writes its path to
# ./weights/.last_pretrain_ckpt, finetune.sh reads it).
#
# Run the two stages independently if you prefer:
#   bash pretrain.sh
#   bash finetune.sh [/path/to/checkpoint_S_42.pt]
#
# See CLAUDE.md "Paper-faithful BP refinement recipe" for details.
# --- Prereqs:  conda activate mdvisco   (run from repo root)
# ============================================================================

HERE="$(cd "$(dirname "$0")" && pwd)"

bash "${HERE}/pretrain.sh"
bash "${HERE}/finetune.sh"

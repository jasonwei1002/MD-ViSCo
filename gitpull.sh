#!/usr/bin/env bash
# One-click: force local code to EXACTLY match origin/<branch>.
#
# Discards local changes AND local commits to git-TRACKED files.
# Does NOT touch untracked / .gitignored files — your data and results are safe:
#   data/, outputs/, multirun/, weights/, wandb/, *.mat, *.h5, checkpoints ...
# (Recovery: a hard reset is still in `git reflog` if you discarded something by mistake.)
#
# Usage:
#   bash gitpull.sh          # sync to origin/main (default)
#   bash gitpull.sh dev      # sync to origin/dev
set -euo pipefail

BRANCH="${1:-main}"
# Remote URL the SERVER uses to reach GitHub (gh-proxy mirror to bypass GitHub
# access restrictions). This mirror is for READ (fetch/pull) only — push from a
# machine that uses the direct github.com URL.
# Override if needed:  MDVISCO_REMOTE=<url> bash gitpull.sh
REMOTE_URL="${MDVISCO_REMOTE:-https://gh-proxy.org/https://github.com/jasonwei1002/MD-ViSCo.git}"

# Operate on the repo this script lives in, regardless of where it's invoked from.
cd "$(cd "$(dirname "$0")" && pwd)"

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "ERROR: not a git repository: $(pwd)" >&2
    exit 1
}

echo ">> repo:   $(pwd)"
echo ">> before: $(git rev-parse --short HEAD 2>/dev/null || echo '(none)') on $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"

# Point origin at the (proxied) remote so fetch works behind GitHub restrictions.
if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$REMOTE_URL"
else
    git remote add origin "$REMOTE_URL"
fi
echo ">> origin: $(git remote get-url origin)"

echo ">> fetching origin ..."
git fetch --prune origin

echo ">> WARNING: discarding local changes/commits on tracked files; matching origin/${BRANCH}"
git checkout -f -B "${BRANCH}" "origin/${BRANCH}"
git reset --hard "origin/${BRANCH}"

echo ">> after:  $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"
echo ">> done. Code now matches origin/${BRANCH}. Untracked/ignored data was NOT removed."

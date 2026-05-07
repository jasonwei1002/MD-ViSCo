#!/usr/bin/env bash
# Sync server with GitHub. Run on the server.
# ⚠ Force-aligns to origin/main; local uncommitted changes will be discarded.
set -e

REPO_URL="https://gh-proxy.org/https://github.com/jasonwei1002/MD-ViSCo.git"
REPO_DIR="$HOME/project/jasonwei/MD-ViSCo"
BRANCH="main"

if [ -d "${REPO_DIR}/.git" ]; then
    cd "${REPO_DIR}"
    git fetch origin "${BRANCH}"
    git reset --hard "origin/${BRANCH}"
    git clean -fd
else
    git clone --branch "${BRANCH}" "${REPO_URL}" "${REPO_DIR}"
fi

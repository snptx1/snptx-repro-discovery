#!/usr/bin/env bash
# Pillar-1 GPU launch (A10G). Runs inside tmux so it survives disconnects.
# Resumable: re-run with --resume appended to skip completed seed x fraction units.
set -euo pipefail
cd /home/snptx/snptx-core
source .venv/bin/activate
export PYTHONPATH=src:.
mkdir -p pilot_phd/preprint/dl_forward/artifacts
echo "START $(date -u +%FT%TZ)  device=$(python -c 'import torch;print(torch.cuda.get_device_name(0))')"
python pilot_phd/preprint/dl_forward/train_multitask_gnn.py "$@"
echo "END   $(date -u +%FT%TZ)"

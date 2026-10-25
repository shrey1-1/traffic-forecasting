#!/bin/bash
# Run a project script on the GPU inside WSL2. From Windows, in the project folder:
#   wsl -d Ubuntu-24.04 -u root -- bash scripts/wsl/run.sh scripts/07_features.py --only abilene:6
# Uses the Linux environment /root/tfgpu (TensorFlow 2.21 with CUDA, same library versions as
# requirements.txt); see "GPU training (optional)" in README.md for how to create it.
. "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/../.."
exec /root/tfgpu/bin/python -u "$@"

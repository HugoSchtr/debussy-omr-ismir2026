#!/bin/bash
#SBATCH --job-name=ismir_cr_zeus_olimpic
#SBATCH --output=/home/hscheith/dev/ISMIR2026/code/zero_shot_eval/zeus/results/slurm_olimpic_%j.out
#SBATCH --error=/home/hscheith/dev/ISMIR2026/code/zero_shot_eval/zeus/results/slurm_olimpic_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --account=inria
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --exclude=gpu004,gpu011,gpu012,gpu013,gpu014,gpu015,gpu017,gpu018

# ISMIR camera-ready T1.3 re-run: Zeus (Synthetic/OLiMPiC), corrected binarize=none default
# (was wrongly `adaptive` in the original submission). Reproduces the Table 5 "Zeus / OLiMPiC"
# row against the full 561-sample/37-doc Debussy corpus (paired_system_level_v2), matching the
# paper's own stated system-level protocol and the SMT rows already in the table -- see the
# sibling rerun_camera_grandstaff.sh for why --exclude-documents is deliberately NOT passed.
set -e
REPO_DIR="/home/hscheith/dev/olimpic-icdar24/github/olimpic-icdar24"
CODE_DIR="/home/hscheith/dev/ISMIR2026/code/zero_shot_eval/zeus"
DATASET="/home/hscheith/dev/smt-finetune/paired_system_level_v2"

export TF_USE_LEGACY_KERAS=1
export OLIMPIC_ICDAR24_DIR="${REPO_DIR}"
# eval_zeus.py shells out to `python3 zeus.py` (not sys.executable), so the venv's bin must be
# first on PATH or the subprocess falls back to system python3, which has no tensorflow -- see
# sibling rerun_camera_grandstaff.sh for the full note.
export PATH="${REPO_DIR}/.venv/bin:${PATH}"
NVIDIA_DIR=$("${REPO_DIR}/.venv/bin/python3" -c "import nvidia; print(nvidia.__path__[0])" 2>/dev/null || true)
if [ -n "$NVIDIA_DIR" ]; then
    export LD_LIBRARY_PATH=$(find "$NVIDIA_DIR" -name "lib" -type d | tr '\n' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
LIBCUDA=$(find /usr/lib64 /lib64 /usr/lib -name "libcuda.so.1" 2>/dev/null | head -1)
if [ -n "$LIBCUDA" ]; then
    export LD_LIBRARY_PATH="$(dirname "$LIBCUDA"):${LD_LIBRARY_PATH}"
fi
nvidia-smi || echo "WARNING: nvidia-smi failed"

"${REPO_DIR}/.venv/bin/python3" "${CODE_DIR}/eval_zeus.py" \
    --model /scratch/hscheith/olimpic-icdar24/models/zeus-olimpic-1.0-2024-02-12.model \
    --dataset "${DATASET}" \
    --output "${CODE_DIR}/results/zeus_olimpic" \
    --zeus-script-dir "${REPO_DIR}/zeus" \
    --compute-tedn --tedn-flavor lmx

#!/bin/bash
#SBATCH --job-name=ismir_cr_zeus_camgs
#SBATCH --output=/home/hscheith/dev/ISMIR2026/code/zero_shot_eval/zeus/results/slurm_camgs_%j.out
#SBATCH --error=/home/hscheith/dev/ISMIR2026/code/zero_shot_eval/zeus/results/slurm_camgs_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --account=inria
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --exclude=gpu004,gpu011,gpu012,gpu013,gpu014,gpu015,gpu017,gpu018

# ISMIR camera-ready T1.3 re-run: Zeus (Camera-GrandStaff-LMX), corrected binarize=none default
# (was wrongly `adaptive` in the original submission -- see eval_zeus.py's own docstring and
# REVISION_PLAN.md T1.3). Reproduces the Table 5 "Zeus / Cam. GrandStaff LMX" row against the
# full 561-sample/37-doc Debussy corpus (paired_system_level_v2), matching the paper's own
# stated system-level protocol ("System-level experiments use all 561 samples across the 37
# documents", paper-6p-cr.tex L297) and the SMT rows already in the same table (confirmed via
# their archived config_used.yaml: exclude_documents: null). No --exclude-documents here --
# cluster_evaluation/config.yaml's 6-document exclude list was NOT actually applied to the run
# that produced the published SMT numbers (config drifted after that run), so applying it here
# would silently mismatch the rest of Table 5.
set -e
REPO_DIR="/home/hscheith/dev/olimpic-icdar24/github/olimpic-icdar24"
CODE_DIR="/home/hscheith/dev/ISMIR2026/code/zero_shot_eval/zeus"
DATASET="/home/hscheith/dev/smt-finetune/paired_system_level_v2"

export TF_USE_LEGACY_KERAS=1
export OLIMPIC_ICDAR24_DIR="${REPO_DIR}"
# eval_zeus.py shells out to `python3 zeus.py` (not sys.executable), so the venv's bin must be
# first on PATH or the subprocess falls back to system python3, which has no tensorflow --
# this is exactly what made jobs 5131814/5131815 fail in ~5 minutes on 2026-07-22, caught only
# because their .err logs were checked the next morning rather than assumed clean.
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
    --model /scratch/hscheith/olimpic-icdar24/models/zeus-camera-grandstaff-lmx-1.0-2024-02-12.model \
    --dataset "${DATASET}" \
    --output "${CODE_DIR}/results/zeus_camera_grandstaff_lmx" \
    --zeus-script-dir "${REPO_DIR}/zeus" \
    --compute-tedn --tedn-flavor lmx

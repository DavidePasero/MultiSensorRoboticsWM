#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --job-name=SPLIT_H5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=03:00:00
#SBATCH --output=out_job_dir/SPLIT_H5_%A.out

set -euo pipefail

export MUJOCO_GL=egl

cd MultiSensorRoboticsWM/

source .venv/bin/activate

SRC="${SRC:-/home/dpasero/project_space/metaworld_bin_picking.h5}"
DST_A="${DST_A:-/home/dpasero/project_space/metaworld_bin_picking_70.h5}"
DST_B="${DST_B:-/home/dpasero/project_space/metaworld_bin_picking_30.h5}"
FRACTION="${FRACTION:-0.7}"
SEED="${SEED:-42}"

echo "SRC=$SRC"
echo "DST_A=$DST_A"
echo "DST_B=$DST_B"
echo "FRACTION=$FRACTION"
echo "SEED=$SEED"

srun .venv/bin/python datasets_utils/split_hdf5_by_episode_fraction.py \
  "$SRC" \
  "$DST_A" \
  "$DST_B" \
  --fraction "$FRACTION" \
  --shuffle \
  --seed "$SEED"

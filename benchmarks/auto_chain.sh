#!/bin/bash
# auto_chain.sh
# Waits for an active Slurm job to finish, backs up data, and submits next job.

TARGET_JOB=${1:-19599793}
NEXT_SCRIPT=${2:-slurm_frontier_spins.sh}

echo "======================================================="
echo " AUTO-CHAIN WATCHER STARTED"
echo " Watching Job ID : $TARGET_JOB"
echo " Next to launch  : $NEXT_SCRIPT"
echo "======================================================"

# 1. Wait until TARGET_JOB disappears from squeue
while squeue -j "$TARGET_JOB" 2>/dev/null | grep -q "$TARGET_JOB"; do
    echo "[$(date) ] Job $TARGET_JOB is still running. Checking again in 60s..."
    sleep 60
done

echo "[$(date) ] Job $TARGET_JOB has completed!"

# 2. Backup newly generated Result 4 data
REPO_DIR="/usr/people/roib/sayak/qutip-bundling-landau"
BACKUP_DIR="$HOME/r4_backup_$(date +%Y_%m_%d_%H5%M%S)"

echo "Creating data backup in $BACKUP_DIR ..."
mkdir -p "$BACKUP_DIR"
cp -r "$REPO_DIR/benchmarks/data/"* "$BACKUP_DIR/" 2>/dev/null || true

# 3. Launch the next job
cd "$REPO_DIR/benchmarks"
echo "Submitting next job: $NEXT_SCRIPT ..."
sbatch "$NEXT_SCRIPT"

echo "Done! New job submitted successfully."

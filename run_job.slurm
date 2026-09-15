#!/bin/bash
#SBATCH --partition=cpu_normal_q
#SBATCH --nodes=5
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=myjob
#SBATCH --output=/scratch/<your_username>/<project_folder>/output.txt

# Exit immediately if a command fails
set -e

# Load required modules
module load Anaconda3/2021.11
module load git

# Initialize conda and activate your environment
eval "$(conda shell.bash hook)"
conda activate <your_conda_env>  # Replace with your actual environment name

# Navigate to your project directory
cd /scratch/<your_username>/<project_folder>

# Set the number of processes
process_num=5

# Run your MPI Python program
mpiexec -n $process_num python DVD.py --data base.1B.fbin.crop_nb_100000000

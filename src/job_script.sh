#!/bin/bash
# Job name:
#SBATCH --job-name=test_run
#
# Account:
#SBATCH --account=fc_mhmm
#
# Partition:
#SBATCH --partition=savio2
#
# QoS:
#SBATCH --qos=savio_normal
#
# Request one node:
#SBATCH --nodes=1
#
# Specify number of tasks for use case (example):
#SBATCH --ntasks-per-node=24
#
# Processors per task:
#SBATCH --cpus-per-task=1
#
# Wall clock limit:
#SBATCH --time=00:00:30
#
## Command(s) to run (example):
./repos/cpHMM/src/cp_test_ht.py
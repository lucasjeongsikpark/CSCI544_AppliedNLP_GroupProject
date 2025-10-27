#!/bin/bash
# Script to run Medical dataset evaluation experiment

cd "$(dirname "$0")/.."

echo "Running Medical Dataset Evaluation Experiment..."
python src/dataset_experiment.py --config src/config_med.json

echo "Medical experiment completed!"

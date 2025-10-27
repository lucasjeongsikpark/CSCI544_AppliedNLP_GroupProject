#!/bin/bash
# Script to run math dataset evaluation experiment

cd "$(dirname "$0")/.."

echo "Running Math Dataset Evaluation Experiment..."
python src/dataset_experiment.py --config src/config_math.json

echo "Math experiment completed!"

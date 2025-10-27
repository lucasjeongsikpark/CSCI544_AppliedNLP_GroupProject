#!/bin/bash
# Script to run OpenQA dataset evaluation experiment

cd "$(dirname "$0")/.."

echo "Running OpenQA Dataset Evaluation Experiment..."
python src/dataset_experiment.py --config src/config_openqa.json

echo "OpenQA experiment completed!"

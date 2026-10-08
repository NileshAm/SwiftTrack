#!/usr/bin/env bash
set -e
/home/nilesh/miniconda3/envs/dragon-track/bin/python /home/nilesh/UOM/CV/Project/ModifiedCode/detect.py
/home/nilesh/miniconda3/envs/dragon-track/bin/python /home/nilesh/UOM/CV/Project/ModifiedCode/train.py
/home/nilesh/miniconda3/envs/dragon-track/bin/python /home/nilesh/UOM/CV/Project/ModifiedCode/tracking.py
/home/nilesh/miniconda3/envs/dragon-track/bin/python /home/nilesh/UOM/CV/Project/ModifiedCode/evaluate.py

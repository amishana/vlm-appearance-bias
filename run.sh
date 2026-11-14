#!/bin/bash
# baseline set up
BACKEND=qwen2_5_vl MODEL=YOUR/PATH/TO/models/Qwen2.5-VL-3B-Instruct bash shellscripts/check_baseline.sh
BACKEND=smolvlm2   MODEL=YOUR/PATH/TO/models/SmolVLM2-2.2B-Instruct  bash shellscripts/check_baseline.sh
BACKEND=internvl3  MODEL=YOUR/PATH/TO/models/InternVL3-8B           bash shellscripts/check_baseline.sh

#Projection method
BACKEND=qwen2_5_vl MODEL=~/Documents/models/Qwen2.5-VL-3B-Instruct bash projection/run_project_local.sh
BACKEND=smolvlm2   MODEL=~/Documents/models/SmolVLM2-2.2B-Instruct  bash projection/run_project_local.sh
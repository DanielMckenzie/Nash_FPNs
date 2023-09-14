#!/bin/bash
python_script = "Generate_Data.py"

for i in {60..90..10}; do
    python3 Generate_Data.py \
        --action_size "$i"
        echo "$i"
done
#!/bin/bash
source activate base
echo ${BASH_ARGV}
python -u lib/run/main.py ${BASH_ARGV}


#!/bin/bash
source activate base
echo ${BASH_ARGV}
python -u main.py ${BASH_ARGV}


#!/bin/sh
nohup python run_experiment_objectworld.py -e $1 > "exp-$1.log" 2>&1 &

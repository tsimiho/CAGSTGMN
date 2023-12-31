#!/bin/bash

python3 run_diffpool.py --dataset $1
python3 run_mincut.py --dataset $1

python3 run_nmf.py --dataset $1
python3 run_lapool.py --dataset $1

python3 run_topk.py --dataset $1
python3 run_sagpool.py --dataset $1

python3 run_graclus.py --dataset $1
python3 run_ndp.py --dataset $1
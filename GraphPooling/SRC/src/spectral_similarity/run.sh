#!/bin/bash

python3 run_diffpool.py --name $1
python3 run_mincut.py --name $1

python3 run_nmf.py --name $1
python3 run_lapool.py --name $1

python3 run_topk.py --name $1
python3 run_sagpool.py --name $1

python3 run_ndp.py --name $1
python3 run_graclus.py --name $1
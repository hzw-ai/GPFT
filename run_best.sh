#!/bin/bash

# set the experiment GPU by the first parameters
export CUDA_VISIBLE_DEVICES=0
# activate your experiment environment
#source activate STSSL

# set the dataset by the second parameters
python train_p.py --datasets=NYCBike1
python train_p.py --datasets=NYCBike1
python train_p.py --datasets=NYCBike1
python train_p.py --datasets=NYCBike1
python train_p.py --datasets=NYCBike1

python train_p.py --datasets=NYCBike2
python train_p.py --datasets=NYCBike2
python train_p.py --datasets=NYCBike2
python train_p.py --datasets=NYCBike2
python train_p.py --datasets=NYCBike2

python train_p.py --datasets=NYCTaxi
python train_p.py --datasets=NYCTaxi
python train_p.py --datasets=NYCTaxi
python train_p.py --datasets=NYCTaxi
python train_p.py --datasets=NYCTaxi

python train_p.py --datasets=BJTaxi
python train_p.py --datasets=BJTaxi
python train_p.py --datasets=BJTaxi
python train_p.py --datasets=BJTaxi
python train_p.py --datasets=BJTaxi
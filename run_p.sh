#!/bin/bash

# set the experiment GPU by the first parameters
export CUDA_VISIBLE_DEVICES=0
# activate your experiment environment
#source activate STSSL

# set the dataset by the second parameters
#python train_p.py --datasets=NYCBike1 --prompt=SimplePrompt
#python train_p.py --datasets=NYCBike1 --prompt=GPFplusAtt --p_num=1
#python train_p.py --datasets=NYCBike1 --prompt=GPFplusAtt --p_num=2
#python train_p.py --datasets=NYCBike1 --prompt=GPFplusAtt --p_num=3
#python train_p.py --datasets=NYCBike1 --prompt=GPFplusAtt --p_num=4
#python train_p.py --datasets=NYCBike1 --prompt=GPFplusAtt --p_num=5
#python train_p.py --datasets=NYCBike1 --prompt=GPFplusAtt --p_num=6
#python train_p.py --datasets=NYCBike1 --prompt=GPFplusAtt --p_num=7
#python train_p.py --datasets=NYCBike1 --prompt=GPFplusAtt --p_num=8
#python train_p.py --datasets=NYCBike1 --prompt=GPFplusAtt --p_num=9
#python train_p.py --datasets=NYCBike1 --prompt=GPFplusAtt --p_num=10
#
#python train_p.py --datasets=NYCBike2 --prompt=SimplePrompt
#python train_p.py --datasets=NYCBike2 --prompt=GPFplusAtt --p_num=1
#python train_p.py --datasets=NYCBike2 --prompt=GPFplusAtt --p_num=2
#python train_p.py --datasets=NYCBike2 --prompt=GPFplusAtt --p_num=3
#python train_p.py --datasets=NYCBike2 --prompt=GPFplusAtt --p_num=4
#python train_p.py --datasets=NYCBike2 --prompt=GPFplusAtt --p_num=5
#python train_p.py --datasets=NYCBike2 --prompt=GPFplusAtt --p_num=6
#python train_p.py --datasets=NYCBike2 --prompt=GPFplusAtt --p_num=7
#python train_p.py --datasets=NYCBike2 --prompt=GPFplusAtt --p_num=8
#python train_p.py --datasets=NYCBike2 --prompt=GPFplusAtt --p_num=9
#python train_p.py --datasets=NYCBike2 --prompt=GPFplusAtt --p_num=10
#
#python train_p.py --datasets=NYCTaxi --prompt=SimplePrompt
#python train_p.py --datasets=NYCTaxi --prompt=GPFplusAtt --p_num=1
#python train_p.py --datasets=NYCTaxi --prompt=GPFplusAtt --p_num=2
#python train_p.py --datasets=NYCTaxi --prompt=GPFplusAtt --p_num=3
#python train_p.py --datasets=NYCTaxi --prompt=GPFplusAtt --p_num=4
#python train_p.py --datasets=NYCTaxi --prompt=GPFplusAtt --p_num=5
#python train_p.py --datasets=NYCTaxi --prompt=GPFplusAtt --p_num=6
#python train_p.py --datasets=NYCTaxi --prompt=GPFplusAtt --p_num=7
#python train_p.py --datasets=NYCTaxi --prompt=GPFplusAtt --p_num=8
#python train_p.py --datasets=NYCTaxi --prompt=GPFplusAtt --p_num=9
#python train_p.py --datasets=NYCTaxi --prompt=GPFplusAtt --p_num=10
#
#python train_p.py --datasets=BJTaxi --prompt=SimplePrompt
#python train_p.py --datasets=BJTaxi --prompt=GPFplusAtt --p_num=1
#python train_p.py --datasets=BJTaxi --prompt=GPFplusAtt --p_num=2
#python train_p.py --datasets=BJTaxi --prompt=GPFplusAtt --p_num=3
#python train_p.py --datasets=BJTaxi --prompt=GPFplusAtt --p_num=4
#python train_p.py --datasets=BJTaxi --prompt=GPFplusAtt --p_num=5
#python train_p.py --datasets=BJTaxi --prompt=GPFplusAtt --p_num=6
#python train_p.py --datasets=BJTaxi --prompt=GPFplusAtt --p_num=7
#python train_p.py --datasets=BJTaxi --prompt=GPFplusAtt --p_num=8
#python train_p.py --datasets=BJTaxi --prompt=GPFplusAtt --p_num=9
#python train_p.py --datasets=BJTaxi --prompt=GPFplusAtt --p_num=10
#
#python train_p.py --datasets=TDrive --prompt=SimplePrompt
#python train_p.py --datasets=TDrive --prompt=GPFplusAtt --p_num=1
#python train_p.py --datasets=TDrive --prompt=GPFplusAtt --p_num=2
#python train_p.py --datasets=TDrive --prompt=GPFplusAtt --p_num=3
#python train_p.py --datasets=TDrive --prompt=GPFplusAtt --p_num=4
#python train_p.py --datasets=TDrive --prompt=GPFplusAtt --p_num=5
#python train_p.py --datasets=TDrive --prompt=GPFplusAtt --p_num=6
#python train_p.py --datasets=TDrive --prompt=GPFplusAtt --p_num=7
#python train_p.py --datasets=TDrive --prompt=GPFplusAtt --p_num=8
#python train_p.py --datasets=TDrive --prompt=GPFplusAtt --p_num=9
#python train_p.py --datasets=TDrive --prompt=GPFplusAtt --p_num=10

python train_p.py --datasets=CHIBike --prompt=SimplePrompt
python train_p.py --datasets=CHIBike --prompt=GPFplusAtt --p_num=1
python train_p.py --datasets=CHIBike --prompt=GPFplusAtt --p_num=2
python train_p.py --datasets=CHIBike --prompt=GPFplusAtt --p_num=3
python train_p.py --datasets=CHIBike --prompt=GPFplusAtt --p_num=4
python train_p.py --datasets=CHIBike --prompt=GPFplusAtt --p_num=5
python train_p.py --datasets=CHIBike --prompt=GPFplusAtt --p_num=6
python train_p.py --datasets=CHIBike --prompt=GPFplusAtt --p_num=7
python train_p.py --datasets=CHIBike --prompt=GPFplusAtt --p_num=8
python train_p.py --datasets=CHIBike --prompt=GPFplusAtt --p_num=9
python train_p.py --datasets=CHIBike --prompt=GPFplusAtt --p_num=10

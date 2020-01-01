#!/usr/bin/env bash
#python train.py -m 'v1' -da True -o 'saved_model/v1_da'
#python train.py -m 'v1' -da True -o 'saved_model/v1_da'
python train.py -m 'v2' -da True -o 'saved_model/v2_da'
python train.py -m 'v2' -da False -o 'saved_model/v2'


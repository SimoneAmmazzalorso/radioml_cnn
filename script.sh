#!/bin/bash
source activate tensorflow
conda list -n tensorflow
conda info -e

python /run_CNN/CNN.py --tag=CNN_notnorm --tag_res=run1 --path=/archive/home/sammazza/radioML/ --path_TD=/archive/home/sammazza/radioML/data/mapsim_PS/ --path_TL=/archive/home/sammazza/radioML/data/mapsim_PS/ --N_start=0 --N_stop=10000 --N_epochs=30 --train --suffix=x1000 --batch_size=8

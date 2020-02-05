#!/bin/bash
source activate tensorflow
#conda list -n tensorflow
conda info -e

python /archive/home/sammazza/radioML/heal_cart_conv.py --tag=CNN_lR --xsize=750 --path=/archive/home/sammazza/radioML/data/mapsim_PS/ --path_dest=/archive/home/sammazza/radioML/data/mapsim_PS_lR/

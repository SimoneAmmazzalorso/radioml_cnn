#!/bin/bash
source activate tensorflow
#conda list -n tensorflow
conda info -e

python /archive/home/sammazza/radioML/dataset_generator.py --tag=CNN_lR --fix_y=87 --path=/archive/home/sammazza/radioML/data/mapsim_PS_lR/ --path_dest=/archive/home/sammazza/radioML/data/mapsim_PS_lR_cropped/

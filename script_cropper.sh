#!/bin/bash
source activate tensorflow
#conda list -n tensorflow
conda info -e

python /archive/home/sammazza/radioML/dataset_generator.py --tag=CNN_GAL --fix_y=450 --path=/archive/home/sammazza/radioML/data/mapsim_PS/ --path_dest=/archive/home/sammazza/radioML/data/mapsim_PS_GAL/

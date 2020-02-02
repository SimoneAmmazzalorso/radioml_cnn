#!/bin/bash
source activate tensorflow
conda list -n tensorflow
conda info -e

python /archive/home/sammazza/radioML/dataset_generator.py --tag=CNN_1 --path=/archive/home/sammazza/radioML/data/mapsim_PS/ --path_dest=/archive/home/sammazza/radioML/data/mapsim_PS_reduced/ 

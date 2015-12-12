#!/bin/bash
for i in  1 2 3 4 5 
do
    gtimeout 320s  mpirun -np 8 python base_model.py
done

#!/bin/bash
for i in 32 64 128 196
do
    mpirun -np 8 python base_model.py $i
    echo $i
done

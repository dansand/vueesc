#!/bin/bash
for i in $(seq 3 0.2 5.2)
do
    mpirun -np 16 python A0.py $i
#    echo $i
done

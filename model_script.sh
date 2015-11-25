#!/bin/bash
for i in  3 6 12 24
do
    mpirun -np 16 python R-11-100.py $i
done

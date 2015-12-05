#!/bin/bash
for i in  10 7 4 1
do
    gtimeout 320s  mpirun -np 8 python A2.py $i
    gtimeout 320s  mpirun -np 8 python A4.py $i
    gtimeout 320s  mpirun -np 8 python A5.py $i
    gtimeout 320s  mpirun -np 8 python A6.py $i
done

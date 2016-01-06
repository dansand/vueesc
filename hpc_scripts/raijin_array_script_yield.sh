#!/bin/bash
for i in  10 7 4 1
do
  qsub -v PBS_ARRAY_INDEX=$yield raijin_array.pbs
done

for yield in $(seq 3 0.2 5.2); do
  qsub -v PBS_ARRAY_INDEX=$yield raijin_array.pbs
done

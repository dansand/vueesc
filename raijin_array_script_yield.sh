

for yield in 10 7 4 1; do
  qsub -v PBS_ARRAY_INDEX=$yield raijin_arrayA2.pbs
  qsub -v PBS_ARRAY_INDEX=$yield raijin_arrayA4.pbs
  qsub -v PBS_ARRAY_INDEX=$yield raijin_arrayA5.pbs
done

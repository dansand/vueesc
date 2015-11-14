for res in 64 96 128; do
  qsub -v PBS_ARRAY_INDEX=$res raijin_array1.pbs
  qsub -v PBS_ARRAY_INDEX=$res raijin_array2.pbs
done

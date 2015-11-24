for res in 144 192 256; do
  qsub -v PBS_ARRAY_INDEX=$res raijin_array1.pbs
  qsub -v PBS_ARRAY_INDEX=$res raijin_array2.pbs
done

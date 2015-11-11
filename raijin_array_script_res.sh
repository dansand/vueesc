for res in 64 96 128; do
  qsub -v PBS_ARRAY_INDEX=$res raijin_array.pbs
done    

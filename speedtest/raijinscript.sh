for res in 16 32 64 128 256; do
  qsub -v PBS_ARRAY_INDEX=$res raijin_array.pbs
done    

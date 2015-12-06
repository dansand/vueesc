for res in 128 144 192; do
  qsub -v PBS_ARRAY_INDEX=$res raijin_arrayR2.pbs
done

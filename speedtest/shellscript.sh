#!/bin/bash
for i in $(seq 4 8 16 32)
do
    mpirun -np $i minimum_chips.py A0.py $i
#    echo $i
done

#!/bin/bash
for i in $(seq 4 8 16 32)
do
    mpirun -np $i thermalConv3D_FEMPIC_res0128_p0001_forMagnus.py $i
#    echo $i
done

#!/bin/bash
for i in 4 8 16 32
do
    mpirun -np $i python minimum_chips.py $i
#    echo $i
done

#!/bin/bash
for i in 4 8 16 32
do
    mpirun -np $i python thermalConv3D_FEMPIC_res0128_p0001_forMagnus.py $i
#    echo $i
done

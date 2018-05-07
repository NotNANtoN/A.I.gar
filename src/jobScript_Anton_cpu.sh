#!/bin/bash
#SBATCH --time=1-08:00:00
#SBATCH --mem=20000
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=antonwiehe@gmail.com
module load Python/3.5.2-foss-2016a
module load matplotlib/1.5.3-foss-2016a-Python-3.5.2
module load tensorflow/1.3.1-foss-2016a-Python-3.5.2
module load h5py/2.5.0-foss-2016a-Python-3.5.1-HDF5-1.8.16 
srun python -O ./aigar.py
0
1000000
0
10000
0
0
1
Default
True
0
1
Default
1

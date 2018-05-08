#!/bin/bash
#SBATCH --time=2-23:00:00
#SBATCH --mem=20000
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.stolt.anso@student.rug.nl
module load Python/3.5.2-foss-2016a
module load matplotlib/1.5.3-foss-2016a-Python-3.5.2
module load TensorFlow/1.6.0-foss-2018a-Python-3.6.4
module load h5py/2.5.0-foss-2016a-Python-3.5.1-HDF5-1.8.16
module load GCCcore/4.9.3 
srun python -O ./aigar.py < nil_input/NN_3.txt

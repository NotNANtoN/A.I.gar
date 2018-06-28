#!/bin/bash
#SBATCH --time=0-20:00:00
#SBATCH --mem=120000
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.stolt.anso@student.rug.nl
#SBATCH --output=SUB_JOB_TEST_%j.out
module load matplotlib/2.1.2-foss-2018a-Python-3.6.4
module load TensorFlow/1.6.0-foss-2018a-Python-3.6.4
module load h5py/2.7.1-foss-2018a-Python-3.6.4
python -O ./aigar.py <<EOF
0
0
0
1
JOB_TRAINING_STEPS
40000
1
CNN_P_REPRESENTATION
True
1
CNN_LAST_GRID
True
1
NUM_NN_BOTS
3
0
0
1
EOF

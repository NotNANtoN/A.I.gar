#!/bin/bash
#SBATCH --time=1-08:00:00
#SBATCH --mem=20000
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.stolt.anso@student.rug.nl
#SBATCH --output=GSPF_11%j.out
module load matplotlib/2.1.2-foss-2018a-Python-3.6.4
module load TensorFlow/1.6.0-foss-2018a-Python-3.6.4
module load h5py/2.7.1-foss-2018a-Python-3.6.4
python -O ./aigar.py <<EOF
0
1000000
0
10000
0
0
1
GRID_SQUARES_PER_FOV
9
0
0
1
EOF

#!/bin/bash
#SBATCH --time=1-08:00:00
#SBATCH --mem=20000
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=antonwiehe@gmail.com
#SBATCH --output=name%j.out
module load matplotlib/2.1.2-foss-2018a-Python-3.6.4
module load TensorFlow/1.6.0-foss-2018a-Python-3.6.4
module load h5py/2.7.1-foss-2018a-Python-3.6.4
python -O ./aigar.py <<EOF
0
0
10000
0
0
1
Default
True
0
0
1
EOF

Running in Slurm clusters
===========

Prerequisites
-------------

Assuming, conda is already installed:

```bash
# GPU
conda env create -f conda_gpu.yml
# or CPU
conda env create -f conda_cpu.yml
conda activate pt_env
```

Submit a job
--------------------------

```bash
sbatch train_complex_c64.sh
```


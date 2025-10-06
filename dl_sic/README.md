Running in Slurm clusters
===========

Prerequisites
-------------

Assuming, conda is already installed:

```bash
conda create -n pt_env python=3.10
conda activate pt_env
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

Submit a job
--------------------------

```bash
sbatch train_complex_c64.sh
```


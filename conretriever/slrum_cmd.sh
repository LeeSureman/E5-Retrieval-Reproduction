srun -p a800 --gres=gpu:2 --nodelist=slurmd-9 --cpus-per-task=24 --mem-per-cpu=4G --pty bash

srun -p xnli --gres=gpu:8 --nodelist=slurmd-7 --cpus-per-task=96 --mem-per-cpu=4G --pty bash

srun -p a800 --gres=gpu:8 --cpus-per-task=96 --mem-per-cpu=4G --pty bash

squeue -p a800 -O "Jobid:6,Partition:10,Name:24,UserName:10,TimeUsed:12,ReasonList:." --sort N+
# 查看集群上显卡和CPU占用情况

srun -p a800 --nodelist=slurmd-7 --pty gpustat -i 2

sinfo -p xnli --Node -O "NodeList:12,GresUsed:30,CPUsState:20,StateLong:12,TimeStamp:12"
sinfo -p x090 --Node -O "NodeList:12,GresUsed:30,CPUsState:20,StateLong:12,TimeStamp:12"


nvidia-smi

srun -p xnli --gres=gpu:8 --nodelist=slurmd-7 --cpus-per-task=96 --mem-per-cpu=4G --pty bash
srun -p xnli --gres=gpu:8 --nodelist=slurmd-8 --cpus-per-task=96 --mem-per-cpu=4G --pty bash
srun -p xnli --gres=gpu:8 --nodelist=slurmd-9 --cpus-per-task=96 --mem-per-cpu=4G --pty bash
srun -p xnli --gres=gpu:2 --nodelist=slurmd-9 --cpus-per-task=24 --mem-per-cpu=4G --pty bash


srun -p xnli --gres=gpu:4 --nodelist=slurmd-9 --cpus-per-task=48 --mem-per-cpu=4G --pty bash

srun -p x090 --gres=gpu:8 --cpus-per-task=96 --mem-per-cpu=4G --pty bash



sinfo -p x090 --Node -O "NodeList:12,GresUsed:30,CPUsState:20,StateLong:12,TimeStamp:12"
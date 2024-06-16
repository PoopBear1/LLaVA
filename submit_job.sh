#!/bin/bash
#SBATCH --job-name=my_job          # 任务名称
#SBATCH --output=output_%j.log     # 标准输出和错误输出文件名
#SBATCH --error=error_%j.log       # 错误输出文件名
#SBATCH --nodes=2                  # 请求2个节点
#SBATCH --ntasks-per-node=1        # 每个节点运行1个任务
#SBATCH --gpus-per-node=4          # 每个节点请求4个GPU
#SBATCH --time=10:00:00            # 任务运行时间，格式为 hh:mm:ss
#SBATCH --account=OD-228963        # 指定项目账户代码
# Load the necessary modules (if any)
# module load my_module

# Activate your environment (if necessary)
conda activate llava

# Run your bash script
srun bash scripts/v1_5/finetune.sh

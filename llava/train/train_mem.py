import sys
sys.path.append('/datasets/work/d61-insect-digitisation/work/Experiments/zha437/zha437/LLaVA')

from llava.train.train import train
import wandb
import os
# 设置 WandB 为离线模式
os.environ["WANDB_MODE"] = "dryrun"
# 初始化 WandB
wandb.init(project="llava")
# wandb.login(key="28b463fd2028ec29ab63bf223ee29499901a8591")

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")

import sys
sys.path.append('/datasets/work/d61-insect-digitisation/work/Experiments/zha437/zha437/LLaVA')

from llava.train.train import train
if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")

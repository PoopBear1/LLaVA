#!/bin/bash
#SBATCH --job-name=my_job          
#SBATCH --output=pretrain_output_%j.log     
#SBATCH --error=pretrain_output_error_%j.log       
#SBATCH --nodes=2           
#SBATCH --ntasks-per-node=1        
#SBATCH --gpus-per-node=4          
#SBATCH --time=00:10:00            
#SBATCH --account=OD-228963        
#SBATCH --cpus-per-task=64   
#SBATCH --mem=128gb
# Load the necessary modules (if any)
# module load my_module

export WANDB_API_KEY="28b463fd2028ec29ab63bf223ee29499901a8591"

# Define master address and port
MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
MASTER_PORT=29500

# Create a hostfile from SLURM environment variables
scontrol show hostname ${SLURM_NODELIST} > hostfile
awk '{print $1" slots=4"}' hostfile > deepspeed_hostfile

# Run the DeepSpeed command
deepspeed --hostfile ./deepspeed_hostfile \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version plain \
    --image_folder  ../../playground/data/LLaVA-Pretrain/images \
    --data_path ../../playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-13b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    # --report_to wandb

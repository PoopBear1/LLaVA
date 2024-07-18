#!/bin/bash

# 脚本名：run_maple_training.sh

# 设置会话和窗口名称
SESSION_NAME="generate_depth_map_on_llava_pretrain"
WINDOW_NAME="Training"

# 检查tmux会话是否已存在
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? == 0 ]; then
    echo "Session $SESSION_NAME already exists. Killing it."
    tmux kill-session -t $SESSION_NAME
fi

# 创建并命名 tmux 会话
tmux new-session -d -s $SESSION_NAME -n $WINDOW_NAME

# 发送命令到tmux窗口以在窗口中激活Conda环境并运行脚本
tmux send-keys -t $SESSION_NAME:Training "export CUDA_VISIBLE_DEVICES=1" C-m
tmux send-keys -t $SESSION_NAME:$WINDOW_NAME "module load cuda/12.1.0" C-m
tmux send-keys -t $SESSION_NAME:$WINDOW_NAME "conda activate llava" C-m
tmux send-keys -t $SESSION_NAME:$WINDOW_NAME "bash scripts/v1_5/gen_depth_map.sh" C-m

# 附加到会话
tmux attach -t $SESSION_NAME

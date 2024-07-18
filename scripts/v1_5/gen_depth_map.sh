image_folder="../../playground/data/LLaVA-Pretrain/images"
output_folder="../../playground/data/LLaVA-Pretrain/depth_images"
encoder="vitl"
file_path="../Depth-Anything-V2/run.py"
ckpt_path="../Depth-Anything-V2/checkpoints/depth_anything_v2_${encoder}.pth"

python $file_path --encoder $encoder --img-path $image_folder --outdir $output_folder --ckpt $ckpt_path --pred-only --grayscale

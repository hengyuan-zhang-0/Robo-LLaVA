# pretrained=/home/henry/LLaVA-NeXT/checkpoints/llava-onevision-qwen2-7b-ov-robovqa-v3
# conv_template=qwen_1_5
# model_name=llava_qwen

# llava-onevision-qwen2-7b-ov-robovqa-v3
MODEL_ARGS_NAME=llavaov-fromScratch-1*1-visualfrozen-8
CONV=qwen_1_5
MODEL_NAME=llava_qwen

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 \
# -m lmms_eval \
#     --model llava_onevision \
#     --model_args pretrained=/home/henry/LLaVA-NeXT/checkpoints/$MODEL_ARGS_NAME,conv_template=$CONV,model_name=$MODEL_NAME \
#     --tasks robovqa \
#     --output_path /home/henry/LLaVA-NeXT/scripts/logs/ \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_onevision \

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 \
-m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=/home/henry/LLaVA-NeXT/checkpoints/$MODEL_ARGS_NAME,conv_template=$CONV,model_name=$MODEL_NAME \
    --tasks egothink \
    --output_path /home/henry/LLaVA-NeXT/scripts/logs/ \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_onevision \

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 \
# -m lmms_eval \
#     --model llava_onevision \
#     --model_args pretrained=/home/henry/LLaVA-NeXT/checkpoints/llavaov-fromScratch-1*1-visualfrozen-8/ \
#     --tasks egothink \
#     --output_path /home/henry/LLaVA-NeXT/scripts/logs/ \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_onevision \

# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 \
# -m lmms_eval \
#     --model llava_onevision \
#     --model_args pretrained=/home/henry/LLaVA-NeXT/checkpoints/llavaov-fromScratch-1*1-visualfrozen-8/ \
#     --tasks mmvet \
#     --output_path /home/henry/LLaVA-NeXT/scripts/logs/ \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_onevision \
MODEL_ARGS_NAME=llavaov-fromScratch-1*1-visualfrozen-8
CONV=qwen_1_5
MODEL_NAME=llava_qwen

CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 \
-m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=/home/henry/LLaVA-NeXT/checkpoints/$MODEL_ARGS_NAME,conv_template=$CONV,model_name=$MODEL_NAME \
    --tasks vqav2 \
    --output_path /home/henry/LLaVA-NeXT/scripts/logs/ \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_onevision \
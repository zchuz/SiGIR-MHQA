source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate llama_factory_810
cd /your/code/path/

MODEL_PATH=/your/pretrained/model/path/
TRAIN_DATA_PATH=/your/data/path/

export WANDB_MODE=offline
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes=4 --main_process_port=24444 -m train.sft \
    --model_name_or_path $MODEL_PATH \
    --tokenizer $MODEL_PATH \
    --train_data_path $TRAIN_DATA_PATH \
    --dataset 2wikimqa hotpotqa musique \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.0 \
    --dtype bf16 \
    --stage sft \
    --do_train \
    --resize_vocab \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --num_train_epochs 2 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --save_steps 999999999 \
    --warmup_ratio 0.03 \
    --save_on_epoch_end \
    --train_batch_size 15 \
    --gradient_accumulation_steps 2 \
    --project_name SiGIR \
    --group_name mistral \
    --run_name mistral_2wiki_hotpotqa_musique_bs128_lr5e-5_ep2 \
    --output_dir mistral_2wiki_hotpotqa_musique_bs128_lr5e-5_ep2 \
    --seed 32799
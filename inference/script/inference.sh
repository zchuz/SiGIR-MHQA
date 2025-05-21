#!/bin/bash
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate vllm
cd /your/code/path/

export CUDA_VISIBLE_DEVICES=0
export MY_PORT=23211
model=/your/merged/model/checkpoint/path/
output=/your/output/path/
python -m vllm.entrypoints.openai.api_server --model $model --dtype auto --api-key abc --disable-log-requests --port $MY_PORT &
API_SERVER_PID=$!
echo "API_SERVER_PID: $API_SERVER_PID"

while ! nc -z localhost $MY_PORT; do
  echo "Waiting for port $MY_PORT to be enabled..."
  sleep 5
done
echo "Port $MY_PORT is enabled, continue with the next task..."

nohup python -m inference.t2_parallel \
    --model_name_or_path $model \
    --dataset 2wikimqa hotpotqa musique \
    --num_datas 500 \
    --test_set \
    --max_workers 500 \
    --overwrite \
    --inference_func ${your_inference_func}\
    # --inference_func greedy \
    # --inference_func guided \
    # --inference_func vanilla \
    --output $output > $output/inference.log 2>&1

echo "Inference task completed, close the program corresponding to port $MY_PORT..."
kill $API_SERVER_PID
wait $API_SERVER_PID 2>/dev/null
echo "The program corresponding to port $MY_PORT has been closed."

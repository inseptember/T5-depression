export CUDA_VISIBLE_DEVICES="0"
export TASK_NAME=sim_v_roberta
export DATA_INPUT=/home/ubuntu/data/user_data/depression/dataset

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python run.py \
  --model_name_or_path hfl/chinese-roberta-wwm-ext \
  --do_train \
  --do_eval \
  --do_predict \
  --train_file=$DATA_INPUT/train.json \
  --validation_file=$DATA_INPUT/val.json \
  --test_file=$DATA_INPUT/test.json \
  --max_seq_length 512 \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 25 \
  --overwrite_output_dir \
  --eval_steps 500 \
  --evaluation_strategy steps \
    --save_total_limit 1 \
    --metric_for_best_model=f1 \
  --output_dir ../output/$TASK_NAME/

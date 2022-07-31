export TASK_NAME=sim_v
export CURR_PWD=./
export DATA_INPUT=$CURR_PWD/dataset

CUDA_VISIBLE_DEVICES="1" python run_pred.py \
  --model_name_or_path $CURR_PWD/output/sim_v/checkpoint-56500 \
  --do_predict \
  --train_file=$DATA_INPUT/train.json \
  --validation_file=$DATA_INPUT/val.json \
  --test_file=$DATA_INPUT/train.json \
  --max_seq_length 512 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 25 \
  --overwrite_output_dir \
  --eval_steps 500 \
  --evaluation_strategy steps \
    --save_total_limit 1 \
    --metric_for_best_model=f1 \
  --output_dir ../output/$TASK_NAME/

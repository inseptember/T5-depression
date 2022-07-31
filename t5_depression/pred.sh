export CUDA_VISIBLE_DEVICES="0"
export TASK_NAME="summary_v"
export BASE_DIR=./
export OUTPUT_DIR=${BASE_DIR}/output/summary_v
export DATA_INPUT=${BASE_DIR}/dataset/gen
export batch_size=8
export lr=5e-5 # t5-large
export model_name=Langboat/mengzi-t5-base
export lr_scheduler=constant_with_warmup
export seed="421"
export label_smoothing="0"
export epoch=25
export eval_steps=500
export warmup_steps=2000


CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python ${PWD}/run.py \
    --model_name_or_path=${OUTPUT_DIR} \
    --do_predict \
    --test_file ${DATA_INPUT}/train.json \
    --train_file ${DATA_INPUT}/train.json \
    --validation_file ${DATA_INPUT}/val.json \
    --text_column text \
    --summary_column summary \
    --source_prefix "span: " \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --max_source_length=512 \
    --max_target_length=64 \
    --num_train_epochs=${epoch} \
    --per_device_train_batch_size=${batch_size} \
    --per_device_eval_batch_size=$((batch_size * 2)) \
    --eval_steps ${eval_steps} \
    --learning_rate=${lr} \
    --lr_scheduler_type=${lr_scheduler} \
    --label_smoothing_factor=${label_smoothing} \
    --warmup_steps ${warmup_steps} \
    --seed=${seed} --disable_tqdm False \
    --predict_with_generate \
    --use_fast_tokenizer=False \
    --evaluation_strategy steps \
    --save_total_limit 1 \
    --num_beams=1 \
    --task_name=${TASK_NAME} \
    --load_best_model_at_end
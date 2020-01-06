rm -drf model_out
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
PIPELINE_CONFIG_PATH=/home/eric/models/research/ssd_mobilenet_v1_pets.config
MODEL_DIR=/home/eric/models/research/model_out
NUM_TRAIN_STEPS=1
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
#!/bin/bash

set -x

export WANDB_API_KEY=0126ce347a0f2f0d757943bfe70b888801e1013e
export WANDB_MODE=offline
export NCCL_DEBUG=INFO

export RAY_MASTER_PORT=6379


timestamp=$(date '+%Y-%m-%d_%H-%M-%S')
PROJECT_NAME=deepeyes_antique
EXPERIMENT_NAME=train-from_sft-$timestamp

BASEDIR=/inspire/hdd/project/qproject-multireasoning/shaowenqi-shaowenqi/zhouzhixiang/Project/verl

SAVE_CHECKPOINT_DIR=${BASEDIR}/checkpoints
# DATASET_TRAIN=${BASEDIR}/dataset/train.parquet
# DATASET_VAL=${BASEDIR}/dataset/val.parquet

DATASET_1=${BASEDIR}/data/antique_data/train_gugong_llm_judge.parquet

REF_MODEL_PATH=/inspire/hdd/project/qproject-multireasoning/shaowenqi-shaowenqi/mazhongtian/model_results/output/Qwen2.5-7B-vl/System-Prompt/v45-20250813-085156/checkpoint-25

JUDGE_MODEL_PATH=/inspire/hdd/project/qproject-multireasoning/shaowenqi-shaowenqi/zhouzhixiang/Model/Qwen2.5-72B-Instruct

export WORLD_SIZE=$((WORLD_SIZE - 1))

ray stop --force

# SGLANG service configuration (on the last rank)
if [ "$PET_NODE_RANK" -eq "$((WORLD_SIZE))" ]; then
    # On SGLANG service node, use localhost
    # llm_ip=$(hostname -I | awk '{print $1}')
    llm_ip=$(ip route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++) if ($i=="src") print $(i+1)}')
    export LLM_AS_A_JUDGE_BASE="http://${llm_ip}:18901/v1"
    mkdir -p $BASEDIR/tmp
    echo $LLM_AS_A_JUDGE_BASE > $BASEDIR/tmp/llm_ip.txt
    echo "LLM_AS_A_JUDGE_BASE: $LLM_AS_A_JUDGE_BASE"
else
    sleep 10
fi

# Start SGLANG service on the last rank
if [ "$PET_NODE_RANK" -eq "$((WORLD_SIZE))" ]; then
    echo "Starting SGLANG service on rank $PET_NODE_RANK..."
    python -m sglang.launch_server --model-path $JUDGE_MODEL_PATH \
        --port 18901 \
        --host 0.0.0.0 \
        --served-model-name "judge" \
        --tp-size 8 \
        --context-length 32768 \
        --trust-remote-code \
        --log-requests false
    # vllm serve $JUDGE_MODEL_PATH \
    #     --port 18901 \
    #     --host 0.0.0.0 \
    #     --gpu-memory-utilization 0.8 \
    #     --max-model-len 32768 \
    #     --tensor-parallel-size 8 \
    #     --served-model-name "judge" \
    #     --trust-remote-code \
    #     --disable-log-requests
    exit 0
fi

# Start Ray cluster, the first rank starts the head node, other ranks join the cluster except the last rank
if [ "$PET_NODE_RANK" -eq 0 ]; then
    ray start --head --port=$RAY_MASTER_PORT --dashboard-host=0.0.0.0 --num-gpus 8
    echo "Started Ray head node at $NODE_IP"
else
    if [ "$PET_NODE_RANK" -lt "$((WORLD_SIZE))" ]; then
        sleep 10
        ray start --address="${MASTER_ADDR}:${RAY_MASTER_PORT}" --num-gpus 8 --block
        echo "Joined Ray cluster at ${MASTER_ADDR}:${RAY_MASTER_PORT}"
    fi
fi

# Wait for 30 seconds to ensure the Ray cluster is ready
sleep 30

# Read the LLM IP from the file
if [ ! -f "$BASEDIR/tmp/llm_ip.txt" ]; then
    echo "Error: LLM IP file not found at $BASEDIR/tmp/llm_ip.txt"
    echo "Please ensure SGLANG service node (last rank) is running first"
    exit 1
fi
export LLM_AS_A_JUDGE_BASE=$(cat $BASEDIR/tmp/llm_ip.txt)
echo "Read LLM_AS_A_JUDGE_BASE: $LLM_AS_A_JUDGE_BASE"

# In master node, wait for the SGLANG service to be ready
if [ "$PET_NODE_RANK" -eq 0 ]; then

    # Create log and checkpoint directories
    if [ ! -d logs/$PROJECT_NAME/$EXPERIMENT_NAME ]; then
        mkdir -p logs/$PROJECT_NAME/$EXPERIMENT_NAME
    fi
    if [ ! -d checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME ]; then
        mkdir -p checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME
    fi
    if [ ! -d checkpoints/logs/tensorboard ]; then
        mkdir -p checkpoints/logs/tensorboard
    fi
    if [ ! -d checkpoints/logs/rl_logging_board ]; then
        mkdir -p checkpoints/logs/rl_logging_board
    fi

    # Wait for the SGLANG service to be ready
    echo "Waiting for SGLANG judge service to be ready at ${LLM_AS_A_JUDGE_BASE}..."
    RETRY=0
    MAX_RETRIES=600  # Wait up to 600 times (about 50 minutes)
    while ! curl -s "${LLM_AS_A_JUDGE_BASE}/models" > /dev/null; do
        sleep 5
        RETRY=$((RETRY+1))
        if [ $RETRY -ge $MAX_RETRIES ]; then
            echo "Error: SGLANG judge service not available after $((MAX_RETRIES * 5)) seconds."
            exit 1
        fi
    done
    echo "SGLANG judge service is up. Proceeding with training..."



PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path=${BASEDIR}/recipe/antique/configs \
    --config-name='antique_multiturn_grpo' \
    data.train_files=${DATASET_1} \
    data.val_files=[${DATASET_VAL}] \
    data.train_batch_size=64 \
    data.max_prompt_length=32768 \
    data.max_response_length=20480 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=recipe/deepeyes/configs/image_zoom_in_tool_config.yaml \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb','tensorboard'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=8 \
    trainer.test_freq=10000 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=1 \
    $@ 2>&1 | tee -a logs/$PROJECT_NAME/$EXPERIMENT_NAME/training.log

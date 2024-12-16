set -e
mkdir -p results/logs results/checkpoints results/evaluation

CUDA_AVAILABLE=$(python -c "import torch; print(int(torch.cuda.is_available()))")
MPS_AVAILABLE=$(python -c "import torch; print(int(torch.backends.mps.is_available()))")

if [ "$CUDA_AVAILABLE" -eq 1 ]; then
    echo "CUDA is available"
    DEVICE="cuda"
elif [ "$MPS_AVAILABLE" -eq 1 ]; then
    echo "MPS is available"
    DEVICE="mps"
else
    echo "Using CPU"
    DEVICE="cpu"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="${TIMESTAMP}"

AGENT_TYPE=$(python -c "import yaml; print(yaml.safe_load(open('configs/default_config.yaml'))['agent']['type'])")
RUN_NAME="${AGENT_TYPE}_${TIMESTAMP}"

MODE="train"
CHECKPOINT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --evaluate)
            MODE="evaluate"
            shift
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: ./run.sh [--evaluate] [--checkpoint path/to/checkpoint.pth] [--device cpu|cuda|mps]"
            exit 1
            ;;
    esac
done

echo "Mode: ${MODE}"
echo "Device: ${DEVICE}"

if [ "$MODE" = "evaluate" ]; then
    if [ -z "$CHECKPOINT" ]; then
        echo "Error: --checkpoint is required in evaluate mode"
        exit 1
    fi
    if [ ! -f "$CHECKPOINT" ]; then
        echo "Error: Checkpoint file not found: $CHECKPOINT"
        exit 1
    fi
    echo "Using checkpoint: ${CHECKPOINT}"
fi

mkdir -p "results/evaluation/${RUN_NAME}"

# train
if [ "$MODE" = "train" ]; then
    echo "Starting training..."
    mkdir -p "results/logs/${RUN_NAME}"
    mkdir -p "results/checkpoints/${RUN_NAME}"
    
    TMP_CONFIG="configs/tmp_config_${TIMESTAMP}.yaml"
    python -c "
import yaml
with open('configs/default_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['agent']['device'] = '${DEVICE}'
with open('${TMP_CONFIG}', 'w') as f:
    yaml.dump(config, f)
"
    python -m src.train \
        --config "${TMP_CONFIG}" \
        --run-name "${RUN_NAME}" || { echo "Training failed"; exit 1; }
    
    rm "${TMP_CONFIG}"

    BEST_MODEL="results/checkpoints/${RUN_NAME}/best_model.pth"
else
    BEST_MODEL="$CHECKPOINT"
fi

echo "Using model at: ${BEST_MODEL}"

# evaluate
echo "Starting evaluation..."
NUM_EPISODES=$(python -c "import yaml; print(yaml.safe_load(open('configs/default_config.yaml'))['evaluation']['num_episodes'])")
if [ "$MODE" = "train" ]; then
    # training mode, use baseline model
    BASELINE_MODEL="results/checkpoints/${RUN_NAME}/baseline_model.pth"
    TMP_CONFIG="configs/tmp_config_${TIMESTAMP}.yaml"
    python -c "
import yaml
with open('configs/default_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['agent']['device'] = '${DEVICE}'
with open('${TMP_CONFIG}', 'w') as f:
    yaml.dump(config, f)
"
    python -m src.evaluate \
        --config "${TMP_CONFIG}" \
        --model "${BEST_MODEL}" \
        --baseline "${BASELINE_MODEL}" \
        --episodes ${NUM_EPISODES} \
        --save-dir "results/evaluation/${RUN_NAME}" || { echo "Evaluation failed"; exit 1; }
    
    rm "${TMP_CONFIG}"
else
    # evaluation mode, use checkpoint
    BASELINE_DIR=$(dirname "${CHECKPOINT}")
    BASELINE_MODEL="${BASELINE_DIR}/baseline_model.pth"
    TMP_CONFIG="configs/tmp_config_${TIMESTAMP}.yaml"
    python -c "
import yaml
with open('configs/default_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['agent']['device'] = '${DEVICE}'
with open('${TMP_CONFIG}', 'w') as f:
    yaml.dump(config, f)
"
    python -m src.evaluate \
        --config "${TMP_CONFIG}" \
        --model "${CHECKPOINT}" \
        --baseline "${BASELINE_MODEL}" \
        --episodes ${NUM_EPISODES} \
        --save-dir "results/evaluation/${RUN_NAME}" || { echo "Evaluation failed"; exit 1; }
    
    rm "${TMP_CONFIG}"
fi

# visualization
echo "Creating visualizations..."
if [ "$MODE" = "train" ]; then
    LOGS_DIR="results/logs/${RUN_NAME}"
    TMP_CONFIG="configs/tmp_config_${TIMESTAMP}.yaml"
    python -c "
import yaml
with open('configs/default_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['agent']['device'] = '${DEVICE}'
with open('${TMP_CONFIG}', 'w') as f:
    yaml.dump(config, f)
"
    python -m src.visualize \
        --tensorboard-dir "${LOGS_DIR}" \
        --eval-results "results/evaluation/${RUN_NAME}/evaluation_results.json" \
        --model "${BEST_MODEL}" \
        --save-dir "results/evaluation/${RUN_NAME}" \
        --timestamp "${TIMESTAMP}" || { echo "Visualization failed"; exit 1; }
    
    rm "${TMP_CONFIG}"
else
    LOGS_DIR=$(dirname $(dirname "$CHECKPOINT"))/logs/$(basename $(dirname "$CHECKPOINT"))
    if [ -d "$LOGS_DIR" ]; then
        TMP_CONFIG="configs/tmp_config_${TIMESTAMP}.yaml"
        python -c "
import yaml
with open('configs/default_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['agent']['device'] = '${DEVICE}'
with open('${TMP_CONFIG}', 'w') as f:
    yaml.dump(config, f)
"
        python -m src.visualize \
            --tensorboard-dir "${LOGS_DIR}" \
            --eval-results "results/evaluation/${RUN_NAME}/evaluation_results.json" \
            --model "${BEST_MODEL}" \
            --save-dir "results/evaluation/${RUN_NAME}" \
            --timestamp "${TIMESTAMP}" || { echo "Visualization failed"; exit 1; }
        
        rm "${TMP_CONFIG}"
    else
        TMP_CONFIG="configs/tmp_config_${TIMESTAMP}.yaml"
        python -c "
import yaml
with open('configs/default_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['agent']['device'] = '${DEVICE}'
with open('${TMP_CONFIG}', 'w') as f:
    yaml.dump(config, f)
"
        python -m src.visualize \
            --eval-results "results/evaluation/${RUN_NAME}/evaluation_results.json" \
            --model "${BEST_MODEL}" \
            --save-dir "results/evaluation/${RUN_NAME}" \
            --timestamp "${TIMESTAMP}" || { echo "Visualization failed"; exit 1; }
        
        rm "${TMP_CONFIG}"
    fi
fi

echo "Process completed!"
echo "Results available in:"
echo "- Evaluation: results/evaluation/${RUN_NAME}"
if [ "$MODE" = "train" ]; then
    echo "- Logs: ${LOGS_DIR}"
    echo "- Models: results/checkpoints/${RUN_NAME}"
    echo "To view training curves, run:"
    echo "tensorboard --logdir ${LOGS_DIR}"
fi
set -e
mkdir -p results/logs results/checkpoints results/evaluation

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
            echo "Usage: ./run.sh [--evaluate] [--checkpoint path/to/checkpoint.pth]"
            exit 1
            ;;
    esac
done

echo "Mode: ${MODE}"
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
    
    python -m src.train \
        --config configs/default_config.yaml \
        --run-name "${RUN_NAME}" || { echo "Training failed"; exit 1; }

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
    python -m src.evaluate \
        --config configs/default_config.yaml \
        --model "${BEST_MODEL}" \
        --baseline "${BASELINE_MODEL}" \
        --episodes ${NUM_EPISODES} \
        --save-dir "results/evaluation/${RUN_NAME}" || { echo "Evaluation failed"; exit 1; }
else
    # evaluation mode, use checkpoint
    BASELINE_DIR=$(dirname "${CHECKPOINT}")
    BASELINE_MODEL="${BASELINE_DIR}/baseline_model.pth"
    python -m src.evaluate \
        --config configs/default_config.yaml \
        --model "${CHECKPOINT}" \
        --baseline "${BASELINE_MODEL}" \
        --episodes ${NUM_EPISODES} \
        --save-dir "results/evaluation/${RUN_NAME}" || { echo "Evaluation failed"; exit 1; }
fi

# visualization
echo "Creating visualizations..."
if [ "$MODE" = "train" ]; then
    LOGS_DIR="results/logs/${RUN_NAME}"
    python -m src.visualize \
        --tensorboard-dir "${LOGS_DIR}" \
        --eval-results "results/evaluation/${RUN_NAME}/evaluation_results.json" \
        --model "${BEST_MODEL}" \
        --save-dir "results/evaluation/${RUN_NAME}" \
        --timestamp "${TIMESTAMP}" || { echo "Visualization failed"; exit 1; }
else
    LOGS_DIR=$(dirname $(dirname "$CHECKPOINT"))/logs/$(basename $(dirname "$CHECKPOINT"))
    if [ -d "$LOGS_DIR" ]; then
        python -m src.visualize \
            --tensorboard-dir "${LOGS_DIR}" \
            --eval-results "results/evaluation/${RUN_NAME}/evaluation_results.json" \
            --model "${BEST_MODEL}" \
            --save-dir "results/evaluation/${RUN_NAME}" \
            --timestamp "${TIMESTAMP}" || { echo "Visualization failed"; exit 1; }
    else
        python -m src.visualize \
            --eval-results "results/evaluation/${RUN_NAME}/evaluation_results.json" \
            --model "${BEST_MODEL}" \
            --save-dir "results/evaluation/${RUN_NAME}" \
            --timestamp "${TIMESTAMP}" || { echo "Visualization failed"; exit 1; }
    fi
fi

echo "Process completed! View results in tensorboard: tensorboard --logdir ${LOGS_DIR}"
if [ "$MODE" = "train" ]; then
    echo "- Logs: ${LOGS_DIR}"
    echo "- Models: results/checkpoints/${RUN_NAME}"
    echo "To view training curves, run:"
    echo "tensorboard --logdir ${LOGS_DIR}"
fi
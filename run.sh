#!/bin/bash
set -e
mkdir -p results/logs results/checkpoints results/evaluation results/visualization

# experiment ID
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="experiment_${TIMESTAMP}"
AGENT_TYPE="dqn"
RUN_NAME="${AGENT_TYPE}_${TIMESTAMP}"

MODE="train"  # default mode
CHECKPOINT=""

# parse command line arguments
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

mkdir -p "results/evaluation/${EXPERIMENT_NAME}"
mkdir -p "results/visualization/vis_${TIMESTAMP}"

if [ "$MODE" = "train" ]; then
    # training mode
    echo "Starting training..."
    mkdir -p "results/logs/${RUN_NAME}"
    mkdir -p "results/checkpoints/${RUN_NAME}"
    
    python -m src.train \
        --config configs/default_config.yaml \
        --run-name "${RUN_NAME}" || { echo "Training failed"; exit 1; }

    BEST_MODEL="results/checkpoints/${RUN_NAME}/best_model.pth"
else
    # evaluation mode
    BEST_MODEL="$CHECKPOINT"
fi

echo "Using model at: ${BEST_MODEL}"

# evaluation
echo "Starting evaluation..."
python -m src.evaluate \
    --config configs/default_config.yaml \
    --model "${BEST_MODEL}" \
    --episodes 100 \
    --save-dir "results/evaluation/${EXPERIMENT_NAME}" || { echo "Evaluation failed"; exit 1; }

# visualization
echo "Creating visualizations..."
if [ "$MODE" = "train" ]; then
    LOGS_DIR="results/logs/${RUN_NAME}"
    python -m src.visualize \
        --tensorboard-dir "${LOGS_DIR}" \
        --eval-results "results/evaluation/${EXPERIMENT_NAME}/evaluation_results.json" \
        --model "${BEST_MODEL}" || { echo "Visualization failed"; exit 1; }
else
    LOGS_DIR=$(dirname $(dirname "$CHECKPOINT"))/logs/$(basename $(dirname "$CHECKPOINT"))
    if [ -d "$LOGS_DIR" ]; then
        python -m src.visualize \
            --tensorboard-dir "${LOGS_DIR}" \
            --eval-results "results/evaluation/${EXPERIMENT_NAME}/evaluation_results.json" \
            --model "${BEST_MODEL}" || { echo "Visualization failed"; exit 1; }
    else
        python -m src.visualize \
            --eval-results "results/evaluation/${EXPERIMENT_NAME}/evaluation_results.json" \
            --model "${BEST_MODEL}" || { echo "Visualization failed"; exit 1; }
    fi
fi

echo "Process completed!"
echo "Results can be found in:"
echo "- Evaluation: results/evaluation/${EXPERIMENT_NAME}"
echo "- Visualizations: results/visualization/vis_${TIMESTAMP}"
if [ "$MODE" = "train" ]; then
    echo "- Logs: ${LOGS_DIR}"
    echo "- Models: results/checkpoints/${RUN_NAME}"
    echo "To view training curves, run:"
    echo "tensorboard --logdir ${LOGS_DIR}"
fi

# ./run.sh --evaluate --checkpoint results/checkpoints/dqn_20241202_114654/best_model.pth
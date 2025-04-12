#!/bin/bash

MODEL_REPO="nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
ALIAS_NAME="llama3.1-8b"
VENV_DIR="vllm_env"
PORT=8080
CONFIG_FILE="models_config.json"
LOGFILE="vllm_$PORT.log"

start() {
  echo "🟢 Starting vLLM with alias: $ALIAS_NAME"

  # Setup virtual environment if not exist
  if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    pip install --upgrade pip
    pip install "vllm[serve]" transformers accelerate
  else
    source $VENV_DIR/bin/activate
  fi

  # Generate model config with float16 dtype
  echo "⚙️ Generating $CONFIG_FILE..."
  cat > $CONFIG_FILE <<EOF
[
  {
    "model_name": "$ALIAS_NAME",
    "model": "$MODEL_REPO",
    "dtype": "float16"
  }
]
EOF

  # Start the vLLM server
  echo "🚀 Launching server on port $PORT..."
  nohup python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_REPO \
    --host 0.0.0.0 \
    --port $PORT \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    > "$LOGFILE" 2>&1 &

  echo "✅ vLLM running on http://localhost:$PORT"
  echo "📄 Logs saved to: $LOGFILE"
}

stop() {
  echo "🛑 Stopping vLLM server..."
  PID=$(lsof -ti tcp:$PORT)
  if [ ! -z "$PID" ]; then
    kill -9 $PID
    echo "✔️ Killed process $PID (port $PORT)"
  else
    echo "ℹ️ No server running on port $PORT"
  fi
}

case "$1" in
  start)
    start
    ;;
  stop)
    stop
    ;;
  *)
    echo "Usage: $0 {start|stop}"
    ;;
esac
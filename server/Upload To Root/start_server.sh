#!/bin/bash

# DeepSeek-OCR Server Startup Script
# Runs in foreground mode on port 7777 with vLLM backend

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Navigate to server directory
cd /workspace/deepseek-ocr-server || {
    echo -e "${RED}Error: Server directory not found at /workspace/deepseek-ocr-server${NC}"
    exit 1
}

# Set environment variables
export MODEL_PATH="/workspace/models/deepseek-ai/DeepSeek-OCR"
export PORT=7777
export HOST="0.0.0.0"
export GPU_MEMORY_UTILIZATION=0.85
export CUDA_VISIBLE_DEVICES=0

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model not found at $MODEL_PATH${NC}"
    echo "Please run the setup script first: bash runpod_terminal_setup.sh"
    exit 1
fi

# Check if port is already in use
if netstat -tuln 2>/dev/null | grep -q ":$PORT "; then
    echo -e "${YELLOW}Warning: Port $PORT is already in use${NC}"
    echo "Another server instance might be running. Check with:"
    echo "  bash check_server.sh"
    echo ""
    echo "To stop the existing server:"
    echo "  bash stop_server.sh"
    exit 1
fi

# Clear screen and show startup message
clear
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}    Starting DEEPSEEK-OCR SERVER${NC}"
echo -e "${GREEN}    PDF to Markdown Conversion${NC}"
echo -e "${GREEN}    Port: $PORT | Backend: vLLM | Mode: Foreground${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""

# Show GPU information
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | while read line; do
    echo "  $line"
done
echo ""

# Show model information
echo "Model Configuration:"
echo "  Path: $MODEL_PATH"
echo "  GPU Memory: ${GPU_MEMORY_UTILIZATION}"
echo ""

echo -e "${YELLOW}Loading model... This may take 30-60 seconds${NC}"
echo "Press Ctrl+C to stop the server"
echo ""
echo "============================================================"
echo ""

# Start the server
python3 -m uvicorn app.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level info \
    --access-log
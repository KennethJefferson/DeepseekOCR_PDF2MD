#!/bin/bash
# DeepSeek-OCR Server Deployment Verification Script
# Run this on Runpod after installation to verify everything is working

echo "=================================================="
echo "   DEEPSEEK-OCR DEPLOYMENT VERIFICATION"
echo "   Testing all components and dependencies"
echo "=================================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track overall status
ERRORS=0
WARNINGS=0

# Function to check command exists
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $2"
        return 0
    else
        echo -e "${RED}✗${NC} $2"
        ERRORS=$((ERRORS+1))
        return 1
    fi
}

# Function to check Python package
check_python_package() {
    if python3 -c "import $1" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Python package: $1"
        return 0
    else
        echo -e "${RED}✗${NC} Python package: $1 (run: pip install $2)"
        ERRORS=$((ERRORS+1))
        return 1
    fi
}

# Function to check file exists
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} File exists: $1"
        return 0
    else
        echo -e "${RED}✗${NC} File missing: $1"
        ERRORS=$((ERRORS+1))
        return 1
    fi
}

# Function to check directory exists
check_directory() {
    if [ -d "$1" ]; then
        SIZE=$(du -sh "$1" 2>/dev/null | cut -f1)
        echo -e "${GREEN}✓${NC} Directory exists: $1 (Size: $SIZE)"
        return 0
    else
        echo -e "${RED}✗${NC} Directory missing: $1"
        ERRORS=$((ERRORS+1))
        return 1
    fi
}

echo "1. SYSTEM DEPENDENCIES"
echo "----------------------"
check_command python3 "Python 3 installed"
check_command pip "pip installed"
check_command pdftoppm "poppler-utils installed (PDF processing)"
check_command nvidia-smi "NVIDIA drivers installed"
check_command curl "curl installed"
echo ""

echo "2. PYTHON PACKAGES"
echo "------------------"
check_python_package torch torch
check_python_package fastapi fastapi
check_python_package uvicorn uvicorn
check_python_package PIL Pillow
check_python_package pdf2image pdf2image
check_python_package fitz PyMuPDF
check_python_package transformers transformers
check_python_package vllm vllm
check_python_package pydantic pydantic
echo ""

echo "3. CUDA/GPU CHECK"
echo "-----------------"
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" 2>/dev/null
if [ $? -eq 0 ]; then
    python3 -c "import torch; print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 2>/dev/null
    python3 -c "import torch; print(f'PyTorch Version: {torch.__version__}')" 2>/dev/null
    echo -e "${GREEN}✓${NC} CUDA/GPU configuration OK"
else
    echo -e "${RED}✗${NC} CUDA/GPU configuration failed"
    ERRORS=$((ERRORS+1))
fi
echo ""

echo "4. DIRECTORY STRUCTURE"
echo "----------------------"
check_directory /workspace/deepseek-ocr-server
check_directory /workspace/deepseek-ocr-server/app
check_directory /workspace/models
check_directory /workspace/logs
check_directory /workspace/temp
echo ""

echo "5. SERVER FILES"
echo "---------------"
check_file /workspace/deepseek-ocr-server/app/main.py
check_file /workspace/deepseek-ocr-server/app/models.py
check_file /workspace/deepseek-ocr-server/app/ocr_service.py
check_file /workspace/deepseek-ocr-server/app/pdf_processor.py
check_file /workspace/deepseek-ocr-server/app/__init__.py
echo ""

echo "6. HELPER SCRIPTS"
echo "-----------------"
check_file /workspace/start_server.sh
check_file /workspace/check_server.sh
check_file /workspace/stop_server.sh

# Check if scripts are executable
if [ -x /workspace/start_server.sh ]; then
    echo -e "${GREEN}✓${NC} start_server.sh is executable"
else
    echo -e "${YELLOW}⚠${NC} start_server.sh not executable (run: chmod +x /workspace/start_server.sh)"
    WARNINGS=$((WARNINGS+1))
fi
echo ""

echo "7. MODEL FILES"
echo "--------------"
MODEL_DIR="/workspace/models/deepseek-ai/DeepSeek-OCR"
if [ -d "$MODEL_DIR" ]; then
    echo -e "${GREEN}✓${NC} Model directory exists: $MODEL_DIR"

    # Check for key model files
    check_file "$MODEL_DIR/config.json"
    check_file "$MODEL_DIR/generation_config.json"

    # Check for model weights
    if ls "$MODEL_DIR"/model*.safetensors 1> /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Model weights found (*.safetensors)"
    else
        echo -e "${RED}✗${NC} Model weights missing (*.safetensors)"
        echo "  Run: huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir $MODEL_DIR"
        ERRORS=$((ERRORS+1))
    fi

    # Check model size
    SIZE=$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)
    echo "  Model size: $SIZE (should be ~8-10GB)"

    # Parse size and check if it's reasonable
    SIZE_NUM=$(echo $SIZE | sed 's/[^0-9.]//g')
    SIZE_UNIT=$(echo $SIZE | sed 's/[0-9.]//g')
    if [[ "$SIZE_UNIT" == "G" ]] && (( $(echo "$SIZE_NUM > 7" | bc -l) )); then
        echo -e "${GREEN}✓${NC} Model size looks correct"
    else
        echo -e "${YELLOW}⚠${NC} Model size seems small, may need to re-download"
        WARNINGS=$((WARNINGS+1))
    fi
else
    echo -e "${RED}✗${NC} Model directory missing: $MODEL_DIR"
    echo "  Run: huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir $MODEL_DIR"
    ERRORS=$((ERRORS+1))
fi
echo ""

echo "8. VLLM COMPATIBILITY"
echo "---------------------"
python3 -c "
try:
    from vllm import LLM
    print('✓ vLLM can be imported')
    # Try to check CUDA compatibility
    import torch
    if torch.cuda.is_available():
        print('✓ vLLM should work with CUDA')
    else:
        print('⚠ CUDA not available for vLLM')
except ImportError as e:
    print(f'✗ vLLM import failed: {e}')
except Exception as e:
    print(f'⚠ vLLM check warning: {e}')
" 2>&1
echo ""

echo "9. PORT AVAILABILITY"
echo "--------------------"
if netstat -tuln 2>/dev/null | grep -q ":7777 "; then
    echo -e "${YELLOW}⚠${NC} Port 7777 is already in use"
    echo "  Run: bash /workspace/stop_server.sh"
    WARNINGS=$((WARNINGS+1))
else
    echo -e "${GREEN}✓${NC} Port 7777 is available"
fi
echo ""

echo "10. MEMORY CHECK"
echo "----------------"
# Check system memory
MEM_TOTAL=$(free -h | grep "^Mem:" | awk '{print $2}')
MEM_AVAIL=$(free -h | grep "^Mem:" | awk '{print $7}')
echo "System RAM: $MEM_AVAIL available of $MEM_TOTAL total"

# Check GPU memory
if command -v nvidia-smi &> /dev/null; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits | head -1)
    GPU_FREE=$(echo $GPU_MEM | cut -d',' -f1)
    GPU_TOTAL=$(echo $GPU_MEM | cut -d',' -f2)
    echo "GPU Memory: ${GPU_FREE}MB free of ${GPU_TOTAL}MB total"

    if [ "$GPU_FREE" -lt "10000" ]; then
        echo -e "${YELLOW}⚠${NC} Low GPU memory (< 10GB free)"
        echo "  The model needs ~10GB VRAM to load"
        WARNINGS=$((WARNINGS+1))
    else
        echo -e "${GREEN}✓${NC} Sufficient GPU memory available"
    fi
fi
echo ""

# Final summary
echo "=================================================="
echo "              VERIFICATION SUMMARY"
echo "=================================================="

if [ $ERRORS -eq 0 ]; then
    if [ $WARNINGS -eq 0 ]; then
        echo -e "${GREEN}✓ ALL CHECKS PASSED!${NC}"
        echo ""
        echo "Your deployment is ready. Start the server with:"
        echo "  bash /workspace/start_server.sh"
    else
        echo -e "${YELLOW}⚠ COMPLETED WITH $WARNINGS WARNING(S)${NC}"
        echo ""
        echo "The deployment should work but review warnings above."
        echo "Start the server with:"
        echo "  bash /workspace/start_server.sh"
    fi
else
    echo -e "${RED}✗ FAILED WITH $ERRORS ERROR(S)${NC}"
    echo ""
    echo "Please fix the errors above before starting the server."
    echo "Common fixes:"
    echo "  - Install missing packages: pip install <package>"
    echo "  - Download model: huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir /workspace/models/deepseek-ai/DeepSeek-OCR"
    echo "  - Create missing directories: mkdir -p /workspace/deepseek-ocr-server/app"
fi

if [ $WARNINGS -gt 0 ]; then
    echo ""
    echo "Warnings detected: $WARNINGS"
fi

echo ""
echo "For manual testing after server starts:"
echo "  curl http://localhost:7777/health"
echo "  bash /workspace/check_server.sh"
echo ""
echo "External access URL format:"
echo "  http://[your-pod-name]-7777.proxy.runpod.net"
echo "=================================================="

# Return appropriate exit code
if [ $ERRORS -gt 0 ]; then
    exit 1
else
    exit 0
fi
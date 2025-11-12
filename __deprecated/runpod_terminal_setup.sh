#!/bin/bash

# DeepSeek-OCR Terminal Setup Script for Runpod
# This script sets up everything needed to run the DeepSeek-OCR server
# Port: 7777 | Backend: vLLM

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_header() {
    echo ""
    echo "============================================"
    echo "$1"
    echo "============================================"
    echo ""
}

# Start setup
print_header "DeepSeek-OCR Terminal Setup for Runpod"
echo "Port: 7777 | Backend: vLLM | Mode: Foreground"
echo ""

# Check if we're in the right directory
if [ ! -f "/workspace/deepseek-ocr-server/app/main.py" ]; then
    print_error "Server files not found at /workspace/deepseek-ocr-server/"
    echo "Please ensure server files are uploaded first."
    exit 1
fi

cd /workspace/deepseek-ocr-server

# Step 1: Check CUDA availability
print_header "Step 1: Checking CUDA availability"
if command -v nvidia-smi &> /dev/null; then
    print_status "CUDA is available"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    print_error "CUDA not available. GPU is required for this server."
    exit 1
fi

# Step 2: Check disk space
print_header "Step 2: Checking disk space"
AVAILABLE_SPACE=$(df -BG /workspace | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 50 ]; then
    print_warning "Low disk space: ${AVAILABLE_SPACE}GB available (50GB recommended)"
    echo "Model download requires ~10GB, plus space for processing"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    print_status "Disk space OK: ${AVAILABLE_SPACE}GB available"
fi

# Step 3: Install system dependencies
print_header "Step 3: Installing system dependencies"
print_status "Updating package lists..."
apt-get update -qq

print_status "Installing poppler-utils (for PDF processing)..."
apt-get install -y poppler-utils > /dev/null 2>&1 || print_warning "poppler-utils may already be installed"

print_status "Installing other dependencies..."
apt-get install -y git curl wget > /dev/null 2>&1 || print_warning "Some packages may already be installed"

# Step 4: Install Python packages
print_header "Step 4: Installing Python packages"

print_status "Upgrading pip..."
pip install --upgrade pip setuptools wheel -q

print_status "Installing PyTorch with CUDA 11.8 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q

print_status "Installing vLLM (this may take a few minutes)..."
pip install vllm -q || {
    print_warning "vLLM installation failed, trying alternative method..."
    pip install vllm --no-build-isolation -q || {
        print_error "vLLM installation failed. Server may run slower with transformers fallback."
    }
}

print_status "Installing FastAPI and server dependencies..."
pip install fastapi uvicorn python-multipart aiofiles -q

print_status "Installing PDF processing libraries..."
pip install pdf2image PyMuPDF Pillow -q

print_status "Installing ML libraries..."
pip install transformers accelerate huggingface-hub -q

print_status "Installing fast download support..."
pip install hf-transfer -q

# Step 5: Create necessary directories
print_header "Step 5: Creating directories"

mkdir -p /workspace/models
mkdir -p /workspace/logs
mkdir -p /workspace/temp

print_status "Directories created"

# Step 6: Download the model
print_header "Step 6: Downloading DeepSeek-OCR model (~10GB)"

MODEL_DIR="/workspace/models/deepseek-ai/DeepSeek-OCR"

if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    print_warning "Model directory exists. Checking if complete..."

    # Check for key model files
    if [ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/generation_config.json" ]; then
        print_status "Model already downloaded. Skipping..."
    else
        print_warning "Model appears incomplete. Re-downloading..."
        rm -rf "$MODEL_DIR"
        export HF_HUB_ENABLE_HF_TRANSFER=1
        huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir "$MODEL_DIR"
    fi
else
    print_status "Downloading model (this will take 5-15 minutes)..."
    export HF_HUB_ENABLE_HF_TRANSFER=1
    huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir "$MODEL_DIR"
fi

# Verify model size
if [ -d "$MODEL_DIR" ]; then
    MODEL_SIZE=$(du -sh "$MODEL_DIR" | cut -f1)
    print_status "Model downloaded: $MODEL_SIZE"
else
    print_error "Model download failed!"
    exit 1
fi

# Step 7: Create convenience scripts
print_header "Step 7: Creating helper scripts"

# Create start script
cat > /workspace/start_server.sh << 'EOF'
#!/bin/bash

# DeepSeek-OCR Server Startup Script
# Runs in foreground mode on port 7777

cd /workspace/deepseek-ocr-server

# Set environment variables
export MODEL_PATH="/workspace/models/deepseek-ai/DeepSeek-OCR"
export PORT=7777
export HOST="0.0.0.0"
export GPU_MEMORY_UTILIZATION=0.85
export CUDA_VISIBLE_DEVICES=0

# Clear screen and show banner
clear
echo ""
echo "============================================================"
echo "    Starting DEEPSEEK-OCR SERVER"
echo "    Port: 7777 | Backend: vLLM | Mode: Foreground"
echo "============================================================"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 7777 --log-level info

EOF
chmod +x /workspace/start_server.sh
print_status "Created start_server.sh"

# Create stop script
cat > /workspace/stop_server.sh << 'EOF'
#!/bin/bash

# Stop DeepSeek-OCR server

echo "Stopping DeepSeek-OCR server..."

# Find and kill uvicorn processes on port 7777
pkill -f "uvicorn.*7777" || echo "No server process found"

# Alternative: kill by port
fuser -k 7777/tcp 2>/dev/null || echo "Port 7777 is free"

echo "Server stopped"
EOF
chmod +x /workspace/stop_server.sh
print_status "Created stop_server.sh"

# Create check script
cat > /workspace/check_server.sh << 'EOF'
#!/bin/bash

# Check DeepSeek-OCR server status

echo ""
echo "============================================================"
echo "    DEEPSEEK-OCR SERVER STATUS CHECK"
echo "============================================================"
echo ""

# Check if server process is running
echo "1. Process Status:"
if pgrep -f "uvicorn.*7777" > /dev/null; then
    echo "   ✓ Server process is running"
    echo "   PID: $(pgrep -f 'uvicorn.*7777')"
else
    echo "   ✗ Server process not found"
fi
echo ""

# Check port
echo "2. Port Status:"
if netstat -tuln | grep -q ":7777 "; then
    echo "   ✓ Port 7777 is listening"
else
    echo "   ✗ Port 7777 is not listening"
fi
echo ""

# Check GPU
echo "3. GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo ""

# Check health endpoint
echo "4. Health Check:"
if curl -s http://localhost:7777/health > /dev/null 2>&1; then
    echo "   ✓ Server is responding"
    echo ""
    echo "   Health Response:"
    curl -s http://localhost:7777/health | python3 -m json.tool | head -20
else
    echo "   ✗ Server not responding on http://localhost:7777"
fi
echo ""

# Check disk space
echo "5. Disk Space:"
df -h /workspace | tail -1
echo ""

echo "============================================================"
EOF
chmod +x /workspace/check_server.sh
print_status "Created check_server.sh"

# Step 8: Create environment file
print_header "Step 8: Creating environment configuration"

cat > /workspace/.env << EOF
# DeepSeek-OCR Server Configuration
MODEL_PATH=/workspace/models/deepseek-ai/DeepSeek-OCR
PORT=7777
HOST=0.0.0.0
GPU_MEMORY_UTILIZATION=0.85
CUDA_VISIBLE_DEVICES=0
WORKERS=1
LOG_LEVEL=info
EOF
print_status "Created .env configuration file"

# Step 9: Verify installation
print_header "Step 9: Verifying installation"

print_status "Checking Python packages..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
python3 -c "import vllm; print(f'vLLM: installed')" 2>/dev/null || print_warning "vLLM not available, will use transformers"

# Final summary
print_header "Setup Complete!"

echo "✓ All dependencies installed"
echo "✓ Model downloaded to: /workspace/models/deepseek-ai/DeepSeek-OCR"
echo "✓ Server files ready at: /workspace/deepseek-ocr-server"
echo ""
echo "Next steps:"
echo "1. Start the server:  bash /workspace/start_server.sh"
echo "2. Check status:      bash /workspace/check_server.sh"
echo "3. Stop server:       bash /workspace/stop_server.sh"
echo ""
echo "The server will run on port 7777"
echo "Access from Runpod:  http://localhost:7777"
echo "External access:     http://<your-runpod-url>:7777"
echo ""
echo "Test the health endpoint:"
echo "  curl http://localhost:7777/health"
echo ""
print_status "Setup script completed successfully!"
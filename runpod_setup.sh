#!/bin/bash

# Runpod Setup Script - Run this directly on the Runpod pod
# This script sets up the DeepSeek-OCR server environment

set -e

# Configuration
WORKSPACE="/workspace"
PROJECT_DIR="${WORKSPACE}/deepseek-ocr-server"
MODEL_DIR="${WORKSPACE}/models/deepseek-ai/DeepSeek-OCR"
VENV_DIR="${WORKSPACE}/venv"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check CUDA availability
check_cuda() {
    log_step "Checking CUDA availability..."

    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. CUDA may not be available."
        return 1
    fi

    nvidia-smi
    log_info "CUDA is available"
    return 0
}

# Install system dependencies
install_system_deps() {
    log_step "Installing system dependencies..."

    apt-get update
    apt-get install -y \
        python3.10 \
        python3.10-dev \
        python3-pip \
        python3.10-venv \
        poppler-utils \
        git \
        wget \
        curl \
        htop \
        tmux \
        vim

    log_info "System dependencies installed"
}

# Setup Python virtual environment
setup_venv() {
    log_step "Setting up Python virtual environment..."

    if [ -d "$VENV_DIR" ]; then
        log_warning "Virtual environment already exists, skipping..."
        return
    fi

    python3.10 -m venv $VENV_DIR
    source $VENV_DIR/bin/activate

    pip install --upgrade pip setuptools wheel
    log_info "Virtual environment created at $VENV_DIR"
}

# Install Python dependencies
install_python_deps() {
    log_step "Installing Python dependencies..."

    source $VENV_DIR/bin/activate

    # Install PyTorch with CUDA support
    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118

    # Install vLLM and flash-attention
    pip install vllm==0.8.5
    pip install flash-attn --no-build-isolation

    # Install other dependencies
    pip install \
        transformers>=4.51.1 \
        fastapi>=0.104.0 \
        uvicorn[standard]>=0.24.0 \
        python-multipart>=0.0.6 \
        aiohttp>=3.9.0 \
        pdf2image>=1.16.3 \
        PyMuPDF>=1.23.0 \
        Pillow>=10.0.0 \
        pydantic>=2.0.0 \
        pyyaml>=6.0 \
        python-dotenv>=1.0.0

    log_info "Python dependencies installed"
}

# Download model weights
download_model() {
    log_step "Downloading DeepSeek-OCR model weights..."

    if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR)" ]; then
        log_warning "Model directory already exists and is not empty"
        read -p "Do you want to re-download the model? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping model download"
            return
        fi
    fi

    mkdir -p $MODEL_DIR

    source $VENV_DIR/bin/activate
    pip install huggingface-hub

    # Download model (approximately 8-10GB)
    log_info "Downloading model (this may take 10-20 minutes)..."
    huggingface-cli download deepseek-ai/DeepSeek-OCR \
        --local-dir $MODEL_DIR \
        --local-dir-use-symlinks False

    log_info "Model downloaded to $MODEL_DIR"
    log_info "Model size: $(du -sh $MODEL_DIR | cut -f1)"
}

# Setup project files
setup_project() {
    log_step "Setting up project files..."

    if [ ! -d "$PROJECT_DIR" ]; then
        log_error "Project directory not found at $PROJECT_DIR"
        log_info "Please upload the server files first using:"
        log_info "  runpod send <pod-id>:/workspace/deepseek-ocr-server ./server/"
        return 1
    fi

    cd $PROJECT_DIR

    # Create necessary directories
    mkdir -p logs temp

    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        cat > .env << EOF
MODEL_PATH=${MODEL_DIR}
GPU_MEMORY_UTILIZATION=0.85
CUDA_VISIBLE_DEVICES=0
HOST=0.0.0.0
PORT=8000
WORKERS=1
LOG_LEVEL=INFO
EOF
        log_info "Created .env file"
    fi

    log_info "Project setup complete"
}

# Create systemd service (optional)
create_service() {
    log_step "Creating systemd service..."

    cat > /etc/systemd/system/deepseek-ocr.service << EOF
[Unit]
Description=DeepSeek-OCR Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=${PROJECT_DIR}
Environment="PATH=${VENV_DIR}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=${VENV_DIR}/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    log_info "Service created. Use 'systemctl start deepseek-ocr' to start"
}

# Create start script
create_start_script() {
    log_step "Creating start script..."

    cat > ${WORKSPACE}/start_server.sh << 'EOF'
#!/bin/bash
source /workspace/venv/bin/activate
cd /workspace/deepseek-ocr-server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
EOF

    chmod +x ${WORKSPACE}/start_server.sh
    log_info "Start script created at ${WORKSPACE}/start_server.sh"
}

# Test the server
test_server() {
    log_step "Testing server..."

    source $VENV_DIR/bin/activate
    cd $PROJECT_DIR

    # Start server in background
    log_info "Starting server for testing..."
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &
    SERVER_PID=$!

    # Wait for server to start
    log_info "Waiting for server to start..."
    sleep 30

    # Test health endpoint
    if curl -f http://localhost:8000/health; then
        log_info "Server is healthy!"
    else
        log_error "Server health check failed"
    fi

    # Stop test server
    kill $SERVER_PID
    log_info "Test complete"
}

# Main setup function
main() {
    echo "======================================"
    echo "DeepSeek-OCR Runpod Setup"
    echo "======================================"
    echo ""

    # Check if we're on Runpod
    if [ ! -d "$WORKSPACE" ]; then
        log_error "This script should be run on a Runpod pod"
        log_error "Workspace directory not found: $WORKSPACE"
        exit 1
    fi

    cd $WORKSPACE

    # Run setup steps
    check_cuda || log_warning "CUDA check failed, continuing anyway..."
    install_system_deps
    setup_venv
    install_python_deps
    download_model

    # Check if project files exist
    if [ -d "$PROJECT_DIR" ]; then
        setup_project
    else
        log_warning "Project files not found. Upload them using:"
        log_warning "  runpod send <pod-id>:/workspace/deepseek-ocr-server ./server/"
    fi

    create_start_script

    # Optional: create systemd service
    read -p "Do you want to create a systemd service? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        create_service
    fi

    # Optional: test the server
    read -p "Do you want to test the server now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        test_server
    fi

    echo ""
    echo "======================================"
    echo "Setup Complete!"
    echo "======================================"
    echo ""
    echo "To start the server:"
    echo "  bash ${WORKSPACE}/start_server.sh"
    echo ""
    echo "Or if you created the systemd service:"
    echo "  systemctl start deepseek-ocr"
    echo ""
    echo "Server will be available at:"
    echo "  http://<your-pod-ip>:8000"
    echo ""
    echo "Test with:"
    echo "  curl http://localhost:8000/health"
}

# Run main function
main
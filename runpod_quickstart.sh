#!/bin/bash

# Runpod Quick Start Script for Your Pod
# Pod: greasy_lime_clownfish
# Run this locally to upload files to your pod

set -e

# Your pod details (from your screenshot)
POD_ID="u2bn60prhjml75"  # Extracted from SSH username
POD_IP="213.173.109.80"
POD_SSH_PORT="16922"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}DeepSeek-OCR Runpod Deployment${NC}"
echo "================================"
echo "Pod: greasy_lime_clownfish"
echo "IP: ${POD_IP}"
echo ""

# Step 1: Create deployment package
echo -e "${GREEN}Step 1: Creating deployment package...${NC}"
mkdir -p deployment_package

# Copy server files
cp -r server/* deployment_package/
cp runpod_setup.sh deployment_package/

# Create simplified setup script
cat > deployment_package/quick_setup.sh << 'EOF'
#!/bin/bash

echo "Starting DeepSeek-OCR Setup on Runpod..."
cd /workspace

# Install system dependencies
apt-get update
apt-get install -y python3-pip python3-venv poppler-utils git

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install fastapi uvicorn python-multipart aiohttp pdf2image PyMuPDF Pillow pydantic pyyaml

# Try to install vLLM (may fail on some pods)
pip install vllm || echo "vLLM installation failed, will use transformers backend"

# Install transformers as fallback
pip install transformers accelerate

# Create directories
mkdir -p /workspace/deepseek-ocr-server
mkdir -p /workspace/models
mkdir -p /workspace/logs
mkdir -p /workspace/temp

echo "Setup complete! Now:"
echo "1. Upload server files to /workspace/deepseek-ocr-server"
echo "2. Download model with: huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir /workspace/models/deepseek-ai/DeepSeek-OCR"
echo "3. Start server: cd /workspace/deepseek-ocr-server && python -m uvicorn app.main:app --host 0.0.0.0 --port 8888"
EOF

chmod +x deployment_package/quick_setup.sh

# Step 2: Create upload instructions
echo -e "${GREEN}Step 2: Files ready for upload${NC}"
echo ""
echo "Now, open your Web Terminal and run these commands:"
echo ""
echo -e "${YELLOW}# In Web Terminal:${NC}"
echo "cd /workspace"
echo "mkdir -p deepseek-ocr-server"
echo ""
echo -e "${YELLOW}# Then, from your local machine, upload the files:${NC}"
echo "# Option 1: Using Runpod CLI (if installed)"
echo "runpod send ${POD_ID}:/workspace/ ./deployment_package/"
echo ""
echo "# Option 2: Using SCP (if you have SSH key setup)"
echo "scp -P ${POD_SSH_PORT} -r ./deployment_package/* root@${POD_IP}:/workspace/"
echo ""
echo "# Option 3: Manual upload via Jupyter Lab"
echo "# 1. Open Jupyter Lab at port 8888"
echo "# 2. Upload the deployment_package folder"
echo "# 3. Move files to /workspace/"
echo ""

# Step 3: Create alternative start script for port 8888
cat > deployment_package/start_server_8888.sh << 'EOF'
#!/bin/bash

# Start server on port 8888 (since it's already exposed)
source /workspace/venv/bin/activate
cd /workspace/deepseek-ocr-server

# Set environment variables
export MODEL_PATH=/workspace/models/deepseek-ai/DeepSeek-OCR
export GPU_MEMORY_UTILIZATION=0.85
export HOST=0.0.0.0
export PORT=8888  # Using Jupyter port

# Start server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8888 --workers 1
EOF

chmod +x deployment_package/start_server_8888.sh

echo -e "${GREEN}Step 3: Quick commands for Web Terminal${NC}"
echo ""
echo "Once in Web Terminal, run:"
echo "================================"
echo "# Quick setup"
echo "bash /workspace/quick_setup.sh"
echo ""
echo "# Download model (8-10GB)"
echo "pip install huggingface-hub"
echo "huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir /workspace/models/deepseek-ai/DeepSeek-OCR"
echo ""
echo "# Start server on port 8888"
echo "bash /workspace/start_server_8888.sh"
echo "================================"
echo ""
echo -e "${GREEN}Files are ready in: ./deployment_package/${NC}"
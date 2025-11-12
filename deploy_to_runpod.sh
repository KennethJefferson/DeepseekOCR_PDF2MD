#!/bin/bash

# DeepSeek-OCR Deployment Script for Runpod
# This script helps deploy the server to a Runpod pod

set -e

# Configuration
POD_ID="${RUNPOD_POD_ID:-}"
RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-docker.io}"
DOCKER_USERNAME="${DOCKER_USERNAME:-}"
IMAGE_NAME="deepseek-ocr-server"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check requirements
check_requirements() {
    print_status "Checking requirements..."

    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi

    if ! command -v runpod &> /dev/null; then
        print_warning "Runpod CLI not found. Installing..."
        pip install runpod
    fi
}

# Build Docker image
build_image() {
    print_status "Building Docker image..."
    cd server
    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
    cd ..
    print_status "Docker image built successfully"
}

# Push to registry
push_image() {
    if [ -z "$DOCKER_USERNAME" ]; then
        print_warning "DOCKER_USERNAME not set, skipping push to registry"
        return
    fi

    print_status "Pushing image to registry..."
    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${DOCKER_REGISTRY}/${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}
    docker push ${DOCKER_REGISTRY}/${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}
    print_status "Image pushed successfully"
}

# Deploy to Runpod
deploy_to_runpod() {
    if [ -z "$POD_ID" ]; then
        print_error "RUNPOD_POD_ID not set. Please set the pod ID"
        exit 1
    fi

    print_status "Deploying to Runpod pod: ${POD_ID}"

    # Option 1: Upload files directly
    if [ "$1" == "files" ]; then
        print_status "Uploading files to pod..."
        runpod send ${POD_ID}:/workspace/deepseek-ocr-server ./server/

        print_status "Creating start script on pod..."
        cat << 'EOF' > start_server.sh
#!/bin/bash
cd /workspace/deepseek-ocr-server
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
EOF
        runpod send ${POD_ID}:/workspace/start_server.sh ./start_server.sh
        rm start_server.sh

        print_status "Files uploaded. Connect to pod and run: bash /workspace/start_server.sh"

    # Option 2: Use Docker container
    elif [ "$1" == "docker" ]; then
        if [ -z "$DOCKER_USERNAME" ]; then
            print_error "DOCKER_USERNAME required for Docker deployment"
            exit 1
        fi

        print_status "Creating Docker run script..."
        cat << EOF > run_docker.sh
#!/bin/bash
docker pull ${DOCKER_REGISTRY}/${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}
docker run -d \
    --name deepseek-ocr \
    --gpus all \
    -p 8000:8000 \
    -v /workspace/models:/app/models \
    -e MODEL_PATH=/app/models/deepseek-ai/DeepSeek-OCR \
    -e GPU_MEMORY_UTILIZATION=0.85 \
    ${DOCKER_REGISTRY}/${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}
EOF

        runpod send ${POD_ID}:/workspace/run_docker.sh ./run_docker.sh
        rm run_docker.sh

        print_status "Docker script uploaded. Connect to pod and run: bash /workspace/run_docker.sh"
    fi
}

# Download model weights
download_model() {
    print_status "Creating model download script..."

    cat << 'EOF' > download_model.sh
#!/bin/bash
# Download DeepSeek-OCR model weights

MODEL_DIR="/workspace/models/deepseek-ai/DeepSeek-OCR"
mkdir -p ${MODEL_DIR}

echo "Installing huggingface-hub..."
pip install huggingface-hub

echo "Downloading DeepSeek-OCR model..."
huggingface-cli download deepseek-ai/DeepSeek-OCR \
    --local-dir ${MODEL_DIR} \
    --local-dir-use-symlinks False

echo "Model downloaded to ${MODEL_DIR}"
echo "Size: $(du -sh ${MODEL_DIR} | cut -f1)"
EOF

    if [ -n "$POD_ID" ]; then
        print_status "Uploading model download script to pod..."
        runpod send ${POD_ID}:/workspace/download_model.sh ./download_model.sh
        rm download_model.sh
        print_status "Script uploaded. Connect to pod and run: bash /workspace/download_model.sh"
    else
        print_status "Model download script created: download_model.sh"
    fi
}

# Main menu
show_menu() {
    echo ""
    echo "DeepSeek-OCR Runpod Deployment"
    echo "==============================="
    echo "1. Build Docker image locally"
    echo "2. Push image to registry"
    echo "3. Deploy files to Runpod"
    echo "4. Deploy Docker to Runpod"
    echo "5. Download model weights (on pod)"
    echo "6. Full deployment (build + push + deploy)"
    echo "7. Exit"
    echo ""
    read -p "Select option: " choice

    case $choice in
        1)
            check_requirements
            build_image
            ;;
        2)
            push_image
            ;;
        3)
            deploy_to_runpod "files"
            ;;
        4)
            deploy_to_runpod "docker"
            ;;
        5)
            download_model
            ;;
        6)
            check_requirements
            build_image
            push_image
            deploy_to_runpod "docker"
            download_model
            ;;
        7)
            exit 0
            ;;
        *)
            print_error "Invalid option"
            ;;
    esac

    show_menu
}

# Parse command line arguments
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --build           Build Docker image"
    echo "  --push            Push image to registry"
    echo "  --deploy-files    Deploy files to Runpod"
    echo "  --deploy-docker   Deploy using Docker"
    echo "  --download-model  Create model download script"
    echo "  --full            Full deployment"
    echo "  --help            Show this help"
    echo ""
    echo "Environment variables:"
    echo "  RUNPOD_POD_ID     Pod ID for deployment"
    echo "  RUNPOD_API_KEY    Runpod API key"
    echo "  DOCKER_USERNAME   Docker Hub username"
    echo "  DOCKER_REGISTRY   Docker registry (default: docker.io)"
    echo "  IMAGE_TAG         Docker image tag (default: latest)"
    exit 0
fi

# Handle command line options
if [ "$1" == "--build" ]; then
    check_requirements
    build_image
elif [ "$1" == "--push" ]; then
    push_image
elif [ "$1" == "--deploy-files" ]; then
    deploy_to_runpod "files"
elif [ "$1" == "--deploy-docker" ]; then
    deploy_to_runpod "docker"
elif [ "$1" == "--download-model" ]; then
    download_model
elif [ "$1" == "--full" ]; then
    check_requirements
    build_image
    push_image
    deploy_to_runpod "docker"
    download_model
else
    # Interactive menu
    show_menu
fi

print_status "Done!"
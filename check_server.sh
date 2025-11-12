#!/bin/bash

# DeepSeek-OCR Server Status Check Script
# Comprehensive diagnostic tool for the DeepSeek-OCR server

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print status
print_ok() {
    echo -e "   ${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "   ${RED}✗${NC} $1"
}

print_warning() {
    echo -e "   ${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "   ${BLUE}ℹ${NC} $1"
}

# Header
echo ""
echo "============================================================"
echo "    DEEPSEEK-OCR SERVER STATUS CHECK"
echo "    $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# 1. Process Status
echo "1. Process Status:"
echo "   ----------------"
if pgrep -f "uvicorn.*7777" > /dev/null; then
    PIDS=$(pgrep -f "uvicorn.*7777")
    print_ok "Server process is running"
    print_info "PID(s): $PIDS"

    # Get process details
    for PID in $PIDS; do
        if [ -f "/proc/$PID/cmdline" ]; then
            CMD=$(tr '\0' ' ' < /proc/$PID/cmdline | head -c 100)
            print_info "Command: $CMD..."
        fi
    done

    # Get process uptime
    for PID in $PIDS; do
        if [ -f "/proc/$PID/stat" ]; then
            STARTTIME=$(awk '{print $22}' /proc/$PID/stat)
            UPTIME=$(ps -o etime= -p $PID 2>/dev/null | xargs)
            if [ ! -z "$UPTIME" ]; then
                print_info "Uptime: $UPTIME"
            fi
        fi
    done
else
    print_error "Server process not found"
    print_info "Start server with: bash start_server.sh"
fi
echo ""

# 2. Port Status
echo "2. Port Status:"
echo "   ------------"
if netstat -tuln 2>/dev/null | grep -q ":7777 "; then
    print_ok "Port 7777 is listening"

    # Get more details about the port
    LISTEN_INFO=$(netstat -tulnp 2>/dev/null | grep ":7777 " | head -1)
    if [ ! -z "$LISTEN_INFO" ]; then
        print_info "Binding: 0.0.0.0:7777 (accessible from all interfaces)"
    fi
else
    print_error "Port 7777 is not listening"

    # Check if port is in use by something else
    if fuser 7777/tcp 2>/dev/null > /dev/null; then
        print_warning "Port 7777 is in use by another process"
    else
        print_info "Port 7777 is free"
    fi
fi
echo ""

# 3. GPU Status
echo "3. GPU Status:"
echo "   ----------"
if command -v nvidia-smi &> /dev/null; then
    # Get GPU info
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null)

    if [ ! -z "$GPU_INFO" ]; then
        print_ok "GPU detected"

        # Parse GPU information
        IFS=',' read -r GPU_NAME MEM_TOTAL MEM_USED MEM_FREE GPU_UTIL GPU_TEMP <<< "$GPU_INFO"

        print_info "Model: $GPU_NAME"
        print_info "Memory: $MEM_USED used / $MEM_TOTAL total ($MEM_FREE free)"
        print_info "Utilization:$GPU_UTIL"
        print_info "Temperature:$GPU_TEMP"

        # Check if memory usage is high
        MEM_USED_NUM=$(echo "$MEM_USED" | sed 's/[^0-9]//g')
        MEM_TOTAL_NUM=$(echo "$MEM_TOTAL" | sed 's/[^0-9]//g')

        if [ "$MEM_USED_NUM" -gt 0 ] && [ "$MEM_TOTAL_NUM" -gt 0 ]; then
            MEM_PERCENT=$((MEM_USED_NUM * 100 / MEM_TOTAL_NUM))
            if [ "$MEM_PERCENT" -gt 80 ]; then
                print_warning "GPU memory usage is high (${MEM_PERCENT}%)"
            fi
        fi
    else
        print_error "Unable to query GPU information"
    fi
else
    print_error "nvidia-smi not found - GPU may not be available"
fi
echo ""

# 4. Model Status
echo "4. Model Status:"
echo "   ------------"
MODEL_PATH="/workspace/models/deepseek-ai/DeepSeek-OCR"
if [ -d "$MODEL_PATH" ]; then
    print_ok "Model directory exists"

    # Check key model files
    if [ -f "$MODEL_PATH/config.json" ]; then
        print_ok "config.json found"
    else
        print_error "config.json missing"
    fi

    if [ -f "$MODEL_PATH/generation_config.json" ]; then
        print_ok "generation_config.json found"
    else
        print_warning "generation_config.json missing"
    fi

    # Check model size
    MODEL_SIZE=$(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1)
    if [ ! -z "$MODEL_SIZE" ]; then
        print_info "Model size: $MODEL_SIZE"

        # Check if size seems correct (should be ~8-10GB)
        SIZE_NUM=$(echo "$MODEL_SIZE" | sed 's/G$//' | sed 's/\..*//')
        if [[ "$MODEL_SIZE" == *"G"* ]] && [ "$SIZE_NUM" -lt 5 ]; then
            print_warning "Model seems small (expected ~8-10GB)"
        fi
    fi
else
    print_error "Model directory not found at $MODEL_PATH"
    print_info "Run setup script: bash runpod_terminal_setup.sh"
fi
echo ""

# 5. Health Check
echo "5. Health Check:"
echo "   ------------"
if pgrep -f "uvicorn.*7777" > /dev/null; then
    # Try to connect to health endpoint
    HEALTH_RESPONSE=$(curl -s -m 5 http://localhost:7777/health 2>/dev/null)
    CURL_EXIT=$?

    if [ $CURL_EXIT -eq 0 ] && [ ! -z "$HEALTH_RESPONSE" ]; then
        print_ok "Server is responding"

        # Parse JSON response
        if command -v python3 &> /dev/null; then
            echo ""
            echo "   Health Response:"
            echo "$HEALTH_RESPONSE" | python3 -c "
import json
import sys
try:
    data = json.load(sys.stdin)
    print(f'     Status: {data.get(\"status\", \"unknown\")}')
    print(f'     Model Loaded: {data.get(\"model_loaded\", False)}')
    print(f'     CUDA Available: {data.get(\"cuda_available\", False)}')
    if 'gpu_memory_free' in data:
        print(f'     GPU Memory Free: {data[\"gpu_memory_free\"]}')
    if 'uptime_seconds' in data:
        uptime = int(data['uptime_seconds'])
        hours = uptime // 3600
        minutes = (uptime % 3600) // 60
        seconds = uptime % 60
        print(f'     Uptime: {hours}h {minutes}m {seconds}s')
except:
    print('     [Unable to parse response]')
" 2>/dev/null || echo "     $HEALTH_RESPONSE" | head -100
        else
            echo "     $HEALTH_RESPONSE" | head -100
        fi
    else
        print_error "Server not responding on http://localhost:7777"

        if [ $CURL_EXIT -eq 28 ]; then
            print_warning "Connection timed out (server may be starting up)"
        elif [ $CURL_EXIT -eq 7 ]; then
            print_warning "Connection refused (server may not be fully started)"
        fi

        print_info "Server may still be loading the model (takes 30-60 seconds)"
    fi
else
    print_warning "Cannot check health - server not running"
fi
echo ""

# 6. API Endpoints Test
echo "6. API Endpoints:"
echo "   -------------"
if pgrep -f "uvicorn.*7777" > /dev/null && curl -s -m 2 http://localhost:7777/health > /dev/null 2>&1; then
    # Test root endpoint
    if curl -s -m 2 http://localhost:7777/ > /dev/null 2>&1; then
        print_ok "Root endpoint (/) accessible"
    else
        print_warning "Root endpoint (/) not accessible"
    fi

    # Test status endpoint
    if curl -s -m 2 http://localhost:7777/api/v1/status > /dev/null 2>&1; then
        print_ok "Status endpoint (/api/v1/status) accessible"

        # Get basic stats
        STATUS=$(curl -s http://localhost:7777/api/v1/status 2>/dev/null)
        if [ ! -z "$STATUS" ] && command -v python3 &> /dev/null; then
            echo "$STATUS" | python3 -c "
import json
import sys
try:
    data = json.load(sys.stdin)
    stats = data.get('stats', {})
    if stats:
        print(f'     Requests: {stats.get(\"total_requests\", 0)}')
        print(f'     Success Rate: {stats.get(\"success_rate\", 0):.1f}%')
except:
    pass
" 2>/dev/null
        fi
    else
        print_warning "Status endpoint not accessible"
    fi

    print_info "PDF endpoint: POST /api/v1/ocr/pdf"
    print_info "URL endpoint: POST /api/v1/ocr/pdf-url"
else
    print_warning "Cannot test endpoints - server not ready"
fi
echo ""

# 7. Disk Space
echo "7. Disk Space:"
echo "   ----------"
DISK_INFO=$(df -h /workspace | tail -1)
DISK_USAGE=$(echo "$DISK_INFO" | awk '{print $5}' | sed 's/%//')
DISK_AVAIL=$(echo "$DISK_INFO" | awk '{print $4}')
DISK_TOTAL=$(echo "$DISK_INFO" | awk '{print $2}')

if [ "$DISK_USAGE" -lt 80 ]; then
    print_ok "Disk usage: ${DISK_USAGE}% used"
elif [ "$DISK_USAGE" -lt 90 ]; then
    print_warning "Disk usage: ${DISK_USAGE}% used (getting full)"
else
    print_error "Disk usage: ${DISK_USAGE}% used (critically low space)"
fi

print_info "Available: $DISK_AVAIL / $DISK_TOTAL"

# Check temp directory
if [ -d "/workspace/temp" ]; then
    TEMP_COUNT=$(find /workspace/temp -type f 2>/dev/null | wc -l)
    if [ "$TEMP_COUNT" -gt 100 ]; then
        print_warning "Temp directory has $TEMP_COUNT files (consider cleaning)"
    elif [ "$TEMP_COUNT" -gt 0 ]; then
        print_info "Temp files: $TEMP_COUNT"
    fi
fi
echo ""

# 8. Python Dependencies
echo "8. Python Dependencies:"
echo "   -------------------"
if command -v python3 &> /dev/null; then
    # Check critical packages
    python3 -c "import torch; print('   ✓ PyTorch:', torch.__version__)" 2>/dev/null || print_error "PyTorch not installed"
    python3 -c "import fastapi; print('   ✓ FastAPI:', fastapi.__version__)" 2>/dev/null || print_error "FastAPI not installed"
    python3 -c "import vllm; print('   ✓ vLLM: installed')" 2>/dev/null || print_warning "vLLM not installed (using transformers fallback)"
    python3 -c "import pdf2image; print('   ✓ pdf2image: installed')" 2>/dev/null || print_error "pdf2image not installed"
else
    print_error "Python3 not found"
fi
echo ""

# 9. Recent Logs
echo "9. Recent Logs:"
echo "   -----------"
if [ -f "/workspace/logs/deepseek-ocr.log" ]; then
    print_info "Log file: /workspace/logs/deepseek-ocr.log"
    echo "   Last 5 lines:"
    tail -5 /workspace/logs/deepseek-ocr.log 2>/dev/null | sed 's/^/     /'
else
    print_info "No log file found at /workspace/logs/deepseek-ocr.log"

    # Check for uvicorn output in system logs
    if command -v journalctl &> /dev/null; then
        RECENT_LOGS=$(journalctl -u deepseek-ocr -n 3 --no-pager 2>/dev/null)
        if [ ! -z "$RECENT_LOGS" ]; then
            echo "   System logs:"
            echo "$RECENT_LOGS" | sed 's/^/     /'
        fi
    fi
fi
echo ""

# 10. Quick Actions
echo "10. Quick Actions:"
echo "    -------------"
if pgrep -f "uvicorn.*7777" > /dev/null; then
    print_info "Stop server:    bash stop_server.sh"
    print_info "View logs:      tail -f /workspace/logs/deepseek-ocr.log"
    print_info "Test PDF:       curl -X POST -F 'file=@test.pdf' http://localhost:7777/api/v1/ocr/pdf"
else
    print_info "Start server:   bash start_server.sh"
    print_info "Run setup:      bash runpod_terminal_setup.sh"
fi
echo ""

# Summary
echo "============================================================"
if pgrep -f "uvicorn.*7777" > /dev/null && curl -s -m 2 http://localhost:7777/health > /dev/null 2>&1; then
    echo -e "${GREEN}    SERVER STATUS: RUNNING${NC}"
    echo -e "    Access at: http://localhost:7777"
elif pgrep -f "uvicorn.*7777" > /dev/null; then
    echo -e "${YELLOW}    SERVER STATUS: STARTING${NC}"
    echo -e "    Server is running but not yet responding"
else
    echo -e "${RED}    SERVER STATUS: STOPPED${NC}"
    echo -e "    Run: bash start_server.sh"
fi
echo "============================================================"
echo ""
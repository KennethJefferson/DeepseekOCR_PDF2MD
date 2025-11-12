#!/bin/bash

# DeepSeek-OCR Server Stop Script
# Stops any running DeepSeek-OCR server instances on port 7777

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo ""
echo "============================================================"
echo "    Stopping DEEPSEEK-OCR SERVER"
echo "============================================================"
echo ""

# Function to check if server is running
check_server() {
    pgrep -f "uvicorn.*7777" > /dev/null 2>&1
    return $?
}

# Check if server is running
if check_server; then
    echo -e "${YELLOW}Found running server process(es)${NC}"

    # Get PIDs
    PIDS=$(pgrep -f "uvicorn.*7777")
    echo "Process IDs: $PIDS"
    echo ""

    # Kill processes
    echo "Stopping server..."
    pkill -f "uvicorn.*7777"

    # Wait a moment
    sleep 2

    # Check if successfully stopped
    if check_server; then
        echo -e "${RED}Warning: Server still running, using force kill${NC}"
        pkill -9 -f "uvicorn.*7777"
        sleep 1
    fi

    # Final check
    if check_server; then
        echo -e "${RED}Error: Failed to stop server${NC}"
        echo "You may need to manually kill the process:"
        echo "  kill -9 $PIDS"
        exit 1
    else
        echo -e "${GREEN}✓ Server stopped successfully${NC}"
    fi
else
    echo -e "${YELLOW}No running server found on port 7777${NC}"
fi

# Also check if port is still in use
if netstat -tuln 2>/dev/null | grep -q ":7777 "; then
    echo ""
    echo -e "${YELLOW}Port 7777 still appears to be in use${NC}"
    echo "Attempting to free the port..."
    fuser -k 7777/tcp 2>/dev/null
    sleep 1

    if netstat -tuln 2>/dev/null | grep -q ":7777 "; then
        echo -e "${RED}Warning: Port 7777 is still in use by another process${NC}"
    else
        echo -e "${GREEN}✓ Port 7777 is now free${NC}"
    fi
else
    echo -e "${GREEN}✓ Port 7777 is free${NC}"
fi

# Clean up any temporary files
if [ -d "/workspace/temp" ]; then
    echo ""
    echo "Cleaning temporary files..."
    find /workspace/temp -name "*.pdf" -o -name "*.png" -o -name "*.jpg" | head -20
    TEMP_COUNT=$(find /workspace/temp -type f | wc -l)
    if [ "$TEMP_COUNT" -gt 0 ]; then
        echo "Found $TEMP_COUNT temporary files"
        read -p "Delete temporary files? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -f /workspace/temp/*
            echo -e "${GREEN}✓ Temporary files cleaned${NC}"
        fi
    fi
fi

echo ""
echo "============================================================"
echo -e "${GREEN}Server shutdown complete${NC}"
echo "============================================================"
echo ""
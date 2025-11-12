#!/bin/bash

# Test script to verify your DeepSeek-OCR deployment

# Configuration
API_URL="${1:-http://greasy-lime-clownfish.runpod.io:8888}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "DeepSeek-OCR Deployment Test"
echo "============================"
echo "Testing API at: $API_URL"
echo ""

# Test 1: Health Check
echo -n "1. Health Check... "
HEALTH=$(curl -s -f "$API_URL/health" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PASSED${NC}"
    echo "   Response: $HEALTH"
else
    echo -e "${RED}✗ FAILED${NC}"
    echo "   Could not connect to API"
    echo "   Make sure the server is running on port 8888"
    exit 1
fi

# Test 2: API Root
echo -n "2. API Root... "
ROOT=$(curl -s -f "$API_URL/" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PASSED${NC}"
else
    echo -e "${YELLOW}⚠ WARNING${NC}"
fi

# Test 3: Status Endpoint
echo -n "3. Status Endpoint... "
STATUS=$(curl -s -f "$API_URL/api/v1/status" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PASSED${NC}"
    echo "   Stats: $STATUS"
else
    echo -e "${YELLOW}⚠ WARNING${NC}"
fi

# Test 4: Create a test PDF (if needed)
echo -n "4. Creating test PDF... "
if ! command -v convert &> /dev/null; then
    echo -e "${YELLOW}SKIPPED${NC} (ImageMagick not installed)"
    TEST_PDF=""
else
    # Create a simple test PDF with ImageMagick
    convert -size 200x200 xc:white -pointsize 20 -draw "text 50,100 'Test PDF'" test.pdf 2>/dev/null
    if [ -f "test.pdf" ]; then
        echo -e "${GREEN}✓ CREATED${NC}"
        TEST_PDF="test.pdf"
    else
        echo -e "${YELLOW}SKIPPED${NC}"
        TEST_PDF=""
    fi
fi

# Test 5: Process PDF (if test PDF exists)
if [ -n "$TEST_PDF" ]; then
    echo -n "5. Processing PDF... "
    RESPONSE=$(curl -s -X POST "$API_URL/api/v1/ocr/pdf" \
        -F "file=@$TEST_PDF" \
        -F "resolution=base" 2>/dev/null)

    if echo "$RESPONSE" | grep -q "success.*true"; then
        echo -e "${GREEN}✓ PASSED${NC}"
        echo "   PDF processed successfully!"
    else
        echo -e "${RED}✗ FAILED${NC}"
        echo "   Response: $RESPONSE"
    fi

    # Clean up
    rm -f test.pdf
else
    echo "5. Processing PDF... ${YELLOW}SKIPPED${NC}"
fi

echo ""
echo "============================"
echo "Test Summary:"
echo ""

# Check if server is fully operational
if echo "$HEALTH" | grep -q "model_loaded.*true"; then
    echo -e "${GREEN}✅ Server is READY!${NC}"
    echo ""
    echo "Your API endpoint: $API_URL"
    echo ""
    echo "To use with Go client:"
    echo "  ./deepseek-client -workers 4 -scan /path/to/pdfs -api $API_URL"
else
    echo -e "${YELLOW}⚠️  Server is running but model may not be loaded${NC}"
    echo ""
    echo "Check if model is downloaded:"
    echo "  ls -la /workspace/models/deepseek-ai/DeepSeek-OCR"
fi

echo ""
echo "============================"
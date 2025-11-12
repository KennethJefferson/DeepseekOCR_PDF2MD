#!/usr/bin/env python3
"""
Upload server files to Runpod via Jupyter Lab

Since you want to avoid SSH, this script creates a single Python file
that you can paste into Jupyter Lab to set up everything.
"""

import os
import base64
import json

def create_bundled_server():
    """Create a single Python script with all server code bundled"""

    print("Creating bundled server script...")

    # Read all server files
    server_files = {
        'main.py': 'server/app/main.py',
        'models.py': 'server/app/models.py',
        'ocr_service.py': 'server/app/ocr_service.py',
        'pdf_processor.py': 'server/app/pdf_processor.py',
    }

    bundle = []
    bundle.append("""#!/usr/bin/env python3
# DeepSeek-OCR Server Bundle - Paste this into Jupyter Lab
# This script will create all necessary server files

import os
import base64

def setup_server():
    print("Setting up DeepSeek-OCR server...")

    # Create directories
    os.makedirs('/workspace/deepseek-ocr-server/app', exist_ok=True)
    os.makedirs('/workspace/models', exist_ok=True)
    os.makedirs('/workspace/logs', exist_ok=True)
    os.makedirs('/workspace/temp', exist_ok=True)

    # Server files content (base64 encoded to preserve formatting)
    files = {
""")

    # Add each file as base64 encoded content
    for name, path in server_files.items():
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                encoded = base64.b64encode(content.encode()).decode()
                bundle.append(f'        "app/{name}": "{encoded}",')

    bundle.append("""    }

    # Write files
    for filepath, content in files.items():
        fullpath = f'/workspace/deepseek-ocr-server/{filepath}'
        decoded = base64.b64decode(content).decode('utf-8')
        with open(fullpath, 'w', encoding='utf-8') as f:
            f.write(decoded)
        print(f'Created: {fullpath}')

    # Create __init__.py
    with open('/workspace/deepseek-ocr-server/app/__init__.py', 'w') as f:
        f.write('\"\"\"DeepSeek-OCR Server\"\"\"\\n__version__ = "1.0.0"\\n')

    print("\\nServer files created successfully!")
    print("\\nNext steps:")
    print("1. Install dependencies:")
    print("   !pip install fastapi uvicorn python-multipart aiohttp pdf2image PyMuPDF Pillow pydantic pyyaml transformers accelerate")
    print("2. Download model:")
    print("   !huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir /workspace/models/deepseek-ai/DeepSeek-OCR")
    print("3. Start server:")
    print("   !cd /workspace/deepseek-ocr-server && python -m uvicorn app.main:app --host 0.0.0.0 --port 8888")

if __name__ == "__main__":
    setup_server()
""")

    # Write the bundle
    with open('jupyter_setup_bundle.py', 'w', encoding='utf-8') as f:
        f.write('\n'.join(bundle))

    print("Created: jupyter_setup_bundle.py")
    print("\nInstructions:")
    print("1. Open Jupyter Lab on port 8888 in your Runpod pod")
    print("2. Create a new Python notebook")
    print("3. Copy the entire content of jupyter_setup_bundle.py")
    print("4. Paste and run it in a notebook cell")
    print("5. The script will create all server files automatically")
    print("\nNo SSH or file transfer needed!")

if __name__ == "__main__":
    create_bundled_server()

    # Also create a simple requirements file for Jupyter
    with open('jupyter_requirements.txt', 'w') as f:
        f.write("""# Install these in Jupyter Lab
torch==2.6.0
torchvision==0.21.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
aiohttp>=3.9.0
pdf2image>=1.16.3
PyMuPDF>=1.23.0
Pillow>=10.0.0
pydantic>=2.0.0
pyyaml>=6.0
transformers>=4.51.1
accelerate
huggingface-hub
""")
    print("Created: jupyter_requirements.txt")
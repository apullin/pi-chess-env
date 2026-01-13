#!/bin/bash
# Install PyTorch with CUDA support if CUDA is available

if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support..."
    uv pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
else
    echo "No CUDA detected, using CPU PyTorch (already installed)"
fi

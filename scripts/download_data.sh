#!/bin/bash
# Download and preprocess industrial datasets for LafTJU-TII paper
# Usage: bash download_data.sh [cwru|cmapss|all]

set -e
cd "$(dirname "$0")"

SCRIPT_DIR="$(pwd)"
DATA_DIR="$(cd .. && pwd)/data"

mkdir -p "$DATA_DIR/cwru"
mkdir -p "$DATA_DIR/cmapss"

download_cwru() {
    echo "=== Downloading CWRU Bearing Dataset ==="
    echo "Note: CWRU data must be downloaded manually from:"
    echo "  https://engineering.case.edu/bearingdatacenter"
    echo ""
    echo "Expected directory structure after download:"
    echo "  $DATA_DIR/cwru/"
    echo "  ├── Normal/"
    echo "  │   ├── 97.mat (0 hp)"
    echo "  │   ├── 98.mat (1 hp)"
    echo "  │   └── 99.mat (2 hp)"
    echo "  ├── Ball_007/"
    echo "  │   ├── 118.mat, 119.mat, 120.mat (0/1/2 hp)"
    echo "  │   └── ..."
    echo "  ├── Ball_014/"
    echo "  ├── Ball_021/"
    echo "  ├── Inner_007/"
    echo "  ├── Inner_014/"
    echo "  ├── Inner_021/"
    echo "  ├── Outer_007/"
    echo "  ├── Outer_014/"
    echo "  └── Outer_021/"
    echo ""
    echo "Place .mat files in the appropriate subdirectories under $DATA_DIR/cwru/"
}

download_cmapss() {
    echo "=== Downloading NASA C-MAPSS Dataset ==="
    echo "Note: C-MAPSS data can be downloaded from:"
    echo "  https://data.nasa.gov/Aerospace/C-MAPSS-Aircraft-Engine-Simulator-Data/vrks-gjaa"
    echo ""
    echo "Expected files after download and extraction:"
    echo "  $DATA_DIR/cmapss/"
    echo "  ├── train_FD001.txt"
    echo "  ├── test_FD001.txt"
    echo "  ├── RUL_FD001.txt"
    echo "  ├── train_FD002.txt"
    echo "  ├── test_FD002.txt"
    echo "  ├── RUL_FD002.txt"
    echo "  ├── train_FD003.txt"
    echo "  ├── test_FD003.txt"
    echo "  ├── RUL_FD003.txt"
    echo "  ├── train_FD004.txt"
    echo "  ├── test_FD004.txt"
    echo "  └── RUL_FD004.txt"
    echo ""
    echo "Place .txt files in $DATA_DIR/cmapss/"
}

# Auto-download C-MAPSS if possible
try_auto_cmapss() {
    echo "Attempting auto-download of C-MAPSS..."
    pip install -q kaggle 2>/dev/null || true

    # Try direct download from NASA
    CMAPSS_URL="https://ti.arc.nasa.gov/c/6/"
    echo "If auto-download fails, download manually from: $CMAPSS_URL"

    if command -v wget &> /dev/null; then
        echo "Trying wget..."
        wget -q --show-progress -O "$DATA_DIR/cmapss/CMAPSSData.zip" "$CMAPSS_URL" && \
        cd "$DATA_DIR/cmapss" && unzip -o CMAPSSData.zip && rm CMAPSSData.zip && \
        echo "C-MAPSS downloaded successfully!" && return 0
    fi

    echo "Auto-download failed. Please download manually."
    return 1
}

case "${1:-all}" in
    cwru)   download_cwru ;;
    cmapss) download_cmapss; try_auto_cmapss ;;
    all)    download_cwru; download_cmapss; try_auto_cmapss ;;
    *)      echo "Usage: $0 [cwru|cmapss|all]"; exit 1 ;;
esac

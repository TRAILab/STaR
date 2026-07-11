#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

WEIGHTS_DIR="$PROJECT_ROOT/weights"
while [[ $# -gt 0 ]]; do
    case $1 in
        --dir)
            WEIGHTS_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$WEIGHTS_DIR"
echo "Installing weights to $WEIGHTS_DIR"

echo "downloading Tag2Text weights"
wget -nc https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/tag2text_swin_14m.pth -P "$WEIGHTS_DIR"
echo "downloading GroundingDINO weights"
wget -nc https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P "$WEIGHTS_DIR"
echo "downloading TAP weights"
wget -nc https://huggingface.co/BAAI/tokenize-anything/resolve/main/models/tap_vit_l_v1_0.pkl -P "$WEIGHTS_DIR"
echo "downloading TAP merged"
wget -nc https://huggingface.co/BAAI/tokenize-anything/resolve/main/concepts/merged_2560.pkl -P "$WEIGHTS_DIR"
echo "downloading 4DMOS 10_scans"
wget -nc https://www.ipb.uni-bonn.de/html/projects/4DMOS/10_scans.zip -P "$WEIGHTS_DIR"
echo "downloading SBert repo"
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 "$WEIGHTS_DIR/all-MiniLM-L6-v2"
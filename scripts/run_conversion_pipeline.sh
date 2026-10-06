#!/usr/bin/env bash
#
# Runs the three-stage document conversion pipeline end-to-end:
#   1. convert  - raw documents -> markdown
#   2. resolve  - bbox placeholders -> cropped images
#   3. caption  - resolved images -> captions
#
# Assumes the 'doc2md' conda environment is already activated and 
# OCR and VLM models are served as per defaults in the scripts.
#
# Usage:
#   bash scripts/run_conversion_pipeline.sh -i <input-dir> [-o <output-dir>] [-f] [-v]
#
# To keep a record of a run for later diagnosis, redirect stdout/stderr to a
# timestamped log file:
#
#   mkdir -p data/local/logs
#   bash scripts/run_conversion_pipeline.sh \
#       -i data/examples/raw \
#       -o data/local/md/example \
#       --ocr-base-url http://localhost:8001 \
#       --caption-model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8 \
#       --caption-base-url http://localhost:8000 \
#       2>&1 | tee "data/logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
#
# Each stage prints an "=== Stage N: ... ===" marker, and the script exits
# immediately if any stage exits non-zero (set -e), so the end of the log
# file identifies which stage failed and why. Note: per-file failures inside
# a stage don't always cause a non-zero exit, so also grep the log for
# '| ERROR' to catch those.

set -euo pipefail

INPUT_DIR=""
OUTPUT_DIR=""
FORCE=""
VERBOSE=""
OCR_MODEL="lightonai/LightOnOCR-2-1B-bbox-soup"
OCR_BASE_URL="http://localhost:8002"
CAPTION_MODEL="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8"
CAPTION_BASE_URL="http://localhost:8000"
CONVERSION_METHOD="vlm"

[usage() {
    cat <<EOF
Usage: $(basename "$0") -i <input-dir> [options]

Required:
  -i, --input-dir DIR   Directory of raw documents for Stage 1

Options:
  -o, --output-dir DIR   Base output directory (created if missing)
  --ocr-model MODEL      non-default OCR model name (default: "lightonai/LightOnOCR-2-1B")
  --ocr-base-url URL     non-default base URL for OCR model server (default: "http://localhost:8002")
  --caption-model MODEL  non-default caption model name (default: "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8")
  --caption-base-url URL non-default base URL for caption model server (default: "http://localhost:8000")
  -f, --force            Reprocess existing outputs (passed to all stages)
  -v, --verbose           DEBUG-level logging (passed to all stages)
  -h, --help              Show this help and exit

Example (capture output to a timestamped log file):
  mkdir -p data/local/logs
  $(basename "$0") -i data/examples/raw \\
      -o data/examples/md/ \\
      --ocr-base-url http://localhost:8001 \\
      --caption-base-url http://localhost:8000 \\
      2>&1 | tee "data/logs/pipeline_\$(date +%Y%m%d_%H%M%S).log"
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --ocr-model)
            OCR_MODEL="$2"
            shift 2
            ;;
        --ocr-base-url)
            OCR_BASE_URL="$2"
            shift 2
            ;;
        --caption-model)
            CAPTION_MODEL="$2"
            shift 2
            ;;
        --caption-base-url)
            CAPTION_BASE_URL="$2"
            shift 2
            ;;
        --conversion-method)
            CONVERSION_METHOD="$2"
            shift 2
            ;;
        -f|--force)
            FORCE="--force"
            shift
            ;;
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$INPUT_DIR" ]]; then
    echo "Error: --input-dir is required." >&2
    usage >&2
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: input directory does not exist: $INPUT_DIR" >&2
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "No output directory specified. Output dir will be inferred from input directory." >&2
    input_basename=$(basename "$INPUT_DIR")
    OUTPUT_DIR="data/local/md/${input_basename}"
    echo "Inferred output directory: $OUTPUT_DIR" >&2
fi

mkdir -p "$OUTPUT_DIR"
RESOLVED_DIR="${OUTPUT_DIR}/resolved"

echo "=== Running document conversion pipeline ==="
echo "=== Input directory: $INPUT_DIR ==="
echo "=== Output directory: $OUTPUT_DIR ==="
echo "=== Date: $(date +%Y-%m-%d_%H-%M-%S) ==="
echo "=== OCR model: $OCR_MODEL ==="
echo "=== OCR base URL: $OCR_BASE_URL ==="
echo "=== Caption model: $CAPTION_MODEL ==="
echo "=== Caption base URL: $CAPTION_BASE_URL ==="
echo "=== Stage 1: convert ==="


python -m doc2md.convert \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    --model "$OCR_MODEL" \
    --base-url "$OCR_BASE_URL" \
    --conversion-method "$CONVERSION_METHOD" \
    $FORCE $VERBOSE

echo "=== Stage 2: resolve ==="
python -m doc2md.resolve \
    -i "$OUTPUT_DIR" \
    $FORCE $VERBOSE

echo "=== Stage 3: caption ==="
python -m doc2md.caption \
    -i "$RESOLVED_DIR" \
    --model "$CAPTION_MODEL" \
    --base-url "$CAPTION_BASE_URL" \
    $FORCE $VERBOSE

echo "=== Pipeline completed successfully ==="
echo "=== Date: $(date +%Y-%m-%d_%H-%M-%S) ==="
echo "=== Processed files are in: $RESOLVED_DIR/captioned ==="

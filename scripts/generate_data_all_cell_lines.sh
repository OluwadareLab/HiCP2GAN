#!/bin/bash
#
# Orchestrates Hi-C data preparation by calling the standalone Python scripts in
# data/ (Read_Data.py, Downsample.py, Generate.py) in sequence — same logical
# steps as the original HiCARN-style workflow, one script per stage.
#
# Usage (from anywhere):
#   bash /path/to/HiCP2GAN/scripts/generate_data_all_cell_lines.sh
#
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# Data tree root (must match data/Arg_Parser.py via env)
ROOT_DIR="${HICP2GAN_DATA_ROOT:-Data/R64_down}"
export HICP2GAN_DATA_ROOT="$ROOT_DIR"

CELL_LINES=("GM12878" "NHEK" "K562" "HMEC")
HIGH_RES="10kb"
LOW_RES="40kb"
RATIO=64
CHUNK=40
STRIDE=40
BOUND=201
LR_CUTOFF=100
SCALE=1
POOL_TYPE="max"
MAP_QUALITY="MAPQGE30"
NORM_FILE="KRnorm"

PY="${PYTHON:-python3}"

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_dir() {
    if [ ! -d "$1" ]; then
        print_error "Directory not found: $1"
        return 1
    fi
    return 0
}

setup_raw_dir() {
    local cell_line=$1
    local raw_dir="${ROOT_DIR}/raw/${cell_line}"
    local source_dir=""

    print_info "Setting up raw directory for ${cell_line}..."

    if [ -d "$raw_dir" ] && [ "$(ls -A "$raw_dir" 2>/dev/null)" ]; then
        print_info "Raw directory already exists with data: $raw_dir"
        if find "$raw_dir" -name "*.RAWobserved" -type f 2>/dev/null | grep -q "10kb"; then
            print_info "Raw directory structure looks good, skipping setup"
            return 0
        fi
    fi

    source_dir="${ROOT_DIR}/${cell_line}/raw/${cell_line}"

    if [ ! -d "$source_dir" ]; then
        print_error "Source directory not found: $source_dir"
        print_warning "Please ensure raw data is extracted for ${cell_line}"
        print_info "Expected: ${ROOT_DIR}/${cell_line}/raw/${cell_line}/10kb_resolution_intrachromosomal/"
        return 1
    fi

    mkdir -p "$(dirname "$raw_dir")"

    if [ ! -e "$raw_dir" ]; then
        [ -L "$raw_dir" ] && rm "$raw_dir"
        ln -sfn "$(realpath "$source_dir")" "$raw_dir"
        print_info "Created symlink: ${raw_dir} -> ${source_dir}"
    else
        print_info "Raw directory already exists: $raw_dir"
    fi

    if [ -d "${raw_dir}/10kb_resolution_intrachromosomal" ]; then
        print_info "Verified: 10kb_resolution_intrachromosomal found in ${raw_dir}"
        local chr_count
        chr_count=$(find "${raw_dir}/10kb_resolution_intrachromosomal" -maxdepth 1 -type d -name "chr*" | wc -l)
        print_info "Found ${chr_count} chromosome directories"
        return 0
    fi

    print_error "10kb_resolution_intrachromosomal not found in ${raw_dir}"
    return 1
}

process_raw_data() {
    local cell_line=$1
    local mat_dir="${ROOT_DIR}/${cell_line}/mat"

    if [ -d "$mat_dir" ]; then
        local mat_file_count
        mat_file_count=$(find "$mat_dir" -name "*${HIGH_RES}.npz" -type f | wc -l)
        if [ "$mat_file_count" -gt 0 ]; then
            print_info "Mat files already exist for ${cell_line} (${mat_file_count} ${HIGH_RES} files), skipping Read_Data.py"
            return 0
        fi
    fi

    print_info "Processing raw data for ${cell_line}..."
    "$PY" "${REPO_ROOT}/data/Read_Data.py" \
        -c "${cell_line}" \
        -hr "${HIGH_RES}" \
        -q "${MAP_QUALITY}" \
        -n "${NORM_FILE}"

    if [ ! -d "$mat_dir" ]; then
        print_error "Mat directory not created: $mat_dir"
        return 1
    fi

    local mat_file_count
    mat_file_count=$(find "$mat_dir" -name "*${HIGH_RES}.npz" -type f | wc -l)
    if [ "$mat_file_count" -eq 0 ]; then
        print_error "No ${HIGH_RES} mat files in $mat_dir"
        return 1
    fi

    print_info "Read_Data.py finished for ${cell_line} (${mat_file_count} files)"
    return 0
}

downsample_data() {
    local cell_line=$1
    local mat_dir="${ROOT_DIR}/${cell_line}/mat"

    if [ -d "$mat_dir" ]; then
        local downsampled_count
        downsampled_count=$(find "$mat_dir" -name "*${LOW_RES}.npz" -type f | wc -l)
        if [ "$downsampled_count" -gt 0 ]; then
            print_info "Downsampled files exist for ${cell_line}, skipping Downsample.py"
            return 0
        fi
    fi

    print_info "Downsampling for ${cell_line}..."
    "$PY" "${REPO_ROOT}/data/Downsample.py" \
        -hr "${HIGH_RES}" \
        -lr "${LOW_RES}" \
        -r "${RATIO}" \
        -c "${cell_line}"

    local downsampled_count
    downsampled_count=$(find "$mat_dir" -name "*${LOW_RES}.npz" -type f | wc -l)
    if [ "$downsampled_count" -eq 0 ]; then
        print_error "No ${LOW_RES} mat files in $mat_dir"
        return 1
    fi

    print_info "Downsample.py finished for ${cell_line}"
    return 0
}

generate_dataset() {
    local cell_line=$1
    local dataset=$2
    local mat_dir="${ROOT_DIR}/${cell_line}/mat"
    local out_dir="${ROOT_DIR}/${cell_line}/data_40_40"
    local expected_filename="hicarn_${HIGH_RES}${LOW_RES}_c${CHUNK}_s${STRIDE}_b${BOUND}_nonpool_${dataset}.npz"

    if [ -f "${out_dir}/${expected_filename}" ]; then
        local file_size
        file_size=$(stat -f%z "${out_dir}/${expected_filename}" 2>/dev/null || stat -c%s "${out_dir}/${expected_filename}" 2>/dev/null || echo "0")
        if [ "$file_size" -gt 0 ]; then
            print_info "Already exists: ${expected_filename}, skipping Generate.py for ${dataset}"
            return 0
        fi
        print_warning "Empty file, regenerating: ${expected_filename}"
        rm -f "${out_dir}/${expected_filename}"
    fi

    print_info "Generate.py (${dataset}) for ${cell_line}..."

    local high_res_files low_res_files
    high_res_files=$(find "$mat_dir" -name "*${HIGH_RES}.npz" -type f | wc -l)
    low_res_files=$(find "$mat_dir" -name "*${LOW_RES}.npz" -type f | wc -l)

    if [ "$high_res_files" -eq 0 ] || [ "$low_res_files" -eq 0 ]; then
        print_error "Missing mat files: ${HIGH_RES}=${high_res_files}, ${LOW_RES}=${low_res_files}"
        return 1
    fi

    "$PY" "${REPO_ROOT}/data/Generate.py" \
        -hr "${HIGH_RES}" \
        -lr "${LOW_RES}" \
        -lrc "${LR_CUTOFF}" \
        -s "${dataset}" \
        -chunk "${CHUNK}" \
        -stride "${STRIDE}" \
        -bound "${BOUND}" \
        -scale "${SCALE}" \
        -type "${POOL_TYPE}" \
        -c "${cell_line}"

    if [ ! -f "${out_dir}/${expected_filename}" ]; then
        print_error "Expected output missing: ${out_dir}/${expected_filename}"
        return 1
    fi

    print_info "Generate.py finished: ${dataset} for ${cell_line}"
    return 0
}

main() {
    if [ ! -f "${REPO_ROOT}/data/Read_Data.py" ] || [ ! -f "${REPO_ROOT}/data/Downsample.py" ] || [ ! -f "${REPO_ROOT}/data/Generate.py" ]; then
        print_error "Missing data/*.py under ${REPO_ROOT}"
        exit 1
    fi

    print_info "Repository: ${REPO_ROOT}"
    print_info "HICP2GAN_DATA_ROOT: ${ROOT_DIR}"
    print_info "Cell lines: ${CELL_LINES[*]}"
    echo ""

    for cell_line in "${CELL_LINES[@]}"; do
        echo ""
        print_info "========== ${cell_line} =========="

        if ! setup_raw_dir "${cell_line}"; then
            print_warning "Skipping ${cell_line} (raw setup)"
            continue
        fi

        local raw_dir="${ROOT_DIR}/raw/${cell_line}"
        local raw_file_count
        raw_file_count=$(find "$raw_dir" -name "*.RAWobserved" -type f 2>/dev/null | grep -c "10kb" || echo "0")
        if [ "$raw_file_count" -eq 0 ]; then
            print_error "No 10kb RAWobserved files in ${raw_dir}"
            continue
        fi
        print_info "Found ${raw_file_count} 10kb RAWobserved files"

        if ! process_raw_data "${cell_line}"; then
            print_warning "Skipping ${cell_line} (Read_Data)"
            continue
        fi
        if ! downsample_data "${cell_line}"; then
            print_warning "Skipping ${cell_line} (Downsample)"
            continue
        fi
        if ! generate_dataset "${cell_line}" "train"; then
            print_warning "train split failed for ${cell_line}"
        fi
        if ! generate_dataset "${cell_line}" "valid"; then
            print_warning "valid split failed for ${cell_line}"
        fi
        local test_dataset="${cell_line}_test"
        if ! generate_dataset "${cell_line}" "${test_dataset}"; then
            print_warning "test split failed for ${cell_line}"
        fi

        print_info "Done ${cell_line}"
    done

    print_info "All cell lines processed."
}

main

#!/usr/bin/env bash
#
# smoke_dl.sh - Test deep-learning training job flow
#
# Usage:
#   ./scripts/smoke_dl.sh              # Train AAPL with 1 epoch (default)
#   ./scripts/smoke_dl.sh NVDA 3       # Train NVDA with 3 epochs
#   ./scripts/smoke_dl.sh --status     # Check device/dependency status only
#
# This script will:
#   1. Check if server is running (start if not)
#   2. Check deep-learning dependencies
#   3. Enqueue a training job
#   4. Poll until done (timeout 10 minutes)
#   5. Print last 50 logs on failure
#
# Environment variables:
#   DPOLARIS_BACKEND_PORT  - Server port (default: 8420)
#   DPOLARIS_DEVICE        - Device for deep-learning (auto|cpu|mps|cuda)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PORT="${DPOLARIS_BACKEND_PORT:-8420}"
HOST="${DPOLARIS_BACKEND_HOST:-127.0.0.1}"
BASE_URL="http://${HOST}:${PORT}"

# Default training parameters
DEFAULT_SYMBOL="AAPL"
DEFAULT_EPOCHS=1
DEFAULT_MODEL_TYPE="lstm"
POLL_INTERVAL=5
TIMEOUT_SECONDS=600  # 10 minutes

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_detail() { echo -e "${CYAN}[DETAIL]${NC} $*"; }

check_health() {
    curl -sf "${BASE_URL}/health" > /dev/null 2>&1
}

check_dependencies() {
    log_info "Checking deep-learning dependencies..."

    local device_info
    device_info=$(curl -sf "${BASE_URL}/api/deep-learning/device" 2>/dev/null) || {
        log_error "Failed to get device info. Is the server running?"
        return 1
    }

    echo "$device_info" | python3 -m json.tool 2>/dev/null

    # Check if deep learning is ready
    local dl_ready
    dl_ready=$(echo "$device_info" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('deep_learning_ready', False))" 2>/dev/null || echo "false")

    if [ "$dl_ready" != "True" ] && [ "$dl_ready" != "true" ]; then
        local torch_available
        local sklearn_available
        torch_available=$(echo "$device_info" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('torch_importable', False))" 2>/dev/null || echo "false")
        sklearn_available=$(echo "$device_info" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('sklearn_importable', False))" 2>/dev/null || echo "false")

        if [ "$torch_available" != "True" ] && [ "$torch_available" != "true" ]; then
            log_error "PyTorch is not available. Install with: pip install torch torchvision"
            return 1
        fi
        if [ "$sklearn_available" != "True" ] && [ "$sklearn_available" != "true" ]; then
            log_error "scikit-learn is not available. Install with: pip install scikit-learn"
            return 1
        fi
    fi

    local device
    device=$(echo "$device_info" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('device', 'unknown'))" 2>/dev/null || echo "unknown")
    log_success "Deep-learning ready. Device: ${device}"
    return 0
}

enqueue_training_job() {
    local symbol="$1"
    local epochs="$2"
    local model_type="$3"

    log_info "Enqueuing training job: ${symbol} (${model_type}, epochs=${epochs})"

    local response
    response=$(curl -sf -X POST "${BASE_URL}/api/jobs/deep-learning/train" \
        -H "Content-Type: application/json" \
        -d "{\"symbol\": \"${symbol}\", \"epochs\": ${epochs}, \"model_type\": \"${model_type}\"}" 2>&1) || {
        log_error "Failed to enqueue training job"
        echo "$response"
        return 1
    }

    local job_id
    job_id=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('id', ''))" 2>/dev/null || echo "")

    if [ -z "$job_id" ]; then
        log_error "Failed to get job ID from response:"
        echo "$response"
        return 1
    fi

    log_success "Job enqueued with ID: ${job_id}"
    echo "$job_id"
}

get_job_status() {
    local job_id="$1"
    curl -sf "${BASE_URL}/api/jobs/${job_id}" 2>/dev/null
}

poll_job() {
    local job_id="$1"
    local start_time
    start_time=$(date +%s)

    log_info "Polling job ${job_id} (timeout: ${TIMEOUT_SECONDS}s)..."

    while true; do
        local now
        now=$(date +%s)
        local elapsed=$((now - start_time))

        if [ $elapsed -ge $TIMEOUT_SECONDS ]; then
            log_error "Job timed out after ${TIMEOUT_SECONDS}s"
            return 1
        fi

        local job_info
        job_info=$(get_job_status "$job_id") || {
            log_error "Failed to get job status"
            return 1
        }

        local status
        status=$(echo "$job_info" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status', 'unknown'))" 2>/dev/null || echo "unknown")

        case "$status" in
            completed)
                log_success "Job completed successfully!"
                echo "$job_info" | python3 -m json.tool 2>/dev/null
                return 0
                ;;
            failed)
                log_error "Job failed!"
                print_job_logs "$job_info" 50
                return 1
                ;;
            queued|running)
                local progress
                progress=$(echo "$job_info" | python3 -c "import sys,json; d=json.load(sys.stdin); r=d.get('result') or {}; print(f\"{r.get('current_epoch', '?')}/{r.get('total_epochs', '?')} epochs\")" 2>/dev/null || echo "...")

                log_info "Status: ${status} | Progress: ${progress} | Elapsed: ${elapsed}s"
                sleep $POLL_INTERVAL
                ;;
            *)
                log_warn "Unknown status: ${status}"
                sleep $POLL_INTERVAL
                ;;
        esac
    done
}

print_job_logs() {
    local job_info="$1"
    local num_lines="${2:-50}"

    log_info "Last ${num_lines} log lines:"
    echo "----------------------------------------"

    echo "$job_info" | python3 -c "
import sys, json
d = json.load(sys.stdin)
logs = d.get('logs', [])
for log in logs[-${num_lines}:]:
    ts = log.get('timestamp', '')[:19] if log.get('timestamp') else ''
    msg = log.get('message', '')
    print(f'{ts} | {msg}')
" 2>/dev/null || echo "(logs unavailable)"

    echo "----------------------------------------"

    # Print error if present
    local error
    error=$(echo "$job_info" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error', '') or '')" 2>/dev/null || echo "")
    if [ -n "$error" ]; then
        log_error "Error: ${error}"
    fi
}

run_smoke_test() {
    local symbol="${1:-$DEFAULT_SYMBOL}"
    local epochs="${2:-$DEFAULT_EPOCHS}"
    local model_type="${3:-$DEFAULT_MODEL_TYPE}"

    echo ""
    echo "============================================"
    echo "  dPolaris Deep-Learning Smoke Test"
    echo "============================================"
    echo ""

    # Step 1: Check if server is running
    log_info "Step 1: Checking server health..."
    if ! check_health; then
        log_warn "Server is not running. Starting server..."
        if [ -x "${SCRIPT_DIR}/smoke_server.sh" ]; then
            "${SCRIPT_DIR}/smoke_server.sh" &
            local server_pid=$!
            sleep 5

            if ! check_health; then
                log_error "Failed to start server"
                kill $server_pid 2>/dev/null || true
                return 1
            fi
        else
            log_error "Server is not running and smoke_server.sh is not available"
            return 1
        fi
    fi
    log_success "Server is healthy"

    # Step 2: Check dependencies
    log_info "Step 2: Checking deep-learning dependencies..."
    if ! check_dependencies; then
        return 1
    fi

    # Step 3: Enqueue training job
    log_info "Step 3: Enqueuing training job..."
    local job_id
    job_id=$(enqueue_training_job "$symbol" "$epochs" "$model_type") || return 1

    # Step 4: Poll until done
    log_info "Step 4: Waiting for job completion..."
    if poll_job "$job_id"; then
        echo ""
        log_success "============================================"
        log_success "  SMOKE TEST PASSED"
        log_success "============================================"
        return 0
    else
        echo ""
        log_error "============================================"
        log_error "  SMOKE TEST FAILED"
        log_error "============================================"
        return 1
    fi
}

show_status_only() {
    log_info "Checking deep-learning status..."

    if ! check_health; then
        log_error "Server is not running"
        return 1
    fi

    check_dependencies
}

# Main entry point
main() {
    case "${1:-}" in
        --status)
            show_status_only
            ;;
        --help|-h)
            echo "Usage: $0 [SYMBOL] [EPOCHS] [--status|--help]"
            echo ""
            echo "Arguments:"
            echo "  SYMBOL    Stock symbol to train (default: AAPL)"
            echo "  EPOCHS    Number of training epochs (default: 1)"
            echo ""
            echo "Options:"
            echo "  --status  Check device/dependency status only"
            echo "  --help    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0              # Train AAPL with 1 epoch"
            echo "  $0 NVDA 3       # Train NVDA with 3 epochs"
            echo "  $0 --status     # Check dependencies only"
            echo ""
            echo "Environment:"
            echo "  DPOLARIS_BACKEND_PORT  Server port (default: 8420)"
            echo "  DPOLARIS_DEVICE        Deep-learning device (auto|cpu|mps|cuda)"
            ;;
        *)
            run_smoke_test "${1:-}" "${2:-}" "${3:-}"
            ;;
    esac
}

main "$@"

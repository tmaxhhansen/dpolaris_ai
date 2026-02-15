#!/usr/bin/env bash
#
# smoke_server.sh - Start the dPolaris backend server and wait for /health
#
# Usage:
#   ./scripts/smoke_server.sh         # Start server, wait for health, keep running
#   ./scripts/smoke_server.sh --check # Only check if server is already healthy
#   ./scripts/smoke_server.sh --stop  # Stop any running server
#
# Environment variables:
#   DPOLARIS_BACKEND_PORT  - Server port (default: 8420)
#   DPOLARIS_BACKEND_HOST  - Server host (default: 127.0.0.1)
#   DPOLARIS_DEVICE        - Device for deep-learning (auto|cpu|mps|cuda)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PORT="${DPOLARIS_BACKEND_PORT:-8420}"
HOST="${DPOLARIS_BACKEND_HOST:-127.0.0.1}"
BASE_URL="http://${HOST}:${PORT}"
HEALTH_URL="${BASE_URL}/health"
STATUS_URL="${BASE_URL}/api/status"
DEVICE_URL="${BASE_URL}/api/deep-learning/device"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

check_health() {
    curl -sf "${HEALTH_URL}" > /dev/null 2>&1
}

wait_for_health() {
    local max_attempts="${1:-60}"
    local attempt=1

    log_info "Waiting for server health at ${HEALTH_URL}..."

    while [ $attempt -le $max_attempts ]; do
        if check_health; then
            log_success "Server is healthy after ${attempt} attempts"
            return 0
        fi

        if [ $((attempt % 10)) -eq 0 ]; then
            log_info "Still waiting... (${attempt}/${max_attempts})"
        fi

        sleep 1
        attempt=$((attempt + 1))
    done

    log_error "Server did not become healthy after ${max_attempts} seconds"
    return 1
}

show_status() {
    log_info "Checking server status..."

    if ! check_health; then
        log_error "Server is not running or not healthy"
        return 1
    fi

    log_success "Health check passed"

    # Show /api/status
    log_info "API Status:"
    curl -sf "${STATUS_URL}" 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "(status unavailable)"

    # Show device info
    log_info "Device info:"
    curl -sf "${DEVICE_URL}" 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "(device info unavailable)"

    return 0
}

find_server_pid() {
    # Find Python process listening on the port
    lsof -ti tcp:"${PORT}" 2>/dev/null || true
}

stop_server() {
    local pids
    pids=$(find_server_pid)

    if [ -z "$pids" ]; then
        log_info "No server running on port ${PORT}"
        return 0
    fi

    log_info "Stopping server (PIDs: ${pids})..."
    echo "$pids" | xargs kill -TERM 2>/dev/null || true

    # Wait for graceful shutdown
    local attempt=1
    while [ $attempt -le 10 ]; do
        pids=$(find_server_pid)
        if [ -z "$pids" ]; then
            log_success "Server stopped"
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
    done

    # Force kill if still running
    pids=$(find_server_pid)
    if [ -n "$pids" ]; then
        log_warn "Force killing server..."
        echo "$pids" | xargs kill -9 2>/dev/null || true
    fi

    log_success "Server stopped"
}

start_server() {
    cd "$REPO_ROOT"

    # Check if server is already running
    if check_health; then
        log_info "Server is already running and healthy"
        show_status
        return 0
    fi

    # Check if port is in use
    local pids
    pids=$(find_server_pid)
    if [ -n "$pids" ]; then
        log_warn "Port ${PORT} is in use by PIDs: ${pids}"
        log_info "Stopping existing processes..."
        stop_server
        sleep 2
    fi

    # Activate virtual environment if present
    if [ -f ".venv/bin/activate" ]; then
        log_info "Activating virtual environment..."
        # shellcheck disable=SC1091
        source .venv/bin/activate
    elif [ -f "venv/bin/activate" ]; then
        log_info "Activating virtual environment..."
        # shellcheck disable=SC1091
        source venv/bin/activate
    fi

    # Set environment defaults
    export DPOLARIS_DEVICE="${DPOLARIS_DEVICE:-auto}"
    export PYTHONUNBUFFERED=1

    log_info "Starting dPolaris server..."
    log_info "  Port: ${PORT}"
    log_info "  Device: ${DPOLARIS_DEVICE}"
    log_info "  Working directory: ${REPO_ROOT}"

    # Start server in background
    python -m api.server &
    local server_pid=$!

    log_info "Server started with PID: ${server_pid}"

    # Wait for health
    if wait_for_health 60; then
        show_status
        log_success "Server is ready at ${BASE_URL}"
        log_info "Press Ctrl+C to stop the server"

        # Wait for server process
        wait $server_pid 2>/dev/null || true
    else
        log_error "Server failed to start"
        kill $server_pid 2>/dev/null || true
        return 1
    fi
}

# Main entry point
main() {
    case "${1:-}" in
        --check)
            show_status
            ;;
        --stop)
            stop_server
            ;;
        --help|-h)
            echo "Usage: $0 [--check|--stop|--help]"
            echo ""
            echo "Options:"
            echo "  --check   Check if server is healthy"
            echo "  --stop    Stop any running server"
            echo "  --help    Show this help message"
            echo ""
            echo "Environment:"
            echo "  DPOLARIS_BACKEND_PORT  Server port (default: 8420)"
            echo "  DPOLARIS_BACKEND_HOST  Server host (default: 127.0.0.1)"
            echo "  DPOLARIS_DEVICE        Deep-learning device (auto|cpu|mps|cuda)"
            ;;
        *)
            start_server
            ;;
    esac
}

main "$@"

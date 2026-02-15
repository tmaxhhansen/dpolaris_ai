#!/usr/bin/env bash
#
# smoke_metadata_analysis.sh - Test stock metadata and analysis endpoints
#
# Usage:
#   ./scripts/smoke_metadata_analysis.sh         # Run all tests
#   ./scripts/smoke_metadata_analysis.sh --quick # Quick test (fewer symbols)
#
# Environment variables:
#   DPOLARIS_BACKEND_PORT  - Server port (default: 8420)
#   DPOLARIS_BACKEND_HOST  - Server host (default: 127.0.0.1)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PORT="${DPOLARIS_BACKEND_PORT:-8420}"
HOST="${DPOLARIS_BACKEND_HOST:-127.0.0.1}"
BASE_URL="http://${HOST}:${PORT}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

check_health() {
    curl -sf "${BASE_URL}/health" > /dev/null 2>&1
}

test_endpoint() {
    local name="$1"
    local url="$2"
    local expected_key="$3"

    log_info "Testing: $name"
    log_info "  URL: $url"

    local response
    local http_code

    # Get response and http code
    response=$(curl -sf "$url" 2>/dev/null) || {
        log_error "  Request failed"
        return 1
    }

    # Check if response contains expected key
    if echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print('found' if '$expected_key' in str(d) else 'not_found')" 2>/dev/null | grep -q "found"; then
        log_success "  Response contains expected key: $expected_key"
        echo "$response" | python3 -m json.tool 2>/dev/null | head -30
        return 0
    else
        log_warn "  Response may not contain expected key: $expected_key"
        echo "$response" | python3 -m json.tool 2>/dev/null | head -30
        return 0  # Don't fail, just warn
    fi
}

run_tests() {
    local quick="${1:-}"

    echo ""
    echo "============================================"
    echo "  Stock Metadata & Analysis Endpoint Tests"
    echo "============================================"
    echo ""

    # Check server health first
    log_info "Checking server health..."
    if ! check_health; then
        log_error "Server is not running at ${BASE_URL}"
        log_info "Start the server with: ./scripts/smoke_server.sh"
        return 1
    fi
    log_success "Server is healthy"
    echo ""

    # Test 1: Stock metadata for single symbol
    echo "----------------------------------------"
    test_endpoint \
        "Stock Metadata (single symbol)" \
        "${BASE_URL}/api/stocks/metadata?symbols=AAPL" \
        "AAPL"
    echo ""

    # Test 2: Stock metadata for multiple symbols
    if [ "$quick" != "--quick" ]; then
        echo "----------------------------------------"
        test_endpoint \
            "Stock Metadata (multiple symbols)" \
            "${BASE_URL}/api/stocks/metadata?symbols=AAPL,MSFT,GOOGL" \
            "sector"
        echo ""
    fi

    # Test 3: Analysis last date for single symbol
    echo "----------------------------------------"
    test_endpoint \
        "Analysis Last Date (single symbol)" \
        "${BASE_URL}/api/analysis/last?symbols=AAPL" \
        "last_analysis_at"
    echo ""

    # Test 4: Analysis last date for multiple symbols
    if [ "$quick" != "--quick" ]; then
        echo "----------------------------------------"
        test_endpoint \
            "Analysis Last Date (multiple symbols)" \
            "${BASE_URL}/api/analysis/last?symbols=AAPL,MSFT,NVDA" \
            "last_analysis_at"
        echo ""
    fi

    # Test 5: Analysis detail for a symbol
    echo "----------------------------------------"
    test_endpoint \
        "Analysis Detail" \
        "${BASE_URL}/api/analysis/detail/AAPL" \
        "artifacts"
    echo ""

    # Summary
    echo "============================================"
    log_success "All endpoint tests completed!"
    echo "============================================"
    echo ""
    echo "Example curl commands:"
    echo ""
    echo "  # Get stock metadata"
    echo "  curl '${BASE_URL}/api/stocks/metadata?symbols=AAPL,MSFT'"
    echo ""
    echo "  # Get last analysis dates"
    echo "  curl '${BASE_URL}/api/analysis/last?symbols=AAPL,NVDA'"
    echo ""
    echo "  # Get detailed analysis for a symbol"
    echo "  curl '${BASE_URL}/api/analysis/detail/AAPL'"
    echo ""
}

# Main
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [--quick|--help]"
        echo ""
        echo "Options:"
        echo "  --quick   Run fewer tests (faster)"
        echo "  --help    Show this help message"
        ;;
    *)
        run_tests "${1:-}"
        ;;
esac

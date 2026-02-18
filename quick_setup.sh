#!/bin/bash
#===============================================================================
# EV-OneHot MCP Quick Setup Script
#===============================================================================
# This script sets up the complete environment for EV-OneHot MCP server.
# Protein fitness prediction combining evolutionary coupling and one-hot encoding.
#
# After cloning the repository, run this script to set everything up:
#   cd ev_onehot_mcp
#   bash quick_setup.sh
#
# Once setup is complete, register in Claude Code with the config shown at the end.
#
# Options:
#   --skip-env        Skip conda environment creation
#   --help            Show this help message
#===============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${SCRIPT_DIR}/env"
PYTHON_VERSION="3.10"

# Print banner
echo -e "${BLUE}"
echo "=============================================="
echo "     EV-OneHot MCP Quick Setup Script        "
echo "=============================================="
echo -e "${NC}"

# Helper functions
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check for conda/mamba
check_conda() {
    if command -v mamba &> /dev/null; then
        CONDA_CMD="mamba"
        info "Using mamba (faster package resolution)"
    elif command -v conda &> /dev/null; then
        CONDA_CMD="conda"
        info "Using conda"
    else
        error "Neither conda nor mamba found. Please install Miniconda or Mambaforge first."
        exit 1
    fi
}

# Parse arguments
SKIP_ENV=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-env) SKIP_ENV=true; shift ;;
        -h|--help)
            echo "Usage: ./quick_setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-env        Skip conda environment creation"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *) warn "Unknown option: $1"; shift ;;
    esac
done

# Check prerequisites
info "Checking prerequisites..."
check_conda
success "Prerequisites check passed"

# Step 1: Create conda environment
echo ""
echo -e "${BLUE}Step 1: Setting up conda environment${NC}"

# Fast path: use pre-packaged conda env from GitHub Releases
PACKED_ENV_URL="${PACKED_ENV_URL:-}"
PACKED_ENV_TAG="${PACKED_ENV_TAG:-envs-v1}"
PACKED_ENV_BASE="https://github.com/charlesxu90/ProteinMCP/releases/download/${PACKED_ENV_TAG}"

if [ "$SKIP_ENV" = true ]; then
    info "Skipping environment creation (--skip-env)"
elif [ -d "$ENV_DIR" ] && [ -f "$ENV_DIR/bin/python" ]; then
    info "Environment already exists at: $ENV_DIR"
elif [ "${USE_PACKED_ENVS:-}" = "1" ] || [ -n "$PACKED_ENV_URL" ]; then
    # Download and extract pre-packaged conda environment
    PACKED_ENV_URL="${PACKED_ENV_URL:-${PACKED_ENV_BASE}/ev_onehot_mcp-env.tar.gz}"
    info "Downloading pre-packaged environment from ${PACKED_ENV_URL}..."
    mkdir -p "$ENV_DIR"
    if wget -qO- "$PACKED_ENV_URL" | tar xzf - -C "$ENV_DIR"; then
        source "$ENV_DIR/bin/activate"
        conda-unpack 2>/dev/null || true
        success "Pre-packaged environment ready"
        SKIP_ENV=true
    else
        warn "Failed to download pre-packaged env, falling back to conda create..."
        rm -rf "$ENV_DIR"
        info "Creating conda environment with Python ${PYTHON_VERSION}..."
        $CONDA_CMD create -p "$ENV_DIR" python=${PYTHON_VERSION} -y
    fi
else
    info "Creating conda environment with Python ${PYTHON_VERSION}..."
    $CONDA_CMD create -p "$ENV_DIR" python=${PYTHON_VERSION} -y
fi

# Step 2: Install dependencies
echo ""
echo -e "${BLUE}Step 2: Installing dependencies${NC}"

if [ "$SKIP_ENV" = true ]; then
    info "Skipping dependency installation (--skip-env)"
else
    if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
        info "Installing from requirements.txt..."
        "${ENV_DIR}/bin/pip" install -r "${SCRIPT_DIR}/requirements.txt"
    else
        info "Installing core dependencies..."
        "${ENV_DIR}/bin/pip" install numpy pandas scikit-learn loguru
    fi

    info "Installing fastmcp..."
    "${ENV_DIR}/bin/pip" install --ignore-installed fastmcp
    success "Dependencies installed"
fi

# Step 3: Verify installation
echo ""
echo -e "${BLUE}Step 3: Verifying installation${NC}"

"${ENV_DIR}/bin/python" -c "import fastmcp; import loguru; print('Core packages OK')" && success "Core packages verified" || error "Package verification failed"

# Print summary
echo ""
echo -e "${GREEN}=============================================="
echo "           Setup Complete!"
echo "==============================================${NC}"
echo ""
echo "Environment: $ENV_DIR"
echo ""
echo -e "${YELLOW}Claude Code Configuration:${NC}"
echo ""
cat << EOF
{
  "mcpServers": {
    "ev_onehot": {
      "command": "${ENV_DIR}/bin/python",
      "args": ["${SCRIPT_DIR}/src/ev_onehot_mcp.py"]
    }
  }
}
EOF
echo ""
echo "To add to Claude Code:"
echo "  claude mcp add ev_onehot -- ${ENV_DIR}/bin/python ${SCRIPT_DIR}/src/ev_onehot_mcp.py"
echo ""

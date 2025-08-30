#!/bin/bash

# Visual Product Matcher - Unix/Linux/macOS Startup Script

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

echo
echo "========================================"
echo "    Visual Product Matcher"
echo "    Unix/Linux/macOS Startup Script"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        print_error "Python is not installed"
        echo "Please install Python 3.8+ from https://python.org"
        echo "Or use your package manager:"
        echo "  Ubuntu/Debian: sudo apt install python3 python3-pip"
        echo "  CentOS/RHEL: sudo yum install python3 python3-pip"
        echo "  macOS: brew install python3"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

print_status "Python detected: $($PYTHON_CMD --version)"

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    print_error "Python 3.8 or higher is required"
    print_info "Current version: $PYTHON_VERSION"
    exit 1
fi

print_status "Python version: $PYTHON_VERSION"

# Check if virtual environment exists
if [ -f "venv/bin/activate" ]; then
    print_info "Activating virtual environment..."
    source venv/bin/activate
    print_status "Virtual environment activated"
elif [ -f "env/bin/activate" ]; then
    print_info "Activating virtual environment..."
    source env/bin/activate
    print_status "Virtual environment activated"
else
    print_info "No virtual environment found, using system Python"
fi

# Make startup script executable
chmod +x start_app.py

echo
print_status "Starting Visual Product Matcher..."
echo

# Start the application using the startup script
$PYTHON_CMD start_app.py --host 0.0.0.0

echo
print_info "Application stopped"

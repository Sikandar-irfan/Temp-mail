#!/bin/bash

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print with color
print_color() {
    color=$1
    message=$2
    printf "${color}${message}${NC}\n"
}

print_color $GREEN "Setting up TempMail Manager..."

# Check Python version
print_color $YELLOW "Checking Python version..."
if ! command -v python3 &>/dev/null; then
    print_color $RED "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if (( $(echo "$PYTHON_VERSION < 3.8" | bc -l) )); then
    print_color $RED "Python 3.8 or higher is required. Found version $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    print_color $YELLOW "Creating virtual environment..."
    python3 -m venv $VENV_DIR
fi

# Activate virtual environment
print_color $YELLOW "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Install Python packages
print_color $YELLOW "Installing Python packages..."
python -m pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
print_color $YELLOW "Creating necessary directories..."
mkdir -p ~/.tempmail/data
mkdir -p ~/.tempmail/logs

# Create activation script
print_color $YELLOW "Creating activation script..."
ACTIVATE_SCRIPT="$PWD/run.sh"
cat > $ACTIVATE_SCRIPT << EOL
#!/bin/bash
source "$PWD/$VENV_DIR/bin/activate"
python "$PWD/cli.py" "\$@"
EOL

chmod +x $ACTIVATE_SCRIPT

print_color $GREEN "Setup completed successfully!"
print_color $YELLOW "To run TempMail Manager:"
print_color $NC "1. Run ./run.sh in this directory"
print_color $NC "2. Your data will be stored in ~/.tempmail/"
print_color $NC "3. For development:"
print_color $NC "   - Activate the environment: source venv/bin/activate"
print_color $NC "   - Run tests: pytest tests/"
print_color $NC "   - Format code: black . && isort ."

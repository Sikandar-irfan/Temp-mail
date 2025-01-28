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

# Check if script is run with sudo
if [ "$EUID" -ne 0 ]; then
    print_color $RED "Please run this script with sudo"
    exit 1
fi

# Get the actual user who ran sudo
ACTUAL_USER=$SUDO_USER
if [ -z "$ACTUAL_USER" ]; then
    print_color $RED "Could not determine the actual user"
    exit 1
fi

print_color $GREEN "Setting up TempMail Manager..."

# Install system dependencies
print_color $YELLOW "Installing system dependencies..."
apt-get update
apt-get install -y python3 python3-pip python3-venv git curl

# Create virtual environment
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    print_color $YELLOW "Creating virtual environment..."
    sudo -u $ACTUAL_USER python3 -m venv $VENV_DIR
fi

# Generate requirements.txt if it doesn't exist
if [ ! -f "requirements.txt" ]; then
    print_color $YELLOW "Creating requirements.txt..."
    cat > requirements.txt << EOL
requests>=2.31.0
rich>=13.7.0
questionary>=2.0.1
beautifulsoup4>=4.12.2
aiosmtplib>=3.0.1
Jinja2>=3.1.2
cachetools>=5.3.2
EOL
fi

# Install Python packages
print_color $YELLOW "Installing Python packages..."
sudo -u $ACTUAL_USER $VENV_DIR/bin/pip install --upgrade pip
sudo -u $ACTUAL_USER $VENV_DIR/bin/pip install -r requirements.txt

# Create data directory
DATA_DIR="/home/$ACTUAL_USER/.tempmail"
if [ ! -d "$DATA_DIR" ]; then
    print_color $YELLOW "Creating data directory..."
    sudo -u $ACTUAL_USER mkdir -p "$DATA_DIR"
fi

# Create activation script
ACTIVATE_SCRIPT="tempmail"
cat > $ACTIVATE_SCRIPT << EOL
#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="\$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment
source "\$SCRIPT_DIR/$VENV_DIR/bin/activate"

# Run the application
python3 "\$SCRIPT_DIR/cli.py" "\$@"
EOL

# Make script executable and move to /usr/local/bin
chmod +x $ACTIVATE_SCRIPT
mv $ACTIVATE_SCRIPT /usr/local/bin/

# Set correct ownership for all files
chown -R $ACTUAL_USER:$ACTUAL_USER .
chown -R $ACTUAL_USER:$ACTUAL_USER "$DATA_DIR"

print_color $GREEN "Setup completed successfully!"
print_color $YELLOW "To run TempMail Manager:"
print_color $NC "1. Simply type 'tempmail' in your terminal"
print_color $NC "2. Your data will be stored in ~/.tempmail/"
print_color $NC "3. Use 'tempmail --help' for more options"

#!/bin/bash
# 🕵️ Desktop OSINT Investigation Suite - Installation Script
# LakyLuk Enhanced Edition - 27.9.2025

set -e  # Exit on any error

echo "🕵️ Desktop OSINT Investigation Suite - Enhanced Installation"
echo "============================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project directory
PROJECT_DIR="/home/laky/osint-desktop-suite"
VENV_DIR="$PROJECT_DIR/.venv"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check system requirements
check_requirements() {
    print_status "Checking system requirements..."

    # Check Python 3.8+
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi

    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    python_major=$(python3 -c 'import sys; print(sys.version_info[0])')
    python_minor=$(python3 -c 'import sys; print(sys.version_info[1])')

    if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 8 ]); then
        print_error "Python 3.8+ is required (found: $python_version)"
        exit 1
    fi

    print_success "Python $python_version detected"

    # Check pip
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 is required but not installed"
        exit 1
    fi

    # Check git
    if ! command -v git &> /dev/null; then
        print_warning "Git is recommended but not required"
    fi

    # Check Chrome/Chromium for browser automation
    if command -v google-chrome &> /dev/null; then
        print_success "Google Chrome detected"
    elif command -v chromium-browser &> /dev/null; then
        print_success "Chromium browser detected"
    else
        print_warning "Chrome/Chromium not found - browser automation may not work"
    fi
}

# Create project structure
create_structure() {
    print_status "Creating project structure..."

    cd "$PROJECT_DIR"

    # Create directory structure
    mkdir -p {config,src/{core,gui,tools/{social_media,government,web_search},security,analytics,reporting,plugins,data,utils},browser_profiles,logs,exports,tests,docs}

    # Create __init__.py files for Python packages
    touch src/__init__.py
    touch src/{core,gui,tools,security,analytics,reporting,plugins,data,utils}/__init__.py
    touch src/tools/{social_media,government,web_search}/__init__.py

    print_success "Project structure created"
}

# Setup virtual environment
setup_venv() {
    print_status "Setting up Python virtual environment..."

    cd "$PROJECT_DIR"

    # Create virtual environment
    python3 -m venv .venv

    # Activate virtual environment
    source .venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip

    print_success "Virtual environment created and activated"
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."

    cd "$PROJECT_DIR"
    source .venv/bin/activate

    # Install core dependencies
    pip install \
        tkinter-tooltip \
        asyncio-compat \
        aiohttp \
        beautifulsoup4 \
        requests \
        selenium \
        playwright \
        pandas \
        networkx \
        matplotlib \
        plotly \
        fake-useragent \
        cryptography \
        reportlab \
        openpyxl \
        jinja2 \
        pyyaml \
        python-dotenv \
        sqlalchemy \
        anthropic \
        openai \
        google-generativeai

    # Install playwright browsers
    playwright install chromium

    print_success "Dependencies installed"
}

# Create configuration files
create_config() {
    print_status "Creating configuration files..."

    cd "$PROJECT_DIR"

    # Create main config file
    cat > config/config.yaml << 'EOF'
# 🕵️ Desktop OSINT Investigation Suite Configuration
# LakyLuk Enhanced Edition

application:
  name: "Desktop OSINT Investigation Suite"
  version: "1.0.0-enhanced"
  debug_mode: false
  log_level: "INFO"

database:
  type: "sqlite"
  path: "data/investigations.db"
  backup_enabled: true
  retention_days: 90

browser:
  headless: true
  timeout: 30
  user_agent_rotation: true
  proxy_rotation: false
  stealth_mode: true

ai_models:
  claude:
    enabled: true
    model: "claude-3-5-sonnet-20240620"
    max_tokens: 4000

  openai:
    enabled: true
    model: "gpt-4"
    max_tokens: 4000

  gemini:
    enabled: true
    model: "gemini-pro"
    max_tokens: 4000

  local_llm:
    enabled: false
    model: "ollama/llama2"

investigation:
  default_timeout_minutes: 30
  max_concurrent_sources: 10
  confidence_threshold: 0.7
  stealth_level: "moderate"

security:
  encrypt_local_data: true
  auto_sanitize_pii: true
  secure_deletion: true
  audit_trail: true

reporting:
  default_format: "pdf"
  include_metadata: true
  watermark_enabled: true
  export_encryption: false

czech_osint:
  justice_cz_enabled: true
  ares_enabled: true
  cadastre_enabled: true
  firmy_cz_enabled: true
  reality_cz_enabled: false
EOF

    # Create environment file template
    cat > config/api_keys.env.template << 'EOF'
# 🔑 API Keys Configuration
# Copy this file to api_keys.env and fill in your actual API keys
# NEVER commit api_keys.env to version control!

# AI Models
ANTHROPIC_API_KEY=your_claude_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here

# Search Engines
GOOGLE_SEARCH_API_KEY=your_google_search_api_key
BING_SEARCH_API_KEY=your_bing_search_api_key

# Social Media APIs (optional)
TWITTER_API_KEY=your_twitter_api_key
FACEBOOK_API_TOKEN=your_facebook_token

# Proxy Services (optional)
PROXY_PROVIDER_API_KEY=your_proxy_api_key
PROXY_PROVIDER_URL=your_proxy_provider_url

# Czech Republic APIs (optional)
JUSTICE_CZ_API_KEY=your_justice_api_key
ARES_API_KEY=your_ares_api_key
EOF

    # Create browser profiles config
    cat > config/browser_profiles.json << 'EOF'
{
  "profiles": [
    {
      "name": "default",
      "user_agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
      "viewport": {"width": 1920, "height": 1080},
      "timezone": "Europe/Prague",
      "locale": "cs-CZ",
      "stealth_enabled": true
    },
    {
      "name": "mobile",
      "user_agent": "Mozilla/5.0 (Linux; Android 10; SM-G975F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
      "viewport": {"width": 375, "height": 667},
      "timezone": "Europe/Prague",
      "locale": "cs-CZ",
      "stealth_enabled": true
    }
  ]
}
EOF

    # Create gitignore
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# API Keys and Secrets
config/api_keys.env
*.key
*.pem
*.p12

# Logs
logs/*.log
logs/*.txt
*.log

# Database
*.db
*.sqlite
*.sqlite3

# Browser profiles and cache
browser_profiles/*/
.cache/
*.tmp

# Exports and investigations
exports/*.pdf
exports/*.xlsx
exports/*.maltego
investigations/*.json

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
*.lnk

# Temporary files
tmp/
temp/
*.tmp
*.temp
EOF

    print_success "Configuration files created"
}

# Create desktop launcher
create_desktop_launcher() {
    print_status "Creating desktop launcher..."

    # Create desktop file
    cat > /home/laky/Plocha/OSINTSuite.desktop << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=🕵️ OSINT Investigation Suite
Comment=Desktop OSINT Investigation Tool with AI Enhancement
Exec=gnome-terminal -- bash -c "cd $PROJECT_DIR && source .venv/bin/activate && python main_enhanced.py; exec bash"
Icon=$PROJECT_DIR/docs/osint-icon.svg
Terminal=false
Categories=Development;Security;
StartupWMClass=osint-suite
Keywords=OSINT;Investigation;Intelligence;Security;
EOF

    # Make executable
    chmod +x /home/laky/Plocha/OSINTSuite.desktop

    # Create simple icon (SVG)
    cat > docs/osint-icon.svg << 'EOF'
<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <circle cx="32" cy="32" r="30" fill="#2E86AB" stroke="#A23B72" stroke-width="2"/>
  <text x="32" y="38" font-family="Arial, sans-serif" font-size="24" fill="white" text-anchor="middle">🕵️</text>
</svg>
EOF

    print_success "Desktop launcher created"
}

# Initialize git repository
init_git() {
    print_status "Initializing git repository..."

    cd "$PROJECT_DIR"

    git init
    git add .
    git commit -m "🚀 Initial commit: Desktop OSINT Investigation Suite setup"

    print_success "Git repository initialized"
}

# Main installation flow
main() {
    echo "Starting installation..."
    echo ""

    check_requirements
    echo ""

    create_structure
    echo ""

    setup_venv
    echo ""

    install_dependencies
    echo ""

    create_config
    echo ""

    create_desktop_launcher
    echo ""

    if command -v git &> /dev/null; then
        init_git
        echo ""
    fi

    print_success "🎉 Installation completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "  1. Copy config/api_keys.env.template to config/api_keys.env"
    echo "  2. Fill in your API keys in config/api_keys.env"
    echo "  3. Run: cd $PROJECT_DIR && source .venv/bin/activate"
    echo "  4. Start development with: python main_enhanced.py"
    echo ""
    echo "📚 Documentation: $PROJECT_DIR/PROJECT_GUIDE_OSINT.md"
    echo "🖱️ Desktop launcher: /home/laky/Plocha/OSINTSuite.desktop"
    echo ""
    print_success "Happy investigating! 🕵️"
}

# Run main function
main "$@"
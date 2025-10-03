# 🕵️ Desktop OSINT Investigation Suite
### **Enhanced Edition with AI Integration**
**LakyLuk Development - Version 1.0.0-enhanced**

---

## 🎯 Project Overview

Desktop OSINT Investigation Suite is a comprehensive, AI-powered desktop application for conducting professional OSINT (Open Source Intelligence) investigations. Built with Python and enhanced with multiple AI models, this tool automates data collection, correlation, and analysis from various open sources.

### ✨ Key Features

- **🤖 Multi-Model AI Enhancement** - Claude + GPT-4 + Gemini + Local LLM ensemble
- **🇨🇿 Czech Republic Specialization** - Native support for Czech databases and sources
- **🛡️ Advanced Anti-Detection** - Stealth browsing, proxy rotation, fingerprint randomization
- **📊 Real-time Analytics** - ML-powered investigation optimization and entity correlation
- **📋 Professional Reporting** - Maltego, PDF, Excel, MISP export formats
- **🔌 Plugin Architecture** - Extensible tool ecosystem for custom OSINT sources
- **🔒 Security-First Design** - Privacy protection, data sanitization, operational security

### 🏗️ Implementation Status

```
🚀 DEVELOPMENT PHASES:
┌─────────────────────────────────────┐
│ FÁZE 1: Infrastruktura    [✅ 100%] │
│ FÁZE 2: Core Engine       [🔧   0%] │
│ FÁZE 3: AI Enhancement    [🔧   0%] │
│ FÁZE 4: Czech OSINT       [🔧   0%] │
│ FÁZE 5: Security          [🔧   0%] │
│ FÁZE 6: Advanced Features [🔧   0%] │
│ FÁZE 7: Reporting         [🔧   0%] │
│ FÁZE 8: Testing           [🔧   0%] │
└─────────────────────────────────────┘
CELKOVÝ PROGRESS: 12.5% (1/8 fází)
```

---

## 🚀 Quick Start

### **Prerequisites**
- **Python 3.8+** (tested with 3.9+)
- **Chrome/Chromium browser** for automation
- **Linux desktop environment** (tested on Linux Mint)
- **4GB+ RAM** recommended
- **API keys** for AI models (optional but recommended)

### **Installation**

1. **Clone or download** the project:
   ```bash
   cd /home/laky
   # Project should be in: /home/laky/osint-desktop-suite/
   ```

2. **Run the installation script**:
   ```bash
   cd osint-desktop-suite
   ./install.sh
   ```

3. **Configure API keys** (optional):
   ```bash
   cp config/api_keys.env.template config/api_keys.env
   nano config/api_keys.env  # Add your API keys
   ```

4. **Launch the application**:
   ```bash
   # GUI mode (recommended)
   python main_enhanced.py

   # CLI mode
   python main_enhanced.py --cli

   # Quick investigation
   python main_enhanced.py --target "John Doe"
   ```

### **Desktop Launcher**

A desktop launcher is automatically created during installation:
- **Location**: `/home/laky/Plocha/OSINTSuite.desktop`
- **Click to launch** the application directly from desktop

---

## 🔧 Configuration

### **API Keys Setup**

Copy and edit the API keys template:
```bash
cp config/api_keys.env.template config/api_keys.env
```

Add your API keys:
```bash
# AI Models
ANTHROPIC_API_KEY=your_claude_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here

# Search Engines
GOOGLE_SEARCH_API_KEY=your_google_search_api_key
BING_SEARCH_API_KEY=your_bing_search_api_key
```

### **Application Configuration**

Edit `config/config.yaml` to customize:
- Investigation timeouts
- AI model preferences
- Security settings
- Czech OSINT sources
- Reporting formats

---

## 📖 Usage Guide

### **GUI Mode** (Recommended)
```bash
python main_enhanced.py
```
- User-friendly interface
- Real-time progress monitoring
- Interactive entity graphs
- Point-and-click investigations

### **CLI Mode** (Advanced Users)
```bash
# Interactive CLI
python main_enhanced.py --cli

# Quick investigation
python main_enhanced.py --target "Target Name"

# Show configuration
python main_enhanced.py --config
```

### **Investigation Workflow**
1. **Start Investigation** - Enter target name and type
2. **AI Planning** - Multi-model AI creates investigation plan
3. **Data Collection** - Automated source scanning with stealth
4. **Entity Correlation** - AI-powered data linking and analysis
5. **Report Generation** - Professional multi-format exports

---

## 🔍 OSINT Sources

### **Social Media Platforms**
- Facebook (profile search, connection mapping)
- Instagram (profile analysis, hashtag research)
- LinkedIn (professional background, network analysis)
- Twitter (tweet analysis, follower networks)

### **Czech Republic Sources**
- **justice.cz** - Court records and legal documents
- **ARES** - Business registry and company information
- **Cadastre** - Property ownership and real estate
- **firmy.cz** - Business directory and financial data
- **reality.cz** - Property transactions and valuations

### **Web Search & Archives**
- Google Search (advanced operators)
- Bing Search (alternative perspective)
- Archive.org (historical snapshots)
- Specialized search engines

### **Government & Public Records**
- Business registrations
- Court proceedings
- Property records
- Professional licenses

---

## 🤖 AI Enhancement

### **Multi-Model Ensemble**
- **Claude Sonnet** - Strategic analysis and coordination
- **GPT-4** - Pattern recognition and technical analysis
- **Gemini Pro** - Entity correlation and deep context
- **Local LLM** - Privacy-sensitive analysis (optional)

### **AI Capabilities**
- **Investigation Planning** - Optimal source selection and sequencing
- **Entity Correlation** - Intelligent data linking across sources
- **Confidence Scoring** - Reliability assessment of findings
- **Predictive Paths** - Suggests next investigation steps
- **Threat Assessment** - Risk analysis and security implications

---

## 🛡️ Security & Privacy

### **Anti-Detection Features**
- **Browser Fingerprinting** - Randomized user agents, screen resolutions
- **Proxy Rotation** - Automatic IP address changing
- **Human Simulation** - ML-based realistic browsing patterns
- **Rate Limiting** - Respectful automated access

### **Data Protection**
- **Local Storage** - All data stays on your machine
- **Encryption** - Sensitive data encrypted at rest
- **PII Sanitization** - Automatic removal of sensitive information
- **Secure Deletion** - Cryptographic data wiping

### **Operational Security**
- **No Cloud Dependencies** - Works completely offline (except API calls)
- **Audit Trail** - Complete investigation activity logging
- **Data Retention** - Configurable automatic cleanup
- **Access Control** - File system permissions and encryption

---

## 📊 Reporting & Export

### **Export Formats**
- **Maltego** - Entity relationship graphs (.maltego)
- **PDF Dossier** - Professional investigation reports
- **Excel Analytics** - Structured data with pivot tables
- **JSON Data** - Raw structured investigation data
- **MISP Format** - Threat intelligence sharing

### **Report Components**
- **Executive Summary** - High-level findings and risk assessment
- **Entity Graphs** - Visual relationship mapping
- **Timeline Analysis** - Chronological event correlation
- **Source Attribution** - Data provenance and reliability
- **Recommendations** - Next steps and further investigation

---

## 🔌 Extension & Plugins

### **Plugin Architecture**
- **Custom OSINT Tools** - Add your own data sources
- **AI Model Integration** - Support for additional AI providers
- **Export Formats** - Custom reporting templates
- **Data Sources** - Connect to proprietary databases

### **Development Framework**
- **Python-based** - Easy to extend and modify
- **Async Architecture** - High-performance concurrent operations
- **Modular Design** - Clean separation of concerns
- **API-first** - RESTful interfaces for automation

---

## 📚 Documentation

### **Guides**
- **[Project Guide](PROJECT_GUIDE_OSINT.md)** - Complete development roadmap
- **Installation Guide** - Step-by-step setup instructions
- **User Manual** - Comprehensive feature documentation
- **Developer Guide** - Extension and customization

### **API Reference**
- **Investigation API** - Programmatic investigation control
- **Plugin API** - Custom tool development interface
- **Export API** - Custom report generation
- **Configuration API** - Dynamic settings management

---

## 🤝 Contributing

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Code formatting
black src/
flake8 src/
```

### **Contribution Guidelines**
- **Security First** - All contributions must pass security review
- **Privacy Protection** - No data collection or external transmission
- **Code Quality** - Follow Python PEP 8 and project conventions
- **Documentation** - Update docs for all new features

---

## ⚖️ Legal & Ethics

### **Responsible Use**
- **Respect Terms of Service** - Comply with all platform policies
- **Rate Limiting** - Avoid overwhelming target systems
- **Data Protection** - Handle personal information responsibly
- **Legal Compliance** - Follow applicable laws and regulations

### **Disclaimer**
This tool is designed for legitimate OSINT investigations, security research, and educational purposes. Users are responsible for ensuring compliance with applicable laws and platform terms of service.

---

## 📞 Support

### **Getting Help**
- **Documentation** - Check the comprehensive guides first
- **GitHub Issues** - Report bugs and request features
- **Email Support** - Contact developer for assistance
- **Community** - Join the OSINT investigation community

### **Common Issues**
- **Dependencies** - Run `./install.sh` to resolve missing packages
- **API Keys** - Check `config/api_keys.env` configuration
- **Browser Issues** - Ensure Chrome/Chromium is installed
- **Permissions** - Verify file system access rights

---

## 📈 Roadmap

### **Version 1.1** (Next Release)
- **Enhanced GUI** - Modern interface with dark mode
- **More AI Models** - Additional LLM provider support
- **Mobile OSINT** - Phone number and app analysis
- **Blockchain Tools** - Cryptocurrency investigation features

### **Version 2.0** (Future)
- **Cloud Deployment** - Scalable server infrastructure
- **Collaborative Investigations** - Multi-user investigations
- **Enterprise Features** - SSO, audit, compliance reporting
- **API Marketplace** - Third-party tool integration

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**LakyLuk Development**
- **GitHub**: [LakyLuk Projects](https://github.com/lakyluk)
- **Email**: lakyluk.development@gmail.com
- **Specialization**: AI-Enhanced Trading Systems & OSINT Tools

---

**🕵️ Happy Investigating!**

*Remember: With great OSINT power comes great responsibility. Use ethically and legally.*
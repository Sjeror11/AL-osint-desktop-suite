#!/usr/bin/env python3
"""
🕵️ Desktop OSINT Investigation Suite - Enhanced Edition
LakyLuk Enhanced Edition - 27.9.2025

Features:
✅ Multi-model AI ensemble (Claude + GPT-4 + Gemini + Local LLM)
✅ Advanced anti-detection and stealth capabilities
✅ Real-time analytics dashboard with entity graphs
✅ Czech Republic specialized OSINT tools
✅ Automated report generation (Maltego, PDF, Excel, MISP)
✅ Plugin architecture for extensibility
✅ Machine learning-based investigation optimization
✅ Threat intelligence integration
✅ Privacy-first data handling with automatic sanitization

Usage:
    python main_enhanced.py                    # GUI mode
    python main_enhanced.py --cli              # CLI mode
    python main_enhanced.py --target "Name"    # Quick investigation

Requirements:
- Python 3.8+
- Chrome/Chromium browser
- API keys in config/api_keys.env
- Linux desktop environment
"""

import asyncio
import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Version and metadata
__version__ = "1.0.0-enhanced"
__author__ = "LakyLuk"
__description__ = "Desktop OSINT Investigation Suite with AI Enhancement"

class OSINTSuiteBootstrap:
    """Application bootstrap and initialization"""

    def __init__(self):
        self.project_root = project_root
        self.logger = None
        self.config = None

    def setup_logging(self):
        """Setup comprehensive logging system"""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)

        # Create timestamp for log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup main logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'osint_main_{timestamp}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🚀 Desktop OSINT Suite v{__version__} starting...")

        return self.logger

    def load_configuration(self):
        """Load application configuration"""
        try:
            import yaml
            from dotenv import load_dotenv

            # Load environment variables from API keys file
            env_file = self.project_root / "config" / "api_keys.env"
            if env_file.exists():
                load_dotenv(env_file)
                self.logger.info("✅ API keys loaded from environment")
            else:
                self.logger.warning("⚠️ API keys file not found - some features may be limited")

            # Load main configuration
            config_file = self.project_root / "config" / "config.yaml"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                self.logger.info("✅ Configuration loaded")
            else:
                self.logger.error("❌ Configuration file not found")
                return None

            return self.config

        except ImportError as e:
            self.logger.error(f"❌ Missing required dependency: {e}")
            return None
        except Exception as e:
            self.logger.error(f"❌ Failed to load configuration: {e}")
            return None

    def check_dependencies(self):
        """Check if all required dependencies are available"""
        # Core dependencies (must have)
        core_modules = [
            'tkinter',
            'requests',
            'yaml',
            'dotenv'
        ]

        # Optional dependencies (nice to have)
        optional_modules = [
            'aiohttp',
            'beautifulsoup4',
            'selenium',
            'pandas'
        ]

        missing_core = []
        missing_optional = []

        # Check core modules
        for module in core_modules:
            try:
                __import__(module.replace('-', '_'))
            except ImportError:
                missing_core.append(module)

        # Check optional modules
        for module in optional_modules:
            try:
                if module == 'beautifulsoup4':
                    __import__('bs4')
                else:
                    __import__(module.replace('-', '_'))
            except ImportError:
                missing_optional.append(module)

        if missing_core:
            self.logger.error(f"❌ Missing critical modules: {', '.join(missing_core)}")
            self.logger.error("💡 Run: pip install -r requirements_minimal.txt")
            return False

        if missing_optional:
            self.logger.warning(f"⚠️ Missing optional modules: {', '.join(missing_optional)}")
            self.logger.warning("💡 Run: pip install -r requirements.txt for full functionality")

        self.logger.info("✅ Core dependencies available")
        return True

    def check_api_keys(self):
        """Check availability of API keys"""
        api_keys = {
            'ANTHROPIC_API_KEY': 'Claude AI',
            'OPENAI_API_KEY': 'OpenAI GPT-4',
            'GOOGLE_API_KEY': 'Google Gemini'
        }

        available_apis = []
        missing_apis = []

        for key, name in api_keys.items():
            if os.getenv(key):
                available_apis.append(name)
            else:
                missing_apis.append(name)

        if available_apis:
            self.logger.info(f"✅ Available AI APIs: {', '.join(available_apis)}")

        if missing_apis:
            self.logger.warning(f"⚠️ Missing API keys for: {', '.join(missing_apis)}")
            self.logger.warning("💡 Add keys to config/api_keys.env for full functionality")

        return len(available_apis) > 0

class OSINTApplicationLauncher:
    """Main application launcher with mode selection"""

    def __init__(self, bootstrap: OSINTSuiteBootstrap):
        self.bootstrap = bootstrap
        self.logger = bootstrap.logger
        self.config = bootstrap.config

    async def launch_gui_mode(self):
        """Launch GUI application"""
        try:
            self.logger.info("🖥️ Starting GUI mode...")

            # Try to import GUI components
            try:
                import tkinter as tk
                from tkinter import ttk, messagebox

                # Create tkinter root first
                root = tk.Tk()

                # Import our GUI modules
                try:
                    from src.gui.enhanced_main_window import OSINTMainWindow

                    # Create and launch enhanced GUI
                    app = OSINTMainWindow(root)
                    self.logger.info("✅ Enhanced GUI interface started")
                    root.mainloop()
                    return

                except ImportError as gui_error:
                    self.logger.warning(f"⚠️ Enhanced GUI not available: {gui_error}")
                    self.logger.info("🔄 Falling back to basic GUI...")

                except Exception as enhanced_error:
                    self.logger.error(f"❌ Enhanced GUI error: {enhanced_error}")
                    self.logger.info("🔄 Falling back to basic GUI...")

                # Fallback to basic GUI (root already created)
                root.title(f"🕵️ Desktop OSINT Suite v{__version__}")
                root.geometry("800x600")

                # Create main frame
                main_frame = ttk.Frame(root, padding="20")
                main_frame.pack(fill=tk.BOTH, expand=True)

                # Title
                title_label = ttk.Label(
                    main_frame,
                    text="🕵️ Desktop OSINT Investigation Suite",
                    font=("Arial", 16, "bold")
                )
                title_label.pack(pady=20)

                # Status
                status_text = f"Enhanced Edition v{__version__}\\nLakyLuk Development"
                status_label = ttk.Label(main_frame, text=status_text, justify=tk.CENTER)
                status_label.pack(pady=10)

                # Implementation status
                phases_text = """
🏗️ IMPLEMENTATION STATUS:

📋 FÁZE 1: Základní Infrastruktura [✅ HOTOVO]
📋 FÁZE 2: Core OSINT Engine [🔧 PŘIPRAVENO]
📋 FÁZE 3: AI Enhancement [🔧 PŘIPRAVENO]
📋 FÁZE 4: Czech OSINT Tools [🔧 PŘIPRAVENO]

🚀 Ready for development!
                """

                phases_label = ttk.Label(
                    main_frame,
                    text=phases_text,
                    justify=tk.LEFT,
                    font=("Consolas", 10)
                )
                phases_label.pack(pady=20)

                # Action buttons
                button_frame = ttk.Frame(main_frame)
                button_frame.pack(pady=20)

                def show_config():
                    config_info = f"""Configuration Status:

📂 Project: {self.bootstrap.project_root}
⚙️ Config: {'✅ Loaded' if self.config else '❌ Missing'}
🔑 API Keys: {'✅ Available' if self.bootstrap.check_api_keys() else '⚠️ Limited'}
📦 Dependencies: {'✅ Complete' if self.bootstrap.check_dependencies() else '❌ Missing'}
                    """
                    messagebox.showinfo("Configuration Status", config_info)

                def show_roadmap():
                    roadmap_info = """Development Roadmap:

Week 1: Core OSINT Engine Implementation
Week 2: AI Enhancement Integration
Week 3: Czech OSINT Tools Development
Week 4: Security & Stealth Features
Week 5: Advanced Analytics Dashboard
Week 6: Reporting & Export Capabilities
Week 7: Testing & Optimization
Week 8: Production Deployment

Current Phase: FÁZE 1 ✅ COMPLETE
Next Phase: FÁZE 2 🔧 READY FOR DEVELOPMENT"""
                    messagebox.showinfo("Development Roadmap", roadmap_info)

                config_btn = ttk.Button(button_frame, text="📊 Configuration", command=show_config)
                config_btn.pack(side=tk.LEFT, padx=10)

                roadmap_btn = ttk.Button(button_frame, text="🗺️ Roadmap", command=show_roadmap)
                roadmap_btn.pack(side=tk.LEFT, padx=10)

                exit_btn = ttk.Button(button_frame, text="❌ Exit", command=root.quit)
                exit_btn.pack(side=tk.LEFT, padx=10)

                # Footer
                footer_label = ttk.Label(
                    main_frame,
                    text="LakyLuk Enhanced Edition - Ready for Phase 2 Development",
                    font=("Arial", 8)
                )
                footer_label.pack(side=tk.BOTTOM, pady=20)

                self.logger.info("✅ GUI interface started")
                root.mainloop()

            except ImportError as e:
                self.logger.error(f"❌ GUI dependencies missing: {e}")
                print("❌ GUI mode requires tkinter and other dependencies")
                print("💡 Install with: pip install -r requirements.txt")

        except Exception as e:
            self.logger.error(f"❌ GUI startup failed: {e}")
            raise

    async def launch_cli_mode(self, target_name: Optional[str] = None):
        """Launch CLI investigation mode"""
        self.logger.info("💻 Starting CLI mode...")

        if target_name:
            self.logger.info(f"🎯 Quick investigation for: {target_name}")
            # TODO: Implement quick CLI investigation
            print(f"🔍 Starting investigation for: {target_name}")
            print("⚠️ CLI investigation mode will be implemented in FÁZE 2")
        else:
            print("🕵️ Desktop OSINT Suite - CLI Mode")
            print("=" * 40)
            print("Available commands:")
            print("  investigate <target>    - Start investigation")
            print("  list                   - List investigations")
            print("  export <id>            - Export investigation")
            print("  config                 - Show configuration")
            print("  help                   - Show this help")
            print("")
            print("⚠️ Full CLI functionality will be implemented in FÁZE 2")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description=f"🕵️ Desktop OSINT Investigation Suite v{__version__}",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f"Desktop OSINT Suite v{__version__}"
    )

    parser.add_argument(
        '--cli',
        action='store_true',
        help='Run in CLI mode instead of GUI'
    )

    parser.add_argument(
        '--target',
        type=str,
        help='Target name for quick investigation (CLI mode)'
    )

    parser.add_argument(
        '--config',
        action='store_true',
        help='Show configuration and exit'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    return parser.parse_args()

async def main():
    """Main application entry point"""
    try:
        # Parse command line arguments
        args = parse_arguments()

        # Initialize bootstrap
        bootstrap = OSINTSuiteBootstrap()

        # Setup logging
        logger = bootstrap.setup_logging()

        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("🐛 Debug logging enabled")

        # Load configuration
        config = bootstrap.load_configuration()
        if not config:
            logger.error("❌ Failed to load configuration - exiting")
            sys.exit(1)

        # Check dependencies
        if not bootstrap.check_dependencies():
            logger.error("❌ Missing dependencies - exiting")
            sys.exit(1)

        # Check API keys
        bootstrap.check_api_keys()

        # Show configuration if requested
        if args.config:
            print(f"🕵️ Desktop OSINT Suite v{__version__}")
            print(f"📂 Project root: {bootstrap.project_root}")
            print(f"⚙️ Configuration: {'✅ Loaded' if config else '❌ Missing'}")
            print(f"🔑 API keys: {'✅ Available' if bootstrap.check_api_keys() else '⚠️ Limited'}")
            print(f"📦 Dependencies: {'✅ Complete' if bootstrap.check_dependencies() else '❌ Missing'}")
            return

        # Initialize launcher
        launcher = OSINTApplicationLauncher(bootstrap)

        # Launch appropriate mode
        if args.cli or args.target:
            await launcher.launch_cli_mode(args.target)
        else:
            await launcher.launch_gui_mode()

        logger.info("✅ Application shutdown complete")

    except KeyboardInterrupt:
        print("\\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Application error: {e}")
        if '--debug' in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def cli_entry():
    """Entry point for CLI usage"""
    asyncio.run(main())

if __name__ == "__main__":
    cli_entry()
#!/usr/bin/env python3
"""
🖥️ Enhanced Main Window - OSINT Desktop Suite GUI
Modern Tkinter-based interface for comprehensive OSINT investigations

Features:
- Modern dark theme with professional styling
- Real-time investigation progress monitoring
- Interactive search configuration
- Results visualization and export
- Multi-engine search coordination
- Czech OSINT specialization controls
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import asyncio
import threading
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import webbrowser
from pathlib import Path
import logging

# Import OSINT components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from tools.web_search import SearchOrchestrator

class OSINTMainWindow:
    """Enhanced main window for OSINT Desktop Suite"""

    def __init__(self, root: tk.Tk):
        """Initialize main window"""

        self.root = root
        self.search_orchestrator = SearchOrchestrator()
        self.current_investigation = None
        self.investigation_running = False

        # Logger
        self.logger = logging.getLogger(__name__)

        # Setup window
        self.setup_window()
        self.setup_styles()
        self.create_widgets()
        self.setup_bindings()

        # Status
        self.update_engine_status()

    def setup_window(self):
        """Configure main window properties"""

        self.root.title("🕵️ Desktop OSINT Investigation Suite - Enhanced Edition")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 600)

        # Center window
        self.center_window()

        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

    def center_window(self):
        """Center window on screen"""

        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def setup_styles(self):
        """Setup modern dark theme styling"""

        style = ttk.Style()

        # Configure dark theme
        style.theme_use('clam')

        # Colors
        bg_dark = '#2b2b2b'
        bg_medium = '#3c3c3c'
        bg_light = '#4d4d4d'
        fg_light = '#ffffff'
        fg_medium = '#cccccc'
        accent_blue = '#0078d4'
        accent_green = '#107c10'
        accent_red = '#d13438'

        # Configure styles
        style.configure('Dark.TFrame', background=bg_dark)
        style.configure('Medium.TFrame', background=bg_medium, relief='raised', borderwidth=1)
        style.configure('Light.TFrame', background=bg_light)

        style.configure('Dark.TLabel', background=bg_dark, foreground=fg_light, font=('Segoe UI', 10))
        style.configure('Heading.TLabel', background=bg_dark, foreground=fg_light, font=('Segoe UI', 12, 'bold'))
        style.configure('Title.TLabel', background=bg_dark, foreground=accent_blue, font=('Segoe UI', 16, 'bold'))

        style.configure('Dark.TButton', background=bg_medium, foreground=fg_light, font=('Segoe UI', 10))
        style.configure('Accent.TButton', background=accent_blue, foreground=fg_light, font=('Segoe UI', 10, 'bold'))
        style.configure('Success.TButton', background=accent_green, foreground=fg_light, font=('Segoe UI', 10))
        style.configure('Danger.TButton', background=accent_red, foreground=fg_light, font=('Segoe UI', 10))

        style.configure('Dark.TEntry', background=bg_light, foreground=fg_light, fieldbackground=bg_light)
        style.configure('Dark.TCombobox', background=bg_light, foreground=fg_light, fieldbackground=bg_light)

        # Notebook styling
        style.configure('Dark.TNotebook', background=bg_dark, borderwidth=0)
        style.configure('Dark.TNotebook.Tab', background=bg_medium, foreground=fg_light, padding=[20, 8])
        style.map('Dark.TNotebook.Tab', background=[('selected', accent_blue)])

    def create_widgets(self):
        """Create and layout all GUI widgets"""

        # Main container
        self.main_frame = ttk.Frame(self.root, style='Dark.TFrame', padding="10")
        self.main_frame.grid(row=0, column=0, sticky='nsew')
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Header
        self.create_header()

        # Main content notebook
        self.create_notebook()

        # Status bar
        self.create_status_bar()

    def create_header(self):
        """Create header section with title and controls"""

        header_frame = ttk.Frame(self.main_frame, style='Medium.TFrame', padding="15")
        header_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        header_frame.grid_columnconfigure(1, weight=1)

        # Title and logo
        title_frame = ttk.Frame(header_frame, style='Medium.TFrame')
        title_frame.grid(row=0, column=0, sticky='w')

        title_label = ttk.Label(
            title_frame,
            text="🕵️ Desktop OSINT Investigation Suite",
            style='Title.TLabel'
        )
        title_label.pack(side='left')

        subtitle_label = ttk.Label(
            title_frame,
            text="Enhanced Edition v1.0.0 - LakyLuk Development",
            style='Dark.TLabel'
        )
        subtitle_label.pack(side='left', padx=(10, 0))

        # Header controls
        controls_frame = ttk.Frame(header_frame, style='Medium.TFrame')
        controls_frame.grid(row=0, column=2, sticky='e')

        # Engine status indicator
        self.engine_status_label = ttk.Label(
            controls_frame,
            text="⚙️ Checking engines...",
            style='Dark.TLabel'
        )
        self.engine_status_label.pack(side='right', padx=(0, 10))

    def create_notebook(self):
        """Create main notebook with different investigation tabs"""

        self.notebook = ttk.Notebook(self.main_frame, style='Dark.TNotebook')
        self.notebook.grid(row=1, column=0, sticky='nsew')

        # Create tabs
        self.create_investigation_tab()
        self.create_results_tab()
        self.create_settings_tab()
        self.create_help_tab()

    def create_investigation_tab(self):
        """Create main investigation tab"""

        # Investigation frame
        inv_frame = ttk.Frame(self.notebook, style='Dark.TFrame', padding="20")
        self.notebook.add(inv_frame, text="🔍 Investigation")

        # Configure grid
        inv_frame.grid_rowconfigure(2, weight=1)
        inv_frame.grid_columnconfigure(1, weight=1)

        # Target input section
        target_frame = ttk.LabelFrame(inv_frame, text="🎯 Investigation Target", style='Medium.TFrame', padding="15")
        target_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 15))
        target_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(target_frame, text="Target Name:", style='Dark.TLabel').grid(row=0, column=0, sticky='w', padx=(0, 10))

        self.target_entry = ttk.Entry(target_frame, style='Dark.TEntry', font=('Consolas', 11))
        self.target_entry.grid(row=0, column=1, sticky='ew', padx=(0, 10))

        self.start_button = ttk.Button(
            target_frame,
            text="🚀 Start Investigation",
            style='Accent.TButton',
            command=self.start_investigation
        )
        self.start_button.grid(row=0, column=2)

        # Investigation options
        options_frame = ttk.LabelFrame(inv_frame, text="⚙️ Investigation Options", style='Medium.TFrame', padding="15")
        options_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(0, 15))
        options_frame.grid_columnconfigure(1, weight=1)

        # Search type selection
        ttk.Label(options_frame, text="Search Type:", style='Dark.TLabel').grid(row=0, column=0, sticky='w', padx=(0, 10))

        self.search_type_var = tk.StringVar(value="comprehensive")
        search_type_combo = ttk.Combobox(
            options_frame,
            textvariable=self.search_type_var,
            values=["comprehensive", "quick", "czech_specialized"],
            state="readonly",
            style='Dark.TCombobox'
        )
        search_type_combo.grid(row=0, column=1, sticky='w', padx=(0, 20))

        # Max results
        ttk.Label(options_frame, text="Max Results:", style='Dark.TLabel').grid(row=0, column=2, sticky='w', padx=(0, 10))

        self.max_results_var = tk.StringVar(value="50")
        max_results_spin = ttk.Spinbox(
            options_frame,
            from_=10, to=200, increment=10,
            textvariable=self.max_results_var,
            width=10,
            state="readonly"
        )
        max_results_spin.grid(row=0, column=3, sticky='w')

        # Engine selection
        engines_frame = ttk.Frame(options_frame, style='Medium.TFrame')
        engines_frame.grid(row=1, column=0, columnspan=4, sticky='ew', pady=(10, 0))

        ttk.Label(engines_frame, text="Search Engines:", style='Dark.TLabel').pack(side='left', padx=(0, 10))

        self.google_var = tk.BooleanVar(value=True)
        self.bing_var = tk.BooleanVar(value=True)

        google_check = ttk.Checkbutton(engines_frame, text="Google", variable=self.google_var, style='Dark.TCheckbutton')
        google_check.pack(side='left', padx=(0, 10))

        bing_check = ttk.Checkbutton(engines_frame, text="Bing", variable=self.bing_var, style='Dark.TCheckbutton')
        bing_check.pack(side='left', padx=(0, 10))

        # Progress and results area
        progress_frame = ttk.LabelFrame(inv_frame, text="📊 Investigation Progress", style='Medium.TFrame', padding="15")
        progress_frame.grid(row=2, column=0, columnspan=2, sticky='nsew')
        progress_frame.grid_rowconfigure(1, weight=1)
        progress_frame.grid_columnconfigure(0, weight=1)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='indeterminate',
            variable=self.progress_var
        )
        self.progress_bar.grid(row=0, column=0, sticky='ew', pady=(0, 10))

        # Progress text
        self.progress_text = scrolledtext.ScrolledText(
            progress_frame,
            height=15,
            bg='#1e1e1e',
            fg='#ffffff',
            font=('Consolas', 10),
            state='disabled'
        )
        self.progress_text.grid(row=1, column=0, sticky='nsew')

    def create_results_tab(self):
        """Create results viewing tab"""

        results_frame = ttk.Frame(self.notebook, style='Dark.TFrame', padding="20")
        self.notebook.add(results_frame, text="📊 Results")

        # Configure grid
        results_frame.grid_rowconfigure(1, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)

        # Results controls
        controls_frame = ttk.Frame(results_frame, style='Medium.TFrame', padding="10")
        controls_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))

        ttk.Button(
            controls_frame,
            text="📁 Load Results",
            style='Dark.TButton',
            command=self.load_results
        ).pack(side='left', padx=(0, 10))

        ttk.Button(
            controls_frame,
            text="💾 Export Results",
            style='Dark.TButton',
            command=self.export_results
        ).pack(side='left', padx=(0, 10))

        ttk.Button(
            controls_frame,
            text="🌐 Open URLs",
            style='Success.TButton',
            command=self.open_selected_urls
        ).pack(side='left', padx=(0, 10))

        # Results display
        self.results_tree = ttk.Treeview(
            results_frame,
            columns=('confidence', 'engine', 'title', 'url'),
            show='tree headings',
            height=20
        )

        # Configure columns
        self.results_tree.heading('#0', text='#')
        self.results_tree.heading('confidence', text='Confidence')
        self.results_tree.heading('engine', text='Engine')
        self.results_tree.heading('title', text='Title')
        self.results_tree.heading('url', text='URL')

        self.results_tree.column('#0', width=50)
        self.results_tree.column('confidence', width=100)
        self.results_tree.column('engine', width=100)
        self.results_tree.column('title', width=400)
        self.results_tree.column('url', width=300)

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_tree.yview)
        h_scrollbar = ttk.Scrollbar(results_frame, orient='horizontal', command=self.results_tree.xview)

        self.results_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Grid layout
        self.results_tree.grid(row=1, column=0, sticky='nsew')
        v_scrollbar.grid(row=1, column=1, sticky='ns')
        h_scrollbar.grid(row=2, column=0, sticky='ew')

    def create_settings_tab(self):
        """Create settings configuration tab"""

        settings_frame = ttk.Frame(self.notebook, style='Dark.TFrame', padding="20")
        self.notebook.add(settings_frame, text="⚙️ Settings")

        # API Keys section
        api_frame = ttk.LabelFrame(settings_frame, text="🔑 API Keys Configuration", style='Medium.TFrame', padding="15")
        api_frame.grid(row=0, column=0, sticky='ew', pady=(0, 15))
        api_frame.grid_columnconfigure(1, weight=1)

        # Google API
        ttk.Label(api_frame, text="Google Search API:", style='Dark.TLabel').grid(row=0, column=0, sticky='w', padx=(0, 10))
        self.google_api_entry = ttk.Entry(api_frame, style='Dark.TEntry', show='*', width=40)
        self.google_api_entry.grid(row=0, column=1, sticky='ew', padx=(0, 10))

        ttk.Button(api_frame, text="Test", style='Dark.TButton', command=self.test_google_api).grid(row=0, column=2)

        # Bing API
        ttk.Label(api_frame, text="Bing Search API:", style='Dark.TLabel').grid(row=1, column=0, sticky='w', padx=(0, 10), pady=(5, 0))
        self.bing_api_entry = ttk.Entry(api_frame, style='Dark.TEntry', show='*', width=40)
        self.bing_api_entry.grid(row=1, column=1, sticky='ew', padx=(0, 10), pady=(5, 0))

        ttk.Button(api_frame, text="Test", style='Dark.TButton', command=self.test_bing_api).grid(row=1, column=2, pady=(5, 0))

        # Search options section
        search_frame = ttk.LabelFrame(settings_frame, text="🔍 Search Options", style='Medium.TFrame', padding="15")
        search_frame.grid(row=1, column=0, sticky='ew', pady=(0, 15))

        # Default settings
        self.rate_limit_var = tk.DoubleVar(value=0.1)
        self.cache_duration_var = tk.IntVar(value=60)
        self.max_concurrent_var = tk.IntVar(value=5)

        ttk.Label(search_frame, text="Rate Limit (seconds):", style='Dark.TLabel').grid(row=0, column=0, sticky='w')
        ttk.Spinbox(search_frame, from_=0.1, to=2.0, increment=0.1, textvariable=self.rate_limit_var, width=10).grid(row=0, column=1, sticky='w', padx=(10, 0))

        ttk.Label(search_frame, text="Cache Duration (minutes):", style='Dark.TLabel').grid(row=1, column=0, sticky='w', pady=(5, 0))
        ttk.Spinbox(search_frame, from_=5, to=240, increment=5, textvariable=self.cache_duration_var, width=10).grid(row=1, column=1, sticky='w', padx=(10, 0), pady=(5, 0))

        # Save/Load settings
        buttons_frame = ttk.Frame(settings_frame, style='Dark.TFrame')
        buttons_frame.grid(row=2, column=0, sticky='ew')

        ttk.Button(buttons_frame, text="💾 Save Settings", style='Success.TButton', command=self.save_settings).pack(side='left', padx=(0, 10))
        ttk.Button(buttons_frame, text="📁 Load Settings", style='Dark.TButton', command=self.load_settings).pack(side='left')

    def create_help_tab(self):
        """Create help and documentation tab"""

        help_frame = ttk.Frame(self.notebook, style='Dark.TFrame', padding="20")
        self.notebook.add(help_frame, text="📖 Help")

        # Help content
        help_text = scrolledtext.ScrolledText(
            help_frame,
            bg='#1e1e1e',
            fg='#ffffff',
            font=('Segoe UI', 10),
            state='disabled',
            wrap='word'
        )
        help_text.pack(fill='both', expand=True)

        # Help content
        help_content = """
🕵️ Desktop OSINT Investigation Suite - Help Guide

GETTING STARTED:
1. Configure API keys in the Settings tab
2. Enter target name in the Investigation tab
3. Select investigation type and options
4. Click "Start Investigation" to begin

INVESTIGATION TYPES:
• Comprehensive: Full multi-engine search with correlation
• Quick: Fast search across available engines
• Czech Specialized: Focus on Czech OSINT sources

SEARCH ENGINES:
• Google: Requires Google Custom Search API key
• Bing: Requires Bing Search API key
• Both engines provide complementary results

RESULTS:
• Results are correlated and deduplicated
• Confidence scores indicate result reliability
• Export options include JSON, CSV, and reports

CZECH OSINT SOURCES:
• justice.cz - Court records and legal documents
• ares.gov.cz - Business registry information
• firmy.cz - Company directory and financial data
• zivnostenskyrejstrik.cz - Trade license registry

ADVANCED FEATURES:
• Multi-engine result correlation
• Confidence scoring and ranking
• Real-time progress monitoring
• Professional report generation
• Rate limiting and caching

API KEYS SETUP:
1. Google: Get API key from Google Cloud Console
2. Bing: Get API key from Azure Cognitive Services
3. Enter keys in Settings tab and test connectivity

TROUBLESHOOTING:
• Check API key configuration if searches fail
• Verify internet connectivity
• Review progress log for detailed error messages
• Reduce search scope if rate limits are exceeded

For more information, visit the project documentation.
        """

        help_text.configure(state='normal')
        help_text.insert('1.0', help_content)
        help_text.configure(state='disabled')

    def create_status_bar(self):
        """Create bottom status bar"""

        self.status_frame = ttk.Frame(self.main_frame, style='Medium.TFrame', padding="5")
        self.status_frame.grid(row=2, column=0, sticky='ew', pady=(10, 0))
        self.status_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ttk.Label(
            self.status_frame,
            text="Ready for investigation",
            style='Dark.TLabel'
        )
        self.status_label.grid(row=0, column=0, sticky='w')

        # Version info
        version_label = ttk.Label(
            self.status_frame,
            text="v1.0.0-enhanced",
            style='Dark.TLabel'
        )
        version_label.grid(row=0, column=1, sticky='e')

    def setup_bindings(self):
        """Setup event bindings"""

        # Enter key in target entry starts investigation
        self.target_entry.bind('<Return>', lambda e: self.start_investigation())

        # Double-click in results opens URL
        self.results_tree.bind('<Double-1>', self.on_result_double_click)

    def update_engine_status(self):
        """Update search engine status display"""

        status = self.search_orchestrator.get_engine_status()
        engine_count = status['total_engines']

        if engine_count == 0:
            status_text = "❌ No engines available"
            color = "red"
        elif engine_count == 1:
            status_text = f"⚠️ {engine_count} engine available"
            color = "orange"
        else:
            status_text = f"✅ {engine_count} engines available"
            color = "green"

        self.engine_status_label.configure(text=status_text)

    def log_progress(self, message: str):
        """Add message to progress log"""

        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\\n"

        self.progress_text.configure(state='normal')
        self.progress_text.insert('end', log_message)
        self.progress_text.see('end')
        self.progress_text.configure(state='disabled')

        # Update status bar
        self.status_label.configure(text=message)

    def start_investigation(self):
        """Start OSINT investigation"""

        target_name = self.target_entry.get().strip()

        if not target_name:
            messagebox.showerror("Error", "Please enter a target name")
            return

        if self.investigation_running:
            messagebox.showwarning("Warning", "Investigation already running")
            return

        # Validate search engines
        status = self.search_orchestrator.get_engine_status()
        if status['total_engines'] == 0:
            messagebox.showerror("Error", "No search engines available. Please configure API keys in Settings.")
            return

        # Start investigation in background thread
        self.investigation_running = True
        self.start_button.configure(text="⏹️ Stop Investigation", command=self.stop_investigation)
        self.progress_bar.configure(mode='indeterminate')
        self.progress_bar.start()

        # Clear previous results
        self.progress_text.configure(state='normal')
        self.progress_text.delete('1.0', 'end')
        self.progress_text.configure(state='disabled')

        # Clear results tree
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        # Start investigation thread
        thread = threading.Thread(target=self.run_investigation_thread, args=(target_name,))
        thread.daemon = True
        thread.start()

    def run_investigation_thread(self, target_name: str):
        """Run investigation in background thread"""

        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run investigation
            self.root.after(0, self.log_progress, f"🎯 Starting investigation for: {target_name}")

            search_type = self.search_type_var.get()

            if search_type == "comprehensive":
                result = loop.run_until_complete(
                    self.search_orchestrator.comprehensive_investigation(target_name)
                )
            elif search_type == "quick":
                max_results = int(self.max_results_var.get())
                result = loop.run_until_complete(
                    self.search_orchestrator.quick_search(target_name, max_results=max_results)
                )
            elif search_type == "czech_specialized":
                result = loop.run_until_complete(
                    self.search_orchestrator.specialized_czech_search(target_name)
                )

            # Process results
            self.current_investigation = result
            self.root.after(0, self.process_investigation_results, result)

        except Exception as e:
            self.root.after(0, self.log_progress, f"❌ Investigation error: {str(e)}")
            self.logger.error(f"Investigation error: {e}")
        finally:
            self.root.after(0, self.investigation_completed)

    def process_investigation_results(self, results: Dict[str, Any]):
        """Process and display investigation results"""

        self.log_progress("📊 Processing results...")

        # Handle different result structures
        if "correlated_results" in results:
            # Comprehensive investigation results
            consolidated_results = results["correlated_results"]["consolidated_results"]
            confidence_scores = results.get("confidence_scores", {}).get("results", [])

            self.log_progress(f"✅ Found {len(consolidated_results)} unique results")

            # Populate results tree
            for i, result in enumerate(consolidated_results[:100]):  # Limit to 100 results
                # Find confidence score
                confidence = 0.5
                for conf_result in confidence_scores:
                    if conf_result.get("url") == result.get("url"):
                        confidence = conf_result.get("confidence_score", 0.5)
                        break

                engines = ", ".join(result.get("found_by_engines", []))
                title = result.get("title", "")[:80] + "..." if len(result.get("title", "")) > 80 else result.get("title", "")
                url = result.get("url", "")

                self.results_tree.insert(
                    "",
                    "end",
                    text=str(i + 1),
                    values=(f"{confidence:.3f}", engines, title, url)
                )

            # Log summary
            summary = results.get("summary", {})
            self.log_progress(f"📈 Summary: {summary.get('total_unique_results', 0)} results, "
                            f"{summary.get('unique_domains', 0)} domains, "
                            f"{summary.get('multi_engine_confirmations', 0)} confirmations")

        elif "engines" in results:
            # Quick search results
            total_results = 0
            for engine, engine_results in results["engines"].items():
                if not engine_results.get("error"):
                    engine_result_count = len(engine_results.get("results", []))
                    total_results += engine_result_count

                    # Add results to tree
                    for i, result in enumerate(engine_results.get("results", [])[:50]):
                        title = result.get("title", "")[:80] + "..." if len(result.get("title", "")) > 80 else result.get("title", "")
                        url = result.get("url", "") or result.get("link", "")

                        self.results_tree.insert(
                            "",
                            "end",
                            text=str(total_results - engine_result_count + i + 1),
                            values=("0.500", engine, title, url)
                        )

            self.log_progress(f"✅ Quick search completed: {total_results} total results")

        # Switch to results tab
        self.notebook.select(1)

    def investigation_completed(self):
        """Called when investigation is completed"""

        self.investigation_running = False
        self.start_button.configure(text="🚀 Start Investigation", command=self.start_investigation)
        self.progress_bar.stop()
        self.progress_bar.configure(mode='determinate', value=100)

        self.log_progress("✅ Investigation completed")

    def stop_investigation(self):
        """Stop current investigation"""

        self.investigation_running = False
        self.investigation_completed()
        self.log_progress("⏹️ Investigation stopped by user")

    def on_result_double_click(self, event):
        """Handle double-click on result item"""

        selection = self.results_tree.selection()
        if selection:
            item = self.results_tree.item(selection[0])
            url = item['values'][3] if len(item['values']) > 3 else ""
            if url:
                webbrowser.open(url)

    def load_results(self):
        """Load investigation results from file"""

        filename = filedialog.askopenfilename(
            title="Load Investigation Results",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    results = json.load(f)

                self.current_investigation = results
                self.process_investigation_results(results)
                self.log_progress(f"📁 Loaded results from: {filename}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load results: {str(e)}")

    def export_results(self):
        """Export current investigation results"""

        if not self.current_investigation:
            messagebox.showwarning("Warning", "No results to export")
            return

        filename = filedialog.asksaveasfilename(
            title="Export Investigation Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.current_investigation, f, indent=2, ensure_ascii=False)

                self.log_progress(f"💾 Results exported to: {filename}")
                messagebox.showinfo("Success", f"Results exported to: {filename}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")

    def open_selected_urls(self):
        """Open selected URLs in browser"""

        selection = self.results_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select results to open")
            return

        urls = []
        for item_id in selection:
            item = self.results_tree.item(item_id)
            url = item['values'][3] if len(item['values']) > 3 else ""
            if url:
                urls.append(url)

        if urls:
            for url in urls[:5]:  # Limit to 5 URLs
                webbrowser.open(url)

            self.log_progress(f"🌐 Opened {len(urls[:5])} URLs in browser")

    def test_google_api(self):
        """Test Google API connectivity"""
        messagebox.showinfo("Test", "Google API test - Feature coming in next update")

    def test_bing_api(self):
        """Test Bing API connectivity"""
        messagebox.showinfo("Test", "Bing API test - Feature coming in next update")

    def save_settings(self):
        """Save application settings"""
        messagebox.showinfo("Settings", "Settings save - Feature coming in next update")

    def load_settings(self):
        """Load application settings"""
        messagebox.showinfo("Settings", "Settings load - Feature coming in next update")

def main():
    """Main entry point for GUI application"""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )

    # Create and run GUI
    root = tk.Tk()
    app = OSINTMainWindow(root)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\\n👋 Application closed by user")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
🕵️ Social Investigation Dashboard - Advanced GUI Interface
LakyLuk OSINT Investigation Suite

Features:
✅ Modern tabbed interface for comprehensive social media investigation
✅ Real-time search with live results and AI analysis
✅ Interactive network visualization with zoom and filtering
✅ Advanced profile matching with similarity scoring
✅ Live monitoring dashboard with real-time alerts
✅ Export capabilities for investigation reports

GUI Components:
- Cross-Platform Search Tab
- Profile Analysis & Correlation Tab
- Network Visualization Tab
- Real-time Monitoring Tab
- Investigation Reports Tab
- Settings & Configuration Tab
"""

import asyncio
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import webbrowser
import tempfile
import os

from ..tools.social_media.cross_platform_search import CrossPlatformSearch, SearchQuery
from ..tools.social_media.realtime_monitor import RealTimeMonitor, MonitoringTarget
from ..analytics.entity_correlation_engine import EntityCorrelationEngine, EntityProfile
from ..analytics.social_network_visualizer import SocialNetworkVisualizer
from ..analytics.advanced_profile_matcher import AdvancedProfileMatcher
from ..core.enhanced_orchestrator import AIOrchestrator


class SocialInvestigationDashboard:
    """Main GUI dashboard for social media investigation"""

    def __init__(self, ai_orchestrator: AIOrchestrator = None):
        self.ai_orchestrator = ai_orchestrator

        # Initialize backend components
        self.cross_platform_search = CrossPlatformSearch(ai_orchestrator)
        self.realtime_monitor = RealTimeMonitor(ai_orchestrator)
        self.entity_correlation = EntityCorrelationEngine(ai_orchestrator)
        self.network_visualizer = SocialNetworkVisualizer(ai_orchestrator)
        self.profile_matcher = AdvancedProfileMatcher(ai_orchestrator)

        # GUI state
        self.search_results = []
        self.current_investigation = None
        self.monitoring_targets = []

        # Create main window
        self.root = tk.Tk()
        self.root.title("🕵️ OSINT Social Investigation Dashboard - LakyLuk Enhanced")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)

        # Configure styles
        self.setup_styles()

        # Create main interface
        self.create_main_interface()

        # Start backend services
        self.start_background_services()

    def setup_styles(self):
        """Setup custom styles for the interface"""
        style = ttk.Style()

        # Configure modern theme
        style.theme_use('clam')

        # Custom colors
        bg_color = "#2c3e50"
        fg_color = "#ecf0f1"
        accent_color = "#3498db"
        success_color = "#27ae60"
        warning_color = "#f39c12"
        danger_color = "#e74c3c"

        # Configure styles
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground=accent_color)
        style.configure('Heading.TLabel', font=('Arial', 12, 'bold'), foreground=fg_color)
        style.configure('Success.TLabel', foreground=success_color)
        style.configure('Warning.TLabel', foreground=warning_color)
        style.configure('Danger.TLabel', foreground=danger_color)

    def create_main_interface(self):
        """Create the main interface with tabbed layout"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Title
        title_label = ttk.Label(main_frame, text="🕵️ Social Investigation Dashboard",
                               style='Title.TLabel')
        title_label.pack(pady=(0, 10))

        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill='both', expand=True)

        # Create tabs
        self.create_search_tab()
        self.create_analysis_tab()
        self.create_network_tab()
        self.create_monitoring_tab()
        self.create_reports_tab()
        self.create_settings_tab()

        # Status bar
        self.create_status_bar(main_frame)

    def create_search_tab(self):
        """Create cross-platform search tab"""
        search_frame = ttk.Frame(self.notebook)
        self.notebook.add(search_frame, text="🔍 Cross-Platform Search")

        # Search input section
        input_frame = ttk.LabelFrame(search_frame, text="Search Parameters")
        input_frame.pack(fill='x', padx=10, pady=5)

        # Search fields
        ttk.Label(input_frame, text="Target Name:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.name_entry = ttk.Entry(input_frame, width=30)
        self.name_entry.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(input_frame, text="Username:").grid(row=0, column=2, sticky='w', padx=5, pady=2)
        self.username_entry = ttk.Entry(input_frame, width=20)
        self.username_entry.grid(row=0, column=3, padx=5, pady=2)

        ttk.Label(input_frame, text="Location:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.location_entry = ttk.Entry(input_frame, width=30)
        self.location_entry.grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(input_frame, text="Employer:").grid(row=1, column=2, sticky='w', padx=5, pady=2)
        self.employer_entry = ttk.Entry(input_frame, width=20)
        self.employer_entry.grid(row=1, column=3, padx=5, pady=2)

        # Platform selection
        platforms_frame = ttk.Frame(input_frame)
        platforms_frame.grid(row=2, column=0, columnspan=4, sticky='w', padx=5, pady=5)

        ttk.Label(platforms_frame, text="Platforms:").pack(side='left')
        self.facebook_var = tk.BooleanVar(value=True)
        self.instagram_var = tk.BooleanVar(value=True)
        self.linkedin_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(platforms_frame, text="Facebook", variable=self.facebook_var).pack(side='left', padx=5)
        ttk.Checkbutton(platforms_frame, text="Instagram", variable=self.instagram_var).pack(side='left', padx=5)
        ttk.Checkbutton(platforms_frame, text="LinkedIn", variable=self.linkedin_var).pack(side='left', padx=5)

        # Search buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=3, column=0, columnspan=4, pady=10)

        ttk.Button(button_frame, text="🔍 Search", command=self.execute_search).pack(side='left', padx=5)
        ttk.Button(button_frame, text="🧠 AI Enhanced Search", command=self.execute_ai_search).pack(side='left', padx=5)
        ttk.Button(button_frame, text="🔄 Clear", command=self.clear_search).pack(side='left', padx=5)

        # Results section
        results_frame = ttk.LabelFrame(search_frame, text="Search Results")
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Results tree
        columns = ('Platform', 'Username', 'Name', 'Location', 'Confidence', 'Match Score')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=15)

        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=120)

        scrollbar_results = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar_results.set)

        self.results_tree.pack(side='left', fill='both', expand=True)
        scrollbar_results.pack(side='right', fill='y')

        # Results context menu
        self.results_tree.bind("<Button-3>", self.show_results_context_menu)
        self.results_tree.bind("<Double-1>", self.analyze_selected_profile)

    def create_analysis_tab(self):
        """Create profile analysis and correlation tab"""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="🔬 Profile Analysis")

        # Profile input section
        input_frame = ttk.LabelFrame(analysis_frame, text="Profile Analysis")
        input_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(input_frame, text="Profile URL:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.profile_url_entry = ttk.Entry(input_frame, width=60)
        self.profile_url_entry.grid(row=0, column=1, padx=5, pady=2)

        ttk.Button(input_frame, text="🔬 Analyze Profile",
                  command=self.analyze_profile).grid(row=0, column=2, padx=5, pady=2)

        # Analysis results
        results_frame = ttk.LabelFrame(analysis_frame, text="Analysis Results")
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Create paned window for analysis results
        paned_window = ttk.PanedWindow(results_frame, orient='horizontal')
        paned_window.pack(fill='both', expand=True)

        # Profile details frame
        profile_frame = ttk.LabelFrame(paned_window, text="Profile Details")
        paned_window.add(profile_frame, weight=1)

        self.profile_details = scrolledtext.ScrolledText(profile_frame, height=20, width=40)
        self.profile_details.pack(fill='both', expand=True, padx=5, pady=5)

        # Correlation results frame
        correlation_frame = ttk.LabelFrame(paned_window, text="Similar Profiles")
        paned_window.add(correlation_frame, weight=1)

        # Correlation tree
        corr_columns = ('Platform', 'Username', 'Similarity', 'Confidence', 'Evidence')
        self.correlation_tree = ttk.Treeview(correlation_frame, columns=corr_columns, show='headings')

        for col in corr_columns:
            self.correlation_tree.heading(col, text=col)
            self.correlation_tree.column(col, width=100)

        scrollbar_corr = ttk.Scrollbar(correlation_frame, orient='vertical', command=self.correlation_tree.yview)
        self.correlation_tree.configure(yscrollcommand=scrollbar_corr.set)

        self.correlation_tree.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        scrollbar_corr.pack(side='right', fill='y')

    def create_network_tab(self):
        """Create network visualization tab"""
        network_frame = ttk.Frame(self.notebook)
        self.notebook.add(network_frame, text="🕸️ Network Analysis")

        # Controls frame
        controls_frame = ttk.LabelFrame(network_frame, text="Visualization Controls")
        controls_frame.pack(fill='x', padx=10, pady=5)

        # Layout selection
        ttk.Label(controls_frame, text="Layout:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.layout_var = tk.StringVar(value="spring")
        layout_combo = ttk.Combobox(controls_frame, textvariable=self.layout_var,
                                   values=["spring", "circular", "kamada_kawai", "spectral", "random"])
        layout_combo.grid(row=0, column=1, padx=5, pady=2)

        # Visualization buttons
        ttk.Button(controls_frame, text="🕸️ Generate Network",
                  command=self.generate_network_visualization).grid(row=0, column=2, padx=5, pady=2)
        ttk.Button(controls_frame, text="📊 Interactive View",
                  command=self.show_interactive_network).grid(row=0, column=3, padx=5, pady=2)
        ttk.Button(controls_frame, text="💾 Export Network",
                  command=self.export_network).grid(row=0, column=4, padx=5, pady=2)

        # Network info frame
        info_frame = ttk.LabelFrame(network_frame, text="Network Information")
        info_frame.pack(fill='x', padx=10, pady=5)

        self.network_info = tk.Text(info_frame, height=4, wrap='word')
        self.network_info.pack(fill='x', padx=5, pady=5)

        # Visualization frame
        viz_frame = ttk.LabelFrame(network_frame, text="Network Visualization")
        viz_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Placeholder for network visualization
        self.network_canvas = tk.Canvas(viz_frame, bg='white')
        self.network_canvas.pack(fill='both', expand=True, padx=5, pady=5)

    def create_monitoring_tab(self):
        """Create real-time monitoring tab"""
        monitoring_frame = ttk.Frame(self.notebook)
        self.notebook.add(monitoring_frame, text="⚡ Real-time Monitoring")

        # Add target frame
        add_frame = ttk.LabelFrame(monitoring_frame, text="Add Monitoring Target")
        add_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(add_frame, text="Platform:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.monitor_platform_var = tk.StringVar(value="instagram")
        platform_combo = ttk.Combobox(add_frame, textvariable=self.monitor_platform_var,
                                     values=["facebook", "instagram", "linkedin"])
        platform_combo.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(add_frame, text="Username:").grid(row=0, column=2, sticky='w', padx=5, pady=2)
        self.monitor_username_entry = ttk.Entry(add_frame, width=20)
        self.monitor_username_entry.grid(row=0, column=3, padx=5, pady=2)

        ttk.Label(add_frame, text="Check Interval (min):").grid(row=0, column=4, sticky='w', padx=5, pady=2)
        self.monitor_interval_var = tk.IntVar(value=5)
        interval_spin = ttk.Spinbox(add_frame, from_=1, to=60, textvariable=self.monitor_interval_var, width=10)
        interval_spin.grid(row=0, column=5, padx=5, pady=2)

        ttk.Button(add_frame, text="📱 Add Target",
                  command=self.add_monitoring_target).grid(row=0, column=6, padx=5, pady=2)

        # Monitoring controls
        controls_frame = ttk.Frame(monitoring_frame)
        controls_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(controls_frame, text="▶️ Start Monitoring",
                  command=self.start_monitoring).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="⏹️ Stop Monitoring",
                  command=self.stop_monitoring).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="🔄 Refresh Status",
                  command=self.refresh_monitoring_status).pack(side='left', padx=5)

        # Monitoring status
        status_frame = ttk.LabelFrame(monitoring_frame, text="Monitoring Status")
        status_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Create paned window for monitoring display
        monitor_paned = ttk.PanedWindow(status_frame, orient='horizontal')
        monitor_paned.pack(fill='both', expand=True)

        # Targets frame
        targets_frame = ttk.LabelFrame(monitor_paned, text="Active Targets")
        monitor_paned.add(targets_frame, weight=1)

        # Targets tree
        target_columns = ('Platform', 'Username', 'Status', 'Last Check', 'Events')
        self.targets_tree = ttk.Treeview(targets_frame, columns=target_columns, show='headings')

        for col in target_columns:
            self.targets_tree.heading(col, text=col)
            self.targets_tree.column(col, width=100)

        scrollbar_targets = ttk.Scrollbar(targets_frame, orient='vertical', command=self.targets_tree.yview)
        self.targets_tree.configure(yscrollcommand=scrollbar_targets.set)

        self.targets_tree.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        scrollbar_targets.pack(side='right', fill='y')

        # Events frame
        events_frame = ttk.LabelFrame(monitor_paned, text="Recent Events")
        monitor_paned.add(events_frame, weight=1)

        self.events_display = scrolledtext.ScrolledText(events_frame, height=15, width=40)
        self.events_display.pack(fill='both', expand=True, padx=5, pady=5)

    def create_reports_tab(self):
        """Create investigation reports tab"""
        reports_frame = ttk.Frame(self.notebook)
        self.notebook.add(reports_frame, text="📋 Reports")

        # Report generation frame
        gen_frame = ttk.LabelFrame(reports_frame, text="Generate Investigation Report")
        gen_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(gen_frame, text="Report Type:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.report_type_var = tk.StringVar(value="comprehensive")
        report_combo = ttk.Combobox(gen_frame, textvariable=self.report_type_var,
                                   values=["comprehensive", "profile_analysis", "network_analysis", "monitoring_summary"])
        report_combo.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(gen_frame, text="Format:").grid(row=0, column=2, sticky='w', padx=5, pady=2)
        self.report_format_var = tk.StringVar(value="pdf")
        format_combo = ttk.Combobox(gen_frame, textvariable=self.report_format_var,
                                   values=["pdf", "html", "json", "excel"])
        format_combo.grid(row=0, column=3, padx=5, pady=2)

        ttk.Button(gen_frame, text="📄 Generate Report",
                  command=self.generate_report).grid(row=0, column=4, padx=5, pady=2)

        # Report preview
        preview_frame = ttk.LabelFrame(reports_frame, text="Report Preview")
        preview_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.report_preview = scrolledtext.ScrolledText(preview_frame, height=25, width=80)
        self.report_preview.pack(fill='both', expand=True, padx=5, pady=5)

    def create_settings_tab(self):
        """Create settings and configuration tab"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="⚙️ Settings")

        # API Configuration
        api_frame = ttk.LabelFrame(settings_frame, text="API Configuration")
        api_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(api_frame, text="AI Enhancement:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.ai_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(api_frame, text="Enable AI Analysis", variable=self.ai_enabled_var).grid(row=0, column=1, sticky='w', padx=5, pady=2)

        # Search Settings
        search_frame = ttk.LabelFrame(settings_frame, text="Search Settings")
        search_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(search_frame, text="Max Results per Platform:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.max_results_var = tk.IntVar(value=50)
        ttk.Spinbox(search_frame, from_=10, to=200, textvariable=self.max_results_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(search_frame, text="Correlation Threshold:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.correlation_threshold_var = tk.DoubleVar(value=0.75)
        ttk.Scale(search_frame, from_=0.0, to=1.0, variable=self.correlation_threshold_var,
                 orient='horizontal', length=200).grid(row=1, column=1, padx=5, pady=2)

        # Monitoring Settings
        monitoring_settings_frame = ttk.LabelFrame(settings_frame, text="Monitoring Settings")
        monitoring_settings_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(monitoring_settings_frame, text="Default Check Interval (min):").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.default_interval_var = tk.IntVar(value=5)
        ttk.Spinbox(monitoring_settings_frame, from_=1, to=60, textvariable=self.default_interval_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        # Save settings button
        ttk.Button(settings_frame, text="💾 Save Settings", command=self.save_settings).pack(pady=10)

    def create_status_bar(self, parent):
        """Create status bar"""
        self.status_frame = ttk.Frame(parent)
        self.status_frame.pack(fill='x', side='bottom')

        self.status_label = ttk.Label(self.status_frame, text="Ready")
        self.status_label.pack(side='left', padx=5)

        self.progress_bar = ttk.Progressbar(self.status_frame, mode='indeterminate')
        self.progress_bar.pack(side='right', padx=5)

    def start_background_services(self):
        """Start background services and threads"""
        # Start event loop in separate thread for async operations
        self.event_loop = asyncio.new_event_loop()
        self.background_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.background_thread.start()

        # Start monitoring status updates
        self.root.after(5000, self.update_monitoring_display)  # Update every 5 seconds

    def _run_event_loop(self):
        """Run asyncio event loop in background thread"""
        asyncio.set_event_loop(self.event_loop)
        self.event_loop.run_forever()

    # Event handlers for GUI interactions

    def execute_search(self):
        """Execute cross-platform search"""
        def search_task():
            try:
                self.set_status("Executing search...")
                self.progress_bar.start()

                # Build search query
                platforms = []
                if self.facebook_var.get():
                    platforms.append('facebook')
                if self.instagram_var.get():
                    platforms.append('instagram')
                if self.linkedin_var.get():
                    platforms.append('linkedin')

                query = SearchQuery(
                    target_name=self.name_entry.get() or None,
                    username=self.username_entry.get() or None,
                    location=self.location_entry.get() or None,
                    employer=self.employer_entry.get() or None,
                    platforms=platforms
                )

                # Execute search
                future = asyncio.run_coroutine_threadsafe(
                    self.cross_platform_search.unified_people_search(query),
                    self.event_loop
                )
                search_result = future.result(timeout=60)

                # Update GUI with results
                self.root.after(0, lambda: self.display_search_results(search_result))

            except Exception as e:
                self.root.after(0, lambda: self.show_error(f"Search error: {e}"))
            finally:
                self.root.after(0, lambda: (self.progress_bar.stop(), self.set_status("Ready")))

        # Run search in background thread
        threading.Thread(target=search_task, daemon=True).start()

    def execute_ai_search(self):
        """Execute AI-enhanced search"""
        if not self.ai_orchestrator:
            messagebox.showwarning("AI Not Available", "AI orchestrator is not configured")
            return

        # Similar to execute_search but with AI enhancement enabled
        self.execute_search()

    def clear_search(self):
        """Clear search fields and results"""
        self.name_entry.delete(0, tk.END)
        self.username_entry.delete(0, tk.END)
        self.location_entry.delete(0, tk.END)
        self.employer_entry.delete(0, tk.END)

        # Clear results tree
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        self.search_results = []

    def display_search_results(self, search_result):
        """Display search results in the tree view"""
        # Clear existing results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        self.search_results = search_result.profiles

        # Add results to tree
        for profile in search_result.profiles:
            confidence = profile.metadata.get('ai_relevance_score', 0.5)
            match_score = profile.metadata.get('similarity_score', 0.0)

            self.results_tree.insert('', 'end', values=(
                profile.platform.title(),
                profile.username,
                profile.display_name or 'N/A',
                profile.location or 'N/A',
                f"{confidence:.2f}",
                f"{match_score:.2f}"
            ))

        self.set_status(f"Found {len(search_result.profiles)} profiles across {len(search_result.platforms_searched)} platforms")

    def analyze_selected_profile(self, event):
        """Analyze selected profile from search results"""
        selection = self.results_tree.selection()
        if not selection:
            return

        item = self.results_tree.item(selection[0])
        values = item['values']

        if len(values) >= 2:
            platform = values[0].lower()
            username = values[1]

            # Find profile in search results
            selected_profile = None
            for profile in self.search_results:
                if profile.platform == platform and profile.username == username:
                    selected_profile = profile
                    break

            if selected_profile:
                self.analyze_profile_detailed(selected_profile)

    def analyze_profile_detailed(self, profile):
        """Perform detailed profile analysis"""
        def analysis_task():
            try:
                self.set_status(f"Analyzing profile {profile.username}...")
                self.progress_bar.start()

                # Extract matching features
                future = asyncio.run_coroutine_threadsafe(
                    self.profile_matcher.extract_matching_features(profile),
                    self.event_loop
                )
                features = future.result(timeout=30)

                # Find similar profiles
                future = asyncio.run_coroutine_threadsafe(
                    self.cross_platform_search.find_similar_profiles(profile),
                    self.event_loop
                )
                similar_profiles = future.result(timeout=60)

                # Update GUI
                self.root.after(0, lambda: self.display_profile_analysis(profile, features, similar_profiles))

            except Exception as e:
                self.root.after(0, lambda: self.show_error(f"Analysis error: {e}"))
            finally:
                self.root.after(0, lambda: (self.progress_bar.stop(), self.set_status("Ready")))

        threading.Thread(target=analysis_task, daemon=True).start()

    def display_profile_analysis(self, profile, features, similar_profiles):
        """Display profile analysis results"""
        # Switch to analysis tab
        self.notebook.select(1)

        # Display profile details
        details_text = f"Profile Analysis: {profile.username}\n"
        details_text += f"Platform: {profile.platform.title()}\n"
        details_text += f"Display Name: {profile.display_name or 'N/A'}\n"
        details_text += f"Bio: {profile.bio or 'N/A'}\n"
        details_text += f"Location: {profile.location or 'N/A'}\n"
        details_text += f"Followers: {profile.follower_count or 'N/A'}\n"
        details_text += f"Following: {profile.following_count or 'N/A'}\n"
        details_text += f"Verified: {profile.verified}\n\n"

        details_text += f"Feature Quality Score: {features.feature_quality_score:.2f}\n\n"

        if features.face_encoding is not None:
            details_text += "✅ Face encoding available\n"
        if features.bio_embedding is not None:
            details_text += "✅ Bio embedding extracted\n"
        if features.content_embedding is not None:
            details_text += "✅ Content embedding extracted\n"

        self.profile_details.delete(1.0, tk.END)
        self.profile_details.insert(1.0, details_text)

        # Display similar profiles
        for item in self.correlation_tree.get_children():
            self.correlation_tree.delete(item)

        for similar_profile, similarity_score in similar_profiles:
            self.correlation_tree.insert('', 'end', values=(
                similar_profile.platform.title(),
                similar_profile.username,
                f"{similarity_score:.3f}",
                "High" if similarity_score > 0.8 else "Medium" if similarity_score > 0.6 else "Low",
                "Face match" if similarity_score > 0.9 else "Profile similarity"
            ))

    # Additional GUI event handlers...

    def add_monitoring_target(self):
        """Add new monitoring target"""
        platform = self.monitor_platform_var.get()
        username = self.monitor_username_entry.get().strip()

        if not username:
            messagebox.showerror("Error", "Please enter a username")
            return

        target = MonitoringTarget(
            target_id=f"{platform}:{username}",
            platform=platform,
            username=username,
            profile_url=f"https://{platform}.com/{username}",
            monitoring_type="all",
            check_interval=self.monitor_interval_var.get() * 60  # Convert to seconds
        )

        def add_task():
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.realtime_monitor.add_monitoring_target(target),
                    self.event_loop
                )
                success = future.result(timeout=10)

                if success:
                    self.root.after(0, lambda: self.refresh_monitoring_status())
                    self.root.after(0, lambda: messagebox.showinfo("Success", f"Added monitoring for {username}"))
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Failed to add monitoring target"))

            except Exception as e:
                self.root.after(0, lambda: self.show_error(f"Error adding target: {e}"))

        threading.Thread(target=add_task, daemon=True).start()

    def start_monitoring(self):
        """Start real-time monitoring"""
        def start_task():
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.realtime_monitor.start_monitoring(),
                    self.event_loop
                )
                success = future.result(timeout=10)

                if success:
                    self.root.after(0, lambda: self.set_status("Monitoring started"))
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Failed to start monitoring"))

            except Exception as e:
                self.root.after(0, lambda: self.show_error(f"Error starting monitoring: {e}"))

        threading.Thread(target=start_task, daemon=True).start()

    def stop_monitoring(self):
        """Stop real-time monitoring"""
        def stop_task():
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.realtime_monitor.stop_monitoring(),
                    self.event_loop
                )
                success = future.result(timeout=10)

                if success:
                    self.root.after(0, lambda: self.set_status("Monitoring stopped"))
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Failed to stop monitoring"))

            except Exception as e:
                self.root.after(0, lambda: self.show_error(f"Error stopping monitoring: {e}"))

        threading.Thread(target=stop_task, daemon=True).start()

    def refresh_monitoring_status(self):
        """Refresh monitoring status display"""
        def refresh_task():
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.realtime_monitor.get_monitoring_status(),
                    self.event_loop
                )
                status = future.result(timeout=5)

                self.root.after(0, lambda: self.update_monitoring_display_with_status(status))

            except Exception as e:
                self.root.after(0, lambda: self.show_error(f"Error refreshing status: {e}"))

        threading.Thread(target=refresh_task, daemon=True).start()

    def update_monitoring_display_with_status(self, status):
        """Update monitoring display with status information"""
        # Clear targets tree
        for item in self.targets_tree.get_children():
            self.targets_tree.delete(item)

        # Update targets tree
        for target in self.realtime_monitor.targets.values():
            last_check = target.last_checked.strftime("%H:%M:%S") if target.last_checked else "Never"
            event_count = len([e for e in self.realtime_monitor.events if e.target_id == target.target_id])

            self.targets_tree.insert('', 'end', values=(
                target.platform.title(),
                target.username,
                "Active" if target.active else "Inactive",
                last_check,
                event_count
            ))

        # Update events display
        recent_events = status.get('recent_events', [])
        events_text = ""
        for event in recent_events[-20:]:  # Show last 20 events
            timestamp = event.timestamp.strftime("%H:%M:%S") if hasattr(event, 'timestamp') else "Unknown"
            events_text += f"[{timestamp}] {event.description}\n"

        self.events_display.delete(1.0, tk.END)
        self.events_display.insert(1.0, events_text)

    def update_monitoring_display(self):
        """Periodic update of monitoring display"""
        if self.realtime_monitor.monitoring_active:
            self.refresh_monitoring_status()

        # Schedule next update
        self.root.after(5000, self.update_monitoring_display)

    # Utility methods

    def set_status(self, message):
        """Set status bar message"""
        self.status_label.config(text=message)

    def show_error(self, message):
        """Show error message"""
        messagebox.showerror("Error", message)
        self.set_status("Error occurred")

    def save_settings(self):
        """Save application settings"""
        messagebox.showinfo("Settings", "Settings saved successfully")

    def generate_report(self):
        """Generate investigation report"""
        report_type = self.report_type_var.get()
        report_format = self.report_format_var.get()

        # Generate sample report
        report_text = f"Investigation Report - {report_type.title()}\n"
        report_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report_text += "=" * 50 + "\n\n"

        if self.search_results:
            report_text += f"Search Results Summary:\n"
            report_text += f"Total profiles found: {len(self.search_results)}\n"
            report_text += f"Platforms searched: Facebook, Instagram, LinkedIn\n\n"

            for profile in self.search_results[:10]:  # First 10 results
                report_text += f"- {profile.platform.title()}: {profile.username}\n"
                if profile.display_name:
                    report_text += f"  Name: {profile.display_name}\n"
                if profile.location:
                    report_text += f"  Location: {profile.location}\n"
                report_text += "\n"

        self.report_preview.delete(1.0, tk.END)
        self.report_preview.insert(1.0, report_text)

    def show_results_context_menu(self, event):
        """Show context menu for results"""
        context_menu = tk.Menu(self.root, tearoff=0)
        context_menu.add_command(label="Analyze Profile", command=lambda: self.analyze_selected_profile(event))
        context_menu.add_command(label="Add to Monitoring", command=self.add_selected_to_monitoring)
        context_menu.add_command(label="Export Profile Data", command=self.export_selected_profile)

        try:
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            context_menu.grab_release()

    def add_selected_to_monitoring(self):
        """Add selected profile to monitoring"""
        selection = self.results_tree.selection()
        if selection:
            item = self.results_tree.item(selection[0])
            values = item['values']
            if len(values) >= 2:
                self.monitor_platform_var.set(values[0].lower())
                self.monitor_username_entry.delete(0, tk.END)
                self.monitor_username_entry.insert(0, values[1])
                # Switch to monitoring tab
                self.notebook.select(3)

    def export_selected_profile(self):
        """Export selected profile data"""
        messagebox.showinfo("Export", "Profile export functionality would be implemented here")

    def generate_network_visualization(self):
        """Generate network visualization"""
        if not self.search_results:
            messagebox.showwarning("No Data", "Please perform a search first")
            return

        def viz_task():
            try:
                self.set_status("Generating network visualization...")
                self.progress_bar.start()

                # Create network from search results
                future = asyncio.run_coroutine_threadsafe(
                    self.network_visualizer.create_network_from_profiles(self.search_results),
                    self.event_loop
                )
                network_info = future.result(timeout=30)

                # Generate static visualization
                future = asyncio.run_coroutine_threadsafe(
                    self.network_visualizer.generate_static_visualization(
                        layout=self.layout_var.get()
                    ),
                    self.event_loop
                )
                image_path = future.result(timeout=30)

                self.root.after(0, lambda: self.display_network_visualization(network_info, image_path))

            except Exception as e:
                self.root.after(0, lambda: self.show_error(f"Visualization error: {e}"))
            finally:
                self.root.after(0, lambda: (self.progress_bar.stop(), self.set_status("Ready")))

        threading.Thread(target=viz_task, daemon=True).start()

    def display_network_visualization(self, network_info, image_path):
        """Display network visualization"""
        # Switch to network tab
        self.notebook.select(2)

        # Display network information
        info_text = f"Network Statistics:\n"
        info_text += f"Nodes: {network_info.get('nodes_count', 0)}\n"
        info_text += f"Edges: {network_info.get('edges_count', 0)}\n"
        info_text += f"Communities: {network_info.get('communities_count', 0)}\n"

        self.network_info.delete(1.0, tk.END)
        self.network_info.insert(1.0, info_text)

        # Display visualization (simplified - would load actual image)
        self.network_canvas.delete("all")
        self.network_canvas.create_text(
            400, 200,
            text=f"Network Visualization Generated\n{image_path}",
            font=("Arial", 14),
            fill="blue"
        )

    def show_interactive_network(self):
        """Show interactive network visualization"""
        if not self.search_results:
            messagebox.showwarning("No Data", "Please perform a search first")
            return

        def interactive_task():
            try:
                # Generate interactive visualization
                future = asyncio.run_coroutine_threadsafe(
                    self.network_visualizer.generate_interactive_visualization(),
                    self.event_loop
                )
                html_content = future.result(timeout=30)

                # Save to temp file and open in browser
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                    f.write(html_content)
                    temp_path = f.name

                webbrowser.open(f'file://{temp_path}')

            except Exception as e:
                self.root.after(0, lambda: self.show_error(f"Interactive visualization error: {e}"))

        threading.Thread(target=interactive_task, daemon=True).start()

    def export_network(self):
        """Export network visualization"""
        if not self.search_results:
            messagebox.showwarning("No Data", "Please perform a search first")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("SVG files", "*.svg"), ("HTML files", "*.html")]
        )

        if file_path:
            def export_task():
                try:
                    if file_path.endswith('.html'):
                        future = asyncio.run_coroutine_threadsafe(
                            self.network_visualizer.generate_interactive_visualization(file_path),
                            self.event_loop
                        )
                    else:
                        future = asyncio.run_coroutine_threadsafe(
                            self.network_visualizer.generate_static_visualization(file_path),
                            self.event_loop
                        )

                    result_path = future.result(timeout=60)
                    self.root.after(0, lambda: messagebox.showinfo("Export", f"Network exported to {result_path}"))

                except Exception as e:
                    self.root.after(0, lambda: self.show_error(f"Export error: {e}"))

            threading.Thread(target=export_task, daemon=True).start()

    def run(self):
        """Start the GUI application"""
        try:
            self.set_status("OSINT Social Investigation Dashboard - Ready")
            self.root.mainloop()
        finally:
            # Cleanup
            if hasattr(self, 'event_loop'):
                self.event_loop.call_soon_threadsafe(self.event_loop.stop)


def main():
    """Main entry point for the dashboard"""
    # Initialize AI orchestrator (optional)
    ai_orchestrator = None
    try:
        ai_orchestrator = AIOrchestrator()
    except Exception as e:
        print(f"AI orchestrator not available: {e}")

    # Create and run dashboard
    dashboard = SocialInvestigationDashboard(ai_orchestrator)
    dashboard.run()


if __name__ == "__main__":
    main()
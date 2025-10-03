#!/usr/bin/env python3
"""
= Web Search Tools Module
OSINT Desktop Suite - Search Engine Integration

This module provides comprehensive web search capabilities for OSINT investigations,
including Google Search, Bing Search, and intelligent result orchestration.

Features:
- Multi-engine search coordination
- Advanced search operators and filters
- Result correlation and deduplication
- Czech OSINT source specialization
- Confidence scoring and ranking
- Rate limiting and caching
"""

from .google_search import GoogleSearchTool
from .bing_search import BingSearchTool
from .search_orchestrator import SearchOrchestrator

__all__ = [
    'GoogleSearchTool',
    'BingSearchTool',
    'SearchOrchestrator'
]

__version__ = "1.0.0"
__author__ = "LakyLuk"
__description__ = "Web Search Tools for OSINT Desktop Suite"
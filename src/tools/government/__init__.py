#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Government OSINT Tools Module
Czech Republic Government Database Access

This module provides specialized tools for accessing Czech government databases
and public records for OSINT investigations.

Features:
- Justice.cz court records and legal documents
- ARES.gov.cz business registry and company information
- Respectful rate limiting and caching
- Legal compliance and ethical access
- Comprehensive search capabilities
"""

from .justice_cz import JusticeCzTool
from .ares_cz import AresCzTool

__all__ = [
    'JusticeCzTool',
    'AresCzTool'
]

__version__ = "1.0.0"
__author__ = "LakyLuk"
__description__ = "Czech Government OSINT Tools for Desktop Suite"

def get_legal_notice() -> str:
    """Get legal compliance notice for government tools"""
    return """
CZECH GOVERNMENT OSINT TOOLS - LEGAL COMPLIANCE

These tools access public information from Czech government databases:
- justice.cz - Court records and legal proceedings
- ares.gov.cz - Business registry and company information

IMPORTANT COMPLIANCE REQUIREMENTS:
- Public data access only
- Respectful rate limiting implemented
- GDPR compliant data handling
- Website terms of service respected
- No unauthorized automated access

CURRENT IMPLEMENTATION STATUS:
- PROTOTYPE - Provides structure and demonstration
- Production use requires detailed form analysis
- Legal review recommended before deployment

For production use, ensure proper legal compliance review.
"""

def get_available_tools() -> dict:
    """Get information about available government OSINT tools"""

    return {
        "justice_cz": {
            "class": "JusticeCzTool",
            "description": "Czech court records and legal proceedings search",
            "databases": [
                "Civil court proceedings",
                "Insolvency registry",
                "Commercial register",
                "Criminal proceedings"
            ],
            "status": "PROTOTYPE"
        },
        "ares_cz": {
            "class": "AresCzTool",
            "description": "Czech business registry and company information",
            "databases": [
                "Company basic information",
                "Business registration data",
                "Tax identification lookup",
                "Legal entity verification"
            ],
            "status": "FUNCTIONAL"
        }
    }
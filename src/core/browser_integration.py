#!/usr/bin/env python3
"""
🔧 Browser Integration Adapter - Social Media OSINT Integration
Desktop OSINT Suite - Browser Automation Integration
LakyLuk Enhanced Edition

Integration layer between social media scanners and browser manager
Features:
- Unified API for social media tools
- Method mapping and compatibility layer
- Enhanced session management for OSINT operations
- Social media specific configurations
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

from .browser_manager import EnhancedBrowserManager, BrowserType, StealthLevel, BrowserProfile, ScrapingTarget
from .proxy_manager import AntiDetectionManager

@dataclass
class SocialMediaSession:
    """Social media browser session configuration"""
    platform: str  # "facebook", "instagram", "linkedin"
    session_id: str
    browser_type: BrowserType
    stealth_level: str
    proxy: Optional[str] = None
    user_agent: Optional[str] = None
    cookies: Optional[Dict] = None
    created_at: datetime = None

class BrowserIntegrationAdapter:
    """
    Integration adapter for social media OSINT tools

    Provides unified interface between social media scanners
    and the enhanced browser manager with OSINT-specific optimizations
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.browser_manager = EnhancedBrowserManager(stealth_level=StealthLevel.MAXIMUM)
        self.anti_detection_manager = AntiDetectionManager()

        # Active sessions tracking
        self.active_sessions: Dict[str, SocialMediaSession] = {}

        # Platform-specific configurations
        self.platform_configs = {
            'facebook': {
                'base_url': 'https://www.facebook.com',
                'mobile_url': 'https://m.facebook.com',
                'stealth_level': 'maximum',
                'user_agent_type': 'desktop',
                'rate_limit': {'requests_per_minute': 10, 'requests_per_hour': 100}
            },
            'instagram': {
                'base_url': 'https://www.instagram.com',
                'stealth_level': 'maximum',
                'user_agent_type': 'mobile',
                'rate_limit': {'requests_per_minute': 8, 'requests_per_hour': 80}
            },
            'linkedin': {
                'base_url': 'https://www.linkedin.com',
                'stealth_level': 'moderate',
                'user_agent_type': 'desktop',
                'rate_limit': {'requests_per_minute': 15, 'requests_per_hour': 150}
            }
        }

    async def initialize(self):
        """Initialize the integration adapter"""
        try:
            self.logger.info("🔧 Initializing Browser Integration Adapter...")

            # Initialize browser manager
            await self.browser_manager.initialize()

            # Initialize anti-detection manager
            await self.anti_detection_manager.initialize()

            self.logger.info("✅ Browser Integration Adapter initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Browser Integration Adapter: {e}")
            return False

    async def create_session(self, platform: str = "generic",
                           stealth_level: str = "high",
                           proxy: Optional[str] = None) -> Any:
        """
        Create a new browser session for social media scanning

        Args:
            platform: Social media platform ("facebook", "instagram", "linkedin")
            stealth_level: Stealth level ("low", "medium", "high", "maximum")
            proxy: Optional proxy to use

        Returns:
            Browser session object compatible with social media scanners
        """
        try:
            self.logger.info(f"🌐 Creating browser session for {platform}")

            # Map stealth level to enum
            stealth_map = {
                "low": StealthLevel.MINIMAL,
                "medium": StealthLevel.MODERATE,
                "high": StealthLevel.MAXIMUM,
                "maximum": StealthLevel.MAXIMUM
            }

            mapped_stealth = stealth_map.get(stealth_level, StealthLevel.MODERATE)

            # Get proxy if requested
            if proxy == "random" or proxy is True:
                proxy = await self.anti_detection_manager.get_proxy()

            # Create browser profile for the session
            from .browser_manager import BrowserProfile

            profile = BrowserProfile(
                name=f"{platform}_profile",
                user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                stealth_enabled=True,
                proxy=proxy
            )

            # Create browser session with enhanced browser manager
            session_success = await self.browser_manager.create_browser_session(
                browser_type=BrowserType.PLAYWRIGHT_CHROMIUM,
                profile=profile
            )

            if session_success:
                # Create session tracking
                session_id = f"{platform}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                session = SocialMediaSession(
                    platform=platform,
                    session_id=session_id,
                    browser_type=BrowserType.PLAYWRIGHT_CHROMIUM,
                    stealth_level=stealth_level,
                    proxy=proxy,
                    created_at=datetime.now()
                )

                self.active_sessions[session_id] = session

                # Return browser session wrapper
                return BrowserSessionWrapper(
                    browser_manager=self.browser_manager,
                    session_id=session_id,
                    platform=platform
                )
            else:
                raise Exception("Failed to create browser session")

        except Exception as e:
            self.logger.error(f"❌ Failed to create session: {e}")
            return None

    async def get_platform_config(self, platform: str) -> Dict[str, Any]:
        """Get platform-specific configuration"""
        return self.platform_configs.get(platform, {})

    async def cleanup_sessions(self):
        """Clean up all active sessions"""
        self.logger.info("🧹 Cleaning up browser sessions...")

        for session_id in list(self.active_sessions.keys()):
            await self.close_session(session_id)

        await self.browser_manager.close_session()

    async def close_session(self, session_id: str):
        """Close specific session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            self.logger.info(f"🗑️ Closed session: {session_id}")

class BrowserSessionWrapper:
    """
    Wrapper class that provides compatibility interface for social media scanners
    """

    def __init__(self, browser_manager: EnhancedBrowserManager, session_id: str, platform: str):
        self.browser_manager = browser_manager
        self.session_id = session_id
        self.platform = platform
        self.logger = logging.getLogger(__name__)

    async def new_page(self):
        """Create new page - compatibility method for social media scanners"""
        try:
            # Return page wrapper that maps to browser manager methods
            return PageWrapper(self.browser_manager)

        except Exception as e:
            self.logger.error(f"❌ Failed to create new page: {e}")
            return None

    async def close(self):
        """Close browser session"""
        await self.browser_manager.close_session()

class PageWrapper:
    """
    Wrapper class that provides page-level compatibility for social media scanners
    """

    def __init__(self, browser_manager: EnhancedBrowserManager):
        self.browser_manager = browser_manager
        self.logger = logging.getLogger(__name__)

    async def goto(self, url: str, wait_until: str = "load"):
        """Navigate to URL"""
        try:
            wait_for_selector = None
            if wait_until == "networkidle":
                # For networkidle, we'll use a generic selector and timeout
                wait_for_selector = "body"

            success = await self.browser_manager.navigate_to_url(url, wait_for_selector)
            return success

        except Exception as e:
            self.logger.error(f"❌ Navigation failed: {e}")
            return False

    async def wait_for_selector(self, selector: str, timeout: int = 10000):
        """Wait for element selector"""
        try:
            # Use browser manager's data extraction as a way to wait
            await asyncio.sleep(timeout / 10000)  # Convert ms to seconds for basic wait
            return True

        except Exception as e:
            self.logger.error(f"❌ Wait for selector failed: {e}")
            return False

    async def evaluate(self, script: str):
        """Evaluate JavaScript - basic implementation"""
        try:
            # Basic implementation - would need enhancement for full JS evaluation
            self.logger.info(f"🔧 JavaScript evaluation requested: {script[:50]}...")
            return None

        except Exception as e:
            self.logger.error(f"❌ JavaScript evaluation failed: {e}")
            return None

    async def content(self):
        """Get page content"""
        try:
            # Use browser manager to extract content
            selectors = {"content": "body"}
            data = await self.browser_manager.extract_data(selectors)
            return data.get("content", "")

        except Exception as e:
            self.logger.error(f"❌ Content extraction failed: {e}")
            return ""

# Global integration adapter instance
_integration_adapter = None

async def get_integration_adapter() -> BrowserIntegrationAdapter:
    """Get global integration adapter instance"""
    global _integration_adapter

    if _integration_adapter is None:
        _integration_adapter = BrowserIntegrationAdapter()
        await _integration_adapter.initialize()

    return _integration_adapter

# Compatibility functions for social media scanners
async def create_stealth_session(platform: str = "generic", stealth_level: str = "high"):
    """Create stealth browser session - compatibility function"""
    adapter = await get_integration_adapter()
    return await adapter.create_session(platform, stealth_level)

async def cleanup_browser_sessions():
    """Cleanup all browser sessions - compatibility function"""
    global _integration_adapter
    if _integration_adapter:
        await _integration_adapter.cleanup_sessions()
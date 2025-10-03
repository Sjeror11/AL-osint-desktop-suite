#!/usr/bin/env python3
"""
🌐 Enhanced Browser Manager - Stealth Web Scraping Engine
Desktop OSINT Suite - Phase 3 Implementation
LakyLuk Enhanced Edition - 27.9.2025

Advanced browser automation with anti-detection capabilities
Features:
- Multi-browser support (Selenium + Playwright)
- Stealth browsing with fingerprint rotation
- Proxy rotation and user agent spoofing
- Session management and cookie persistence
- Anti-detection evasion techniques
- Human-like behavior simulation
"""

import asyncio
import random
import time
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os

# Browser automation imports
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    from playwright.async_api import async_playwright, Browser, Page, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

try:
    from fake_useragent import UserAgent
    FAKE_USERAGENT_AVAILABLE = True
except ImportError:
    FAKE_USERAGENT_AVAILABLE = False

try:
    from .proxy_manager import AntiDetectionManager, ProxyType
    ANTI_DETECTION_AVAILABLE = True
except ImportError:
    ANTI_DETECTION_AVAILABLE = False

from enum import Enum

class BrowserType(Enum):
    """Supported browser types"""
    SELENIUM_CHROME = "selenium_chrome"
    PLAYWRIGHT_CHROMIUM = "playwright_chromium"
    PLAYWRIGHT_FIREFOX = "playwright_firefox"

class StealthLevel(Enum):
    """Stealth operation levels"""
    MINIMAL = "minimal"      # Basic anti-detection
    MODERATE = "moderate"    # Advanced fingerprint rotation
    MAXIMUM = "maximum"      # Full stealth with human simulation

@dataclass
class BrowserProfile:
    """Browser profile configuration"""
    name: str
    user_agent: str
    viewport_width: int = 1920
    viewport_height: int = 1080
    timezone: str = "Europe/Prague"
    locale: str = "cs-CZ"
    stealth_enabled: bool = True
    proxy: Optional[str] = None
    cookies: Optional[Dict] = None

@dataclass
class ScrapingTarget:
    """Web scraping target configuration"""
    url: str
    target_type: str  # "search", "profile", "directory", "document"
    selectors: Dict[str, str]  # CSS selectors for data extraction
    wait_for: Optional[str] = None  # Selector to wait for
    scroll_required: bool = False
    pagination: bool = False
    max_pages: int = 1
    delay_min: float = 1.0
    delay_max: float = 3.0

class EnhancedBrowserManager:
    """
    Enhanced browser automation manager with stealth capabilities

    Provides:
    - Multi-browser support (Selenium Chrome + Playwright)
    - Advanced anti-detection techniques
    - Human-like behavior simulation
    - Proxy rotation and session management
    - Intelligent error handling and recovery
    """

    def __init__(self, stealth_level: StealthLevel = StealthLevel.MODERATE):
        self.logger = logging.getLogger(__name__)
        self.stealth_level = stealth_level

        # Browser instances
        self.selenium_driver: Optional[webdriver.Chrome] = None
        self.playwright_browser: Optional[Browser] = None
        self.playwright_context: Optional[BrowserContext] = None
        self.playwright_page: Optional[Page] = None

        # Configuration
        self.browser_profiles: List[BrowserProfile] = []
        self.current_profile: Optional[BrowserProfile] = None
        self.session_data: Dict[str, Any] = {}

        # User agents for rotation
        self.user_agents: List[str] = []

        # Anti-detection manager
        self.anti_detection_manager = None

        # Statistics
        self.scraping_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'blocked_requests': 0,
            'detection_events': 0,
            'last_rotation': None
        }

    async def initialize(self):
        """Initialize browser manager and load configurations"""
        try:
            self.logger.info("🌐 Initializing Enhanced Browser Manager...")

            # Load browser profiles
            await self._load_browser_profiles()

            # Initialize user agent rotation
            await self._initialize_user_agents()

            # Set up anti-detection measures
            await self._setup_anti_detection()

            # Initialize advanced anti-detection manager
            await self._initialize_anti_detection_manager()

            self.logger.info(f"✅ Browser Manager initialized with {self.stealth_level.value} stealth level")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize browser manager: {e}")
            raise

    async def _load_browser_profiles(self):
        """Load browser profiles from configuration"""

        # Default profiles for different scenarios
        default_profiles = [
            BrowserProfile(
                name="desktop_chrome",
                user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport_width=1920,
                viewport_height=1080
            ),
            BrowserProfile(
                name="mobile_chrome",
                user_agent="Mozilla/5.0 (Linux; Android 10; SM-G975F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
                viewport_width=375,
                viewport_height=667
            ),
            BrowserProfile(
                name="windows_chrome",
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport_width=1366,
                viewport_height=768
            )
        ]

        self.browser_profiles = default_profiles
        self.current_profile = random.choice(self.browser_profiles)

        self.logger.info(f"📱 Loaded {len(self.browser_profiles)} browser profiles")

    async def _initialize_user_agents(self):
        """Initialize user agent rotation system"""

        if FAKE_USERAGENT_AVAILABLE:
            try:
                ua = UserAgent()
                # Generate diverse user agents
                for _ in range(20):
                    self.user_agents.append(ua.random)

                self.logger.info(f"🎭 Generated {len(self.user_agents)} user agents for rotation")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to generate user agents: {e}")

        # Fallback user agents
        if not self.user_agents:
            self.user_agents = [
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ]

    async def _setup_anti_detection(self):
        """Setup anti-detection measures based on stealth level"""

        if self.stealth_level == StealthLevel.MINIMAL:
            self.logger.info("🔒 Minimal stealth: Basic user agent rotation")

        elif self.stealth_level == StealthLevel.MODERATE:
            self.logger.info("🔒 Moderate stealth: Advanced fingerprint rotation")

        elif self.stealth_level == StealthLevel.MAXIMUM:
            self.logger.info("🔒 Maximum stealth: Full human behavior simulation")

    async def _initialize_anti_detection_manager(self):
        """Initialize advanced anti-detection manager"""
        try:
            if ANTI_DETECTION_AVAILABLE and self.stealth_level != StealthLevel.MINIMAL:
                self.anti_detection_manager = AntiDetectionManager()
                await self.anti_detection_manager.initialize()
                self.logger.info("🛡️ Advanced anti-detection manager initialized")
            else:
                self.logger.warning("⚠️ Advanced anti-detection not available or disabled")
        except Exception as e:
            self.logger.warning(f"⚠️ Anti-detection manager initialization failed: {e}")
            self.anti_detection_manager = None

    async def create_browser_session(
        self,
        browser_type: BrowserType = BrowserType.PLAYWRIGHT_CHROMIUM,
        profile: Optional[BrowserProfile] = None
    ) -> bool:
        """Create new browser session with specified configuration"""

        try:
            if profile:
                self.current_profile = profile

            self.logger.info(f"🌐 Creating {browser_type.value} session...")

            if browser_type == BrowserType.SELENIUM_CHROME:
                return await self._create_selenium_session()
            elif browser_type in [BrowserType.PLAYWRIGHT_CHROMIUM, BrowserType.PLAYWRIGHT_FIREFOX]:
                return await self._create_playwright_session(browser_type)
            else:
                raise ValueError(f"Unsupported browser type: {browser_type}")

        except Exception as e:
            self.logger.error(f"❌ Failed to create browser session: {e}")
            return False

    async def _create_selenium_session(self) -> bool:
        """Create Selenium Chrome session with stealth configuration"""

        if not SELENIUM_AVAILABLE:
            raise Exception("Selenium not available")

        try:
            chrome_options = ChromeOptions()

            # Basic configuration
            chrome_options.add_argument(f"--user-agent={self.current_profile.user_agent}")
            chrome_options.add_argument(f"--window-size={self.current_profile.viewport_width},{self.current_profile.viewport_height}")

            # Stealth configuration
            if self.stealth_level != StealthLevel.MINIMAL:
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-blink-features=AutomationControlled")
                chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
                chrome_options.add_experimental_option('useAutomationExtension', False)
                chrome_options.add_argument("--disable-web-security")
                chrome_options.add_argument("--disable-features=VizDisplayCompositor")

            # Headless mode for stealth
            if self.stealth_level == StealthLevel.MAXIMUM:
                chrome_options.add_argument("--headless")

            # Create driver
            self.selenium_driver = webdriver.Chrome(options=chrome_options)

            # Execute stealth scripts
            if self.stealth_level != StealthLevel.MINIMAL:
                self.selenium_driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

            self.logger.info("✅ Selenium Chrome session created")
            return True

        except Exception as e:
            self.logger.error(f"❌ Selenium session creation failed: {e}")
            return False

    async def _create_playwright_session(self, browser_type: BrowserType) -> bool:
        """Create Playwright session with anti-detection"""

        if not PLAYWRIGHT_AVAILABLE:
            raise Exception("Playwright not available")

        try:
            playwright = await async_playwright().start()

            # Browser launch options
            launch_options = {
                "headless": self.stealth_level == StealthLevel.MAXIMUM,
                "args": [
                    "--no-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    f"--window-size={self.current_profile.viewport_width},{self.current_profile.viewport_height}"
                ]
            }

            # Launch browser
            if browser_type == BrowserType.PLAYWRIGHT_CHROMIUM:
                self.playwright_browser = await playwright.chromium.launch(**launch_options)
            else:
                self.playwright_browser = await playwright.firefox.launch(**launch_options)

            # Create context with stealth settings
            context_options = {
                "viewport": {
                    "width": self.current_profile.viewport_width,
                    "height": self.current_profile.viewport_height
                },
                "user_agent": self.current_profile.user_agent,
                "locale": self.current_profile.locale,
                "timezone_id": self.current_profile.timezone
            }

            self.playwright_context = await self.playwright_browser.new_context(**context_options)

            # Add stealth scripts
            if self.stealth_level != StealthLevel.MINIMAL:
                await self.playwright_context.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                    Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                    Object.defineProperty(navigator, 'languages', {get: () => ['cs-CZ', 'cs', 'en']});
                """)

            # Create page
            self.playwright_page = await self.playwright_context.new_page()

            self.logger.info(f"✅ Playwright {browser_type.value} session created")
            return True

        except Exception as e:
            self.logger.error(f"❌ Playwright session creation failed: {e}")
            return False

    async def navigate_to_url(self, url: str, wait_for_selector: Optional[str] = None) -> bool:
        """Navigate to URL with intelligent waiting and error handling"""

        try:
            self.logger.info(f"🌐 Navigating to: {url}")

            # Human-like delay before navigation
            await self._human_delay()

            if self.playwright_page:
                # Playwright navigation
                response = await self.playwright_page.goto(url, wait_until="domcontentloaded")

                if wait_for_selector:
                    await self.playwright_page.wait_for_selector(wait_for_selector, timeout=10000)

                success = response.ok if response else False

            elif self.selenium_driver:
                # Selenium navigation
                self.selenium_driver.get(url)

                if wait_for_selector:
                    WebDriverWait(self.selenium_driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_selector))
                    )

                success = True
            else:
                raise Exception("No active browser session")

            # Update statistics
            self.scraping_stats['total_requests'] += 1
            if success:
                self.scraping_stats['successful_requests'] += 1

            self.logger.info(f"{'✅' if success else '❌'} Navigation {'successful' if success else 'failed'}")
            return success

        except Exception as e:
            self.logger.error(f"❌ Navigation failed: {e}")
            self.scraping_stats['total_requests'] += 1
            return False

    async def extract_data(self, selectors: Dict[str, str]) -> Dict[str, Any]:
        """Extract data using CSS selectors"""

        try:
            extracted_data = {}

            if self.playwright_page:
                # Playwright data extraction
                for field_name, selector in selectors.items():
                    try:
                        elements = await self.playwright_page.query_selector_all(selector)
                        if elements:
                            if len(elements) == 1:
                                extracted_data[field_name] = await elements[0].text_content()
                            else:
                                extracted_data[field_name] = [await elem.text_content() for elem in elements]
                        else:
                            extracted_data[field_name] = None
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to extract {field_name}: {e}")
                        extracted_data[field_name] = None

            elif self.selenium_driver:
                # Selenium data extraction
                for field_name, selector in selectors.items():
                    try:
                        elements = self.selenium_driver.find_elements(By.CSS_SELECTOR, selector)
                        if elements:
                            if len(elements) == 1:
                                extracted_data[field_name] = elements[0].text
                            else:
                                extracted_data[field_name] = [elem.text for elem in elements]
                        else:
                            extracted_data[field_name] = None
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to extract {field_name}: {e}")
                        extracted_data[field_name] = None
            else:
                raise Exception("No active browser session")

            self.logger.info(f"📊 Extracted {len(extracted_data)} data fields")
            return extracted_data

        except Exception as e:
            self.logger.error(f"❌ Data extraction failed: {e}")
            return {}

    async def perform_search(self, query: str, search_selector: str, submit_selector: str) -> bool:
        """Perform search operation with human-like behavior"""

        try:
            self.logger.info(f"🔍 Performing search: {query}")

            if self.playwright_page:
                # Playwright search
                search_input = await self.playwright_page.query_selector(search_selector)
                if search_input:
                    # Human-like typing
                    await search_input.click()
                    await search_input.clear()
                    await self._type_like_human(search_input, query)

                    # Submit search
                    submit_button = await self.playwright_page.query_selector(submit_selector)
                    if submit_button:
                        await submit_button.click()
                    else:
                        await search_input.press("Enter")

                    return True

            elif self.selenium_driver:
                # Selenium search
                search_input = self.selenium_driver.find_element(By.CSS_SELECTOR, search_selector)
                search_input.clear()

                # Human-like typing
                for char in query:
                    search_input.send_keys(char)
                    time.sleep(random.uniform(0.05, 0.15))

                # Submit search
                try:
                    submit_button = self.selenium_driver.find_element(By.CSS_SELECTOR, submit_selector)
                    submit_button.click()
                except:
                    search_input.send_keys("\n")

                return True

            return False

        except Exception as e:
            self.logger.error(f"❌ Search operation failed: {e}")
            return False

    async def _type_like_human(self, element, text: str):
        """Type text with human-like delays and patterns"""

        for char in text:
            await element.type(char)
            # Variable typing speed
            delay = random.uniform(0.05, 0.15)
            if char == ' ':
                delay *= 2  # Longer pause after spaces
            await asyncio.sleep(delay)

    async def _human_delay(self):
        """Introduce human-like delays between actions"""

        if self.stealth_level == StealthLevel.MINIMAL:
            delay = random.uniform(0.5, 1.5)
        elif self.stealth_level == StealthLevel.MODERATE:
            delay = random.uniform(1.0, 3.0)
        else:  # MAXIMUM
            delay = random.uniform(2.0, 5.0)

        await asyncio.sleep(delay)

    async def rotate_fingerprint(self):
        """Rotate browser fingerprint for anti-detection"""

        try:
            # Rotate user agent
            new_user_agent = random.choice(self.user_agents)

            # Create new profile
            new_profile = BrowserProfile(
                name=f"rotated_{datetime.now().strftime('%H%M%S')}",
                user_agent=new_user_agent,
                viewport_width=random.choice([1366, 1920, 1440, 1280]),
                viewport_height=random.choice([768, 1080, 900, 720])
            )

            # Close current session
            await self.close_session()

            # Create new session with rotated profile
            success = await self.create_browser_session(profile=new_profile)

            if success:
                self.scraping_stats['last_rotation'] = datetime.now().isoformat()
                self.logger.info("🔄 Browser fingerprint rotated successfully")

            return success

        except Exception as e:
            self.logger.error(f"❌ Fingerprint rotation failed: {e}")
            return False

    async def close_session(self):
        """Close current browser session and cleanup"""

        try:
            if self.selenium_driver:
                self.selenium_driver.quit()
                self.selenium_driver = None

            if self.playwright_page:
                await self.playwright_page.close()
                self.playwright_page = None

            if self.playwright_context:
                await self.playwright_context.close()
                self.playwright_context = None

            if self.playwright_browser:
                await self.playwright_browser.close()
                self.playwright_browser = None

            self.logger.info("🔒 Browser session closed")

        except Exception as e:
            self.logger.warning(f"⚠️ Session cleanup error: {e}")

    def get_scraping_statistics(self) -> Dict[str, Any]:
        """Get current scraping statistics"""

        success_rate = 0
        if self.scraping_stats['total_requests'] > 0:
            success_rate = self.scraping_stats['successful_requests'] / self.scraping_stats['total_requests']

        return {
            **self.scraping_stats,
            'success_rate': success_rate,
            'current_profile': self.current_profile.name if self.current_profile else None,
            'stealth_level': self.stealth_level.value,
            'last_updated': datetime.now().isoformat()
        }

# Example usage and testing
if __name__ == "__main__":
    async def test_browser_manager():
        """Test the enhanced browser manager"""

        # Initialize browser manager
        browser_manager = EnhancedBrowserManager(stealth_level=StealthLevel.MODERATE)
        await browser_manager.initialize()

        # Create browser session
        success = await browser_manager.create_browser_session(BrowserType.PLAYWRIGHT_CHROMIUM)

        if success:
            # Test navigation
            success = await browser_manager.navigate_to_url("https://httpbin.org/user-agent")

            if success:
                # Test data extraction
                data = await browser_manager.extract_data({
                    "user_agent": "pre"
                })

                print(f"📊 Extracted data: {data}")

            # Get statistics
            stats = browser_manager.get_scraping_statistics()
            print(f"📈 Statistics: {stats}")

            # Close session
            await browser_manager.close_session()

        return success

    # Run test
    if __name__ == "__main__":
        asyncio.run(test_browser_manager())
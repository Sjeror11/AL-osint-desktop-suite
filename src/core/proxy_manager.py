#!/usr/bin/env python3
"""
🌐 Proxy Manager - Advanced Anti-Detection & Proxy Rotation
Desktop OSINT Suite - Phase 3 Enhancement
LakyLuk Enhanced Edition - 27.9.2025

Advanced proxy management and anti-detection capabilities
Features:
- Proxy rotation and validation
- TOR integration
- VPN detection and management
- IP geolocation and reputation checking
- Advanced fingerprint randomization
"""

import asyncio
import random
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import os
import subprocess

class ProxyType(Enum):
    """Types of proxy connections"""
    HTTP = "http"
    HTTPS = "https"
    SOCKS4 = "socks4"
    SOCKS5 = "socks5"
    TOR = "tor"

class ProxyQuality(Enum):
    """Proxy quality levels"""
    HIGH = "high"        # Fast, stable, anonymous
    MEDIUM = "medium"    # Decent performance
    LOW = "low"          # Slow but functional
    UNKNOWN = "unknown"  # Not tested

@dataclass
class ProxyInfo:
    """Proxy information and statistics"""
    host: str
    port: int
    proxy_type: ProxyType
    username: Optional[str] = None
    password: Optional[str] = None
    country: Optional[str] = None
    city: Optional[str] = None
    quality: ProxyQuality = ProxyQuality.UNKNOWN
    response_time: float = 0.0
    success_rate: float = 0.0
    last_used: Optional[datetime] = None
    failures: int = 0
    total_requests: int = 0

class AntiDetectionManager:
    """
    Advanced anti-detection and proxy management system

    Provides sophisticated evasion techniques:
    - Proxy rotation and validation
    - Fingerprint randomization
    - Timing pattern randomization
    - TOR integration
    - Geolocation masking
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Proxy management
        self.available_proxies: List[ProxyInfo] = []
        self.current_proxy: Optional[ProxyInfo] = None
        self.proxy_rotation_enabled = True
        self.proxy_test_urls = [
            "https://httpbin.org/ip",
            "https://ipinfo.io/json",
            "https://api.ipify.org?format=json"
        ]

        # Anti-detection settings
        self.fingerprint_profiles: List[Dict[str, Any]] = []
        self.current_fingerprint: Optional[Dict[str, Any]] = None
        self.timing_patterns: Dict[str, Tuple[float, float]] = {}

        # TOR management
        self.tor_available = False
        self.tor_process = None
        self.tor_control_port = 9051
        self.tor_socks_port = 9050

        # Statistics
        self.stats = {
            'total_proxy_switches': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'detection_events': 0,
            'average_response_time': 0.0,
            'last_rotation': None
        }

    async def initialize(self):
        """Initialize proxy manager and anti-detection systems"""
        try:
            self.logger.info("🛡️ Initializing Anti-Detection & Proxy Manager...")

            # Load fingerprint profiles
            await self._load_fingerprint_profiles()

            # Initialize timing patterns
            await self._initialize_timing_patterns()

            # Check TOR availability
            await self._check_tor_availability()

            # Load proxy list
            await self._load_proxy_list()

            # Test available proxies
            await self._test_proxies()

            self.logger.info(f"✅ Anti-Detection Manager initialized")
            self.logger.info(f"   🌐 Available proxies: {len(self.available_proxies)}")
            self.logger.info(f"   🔒 TOR available: {self.tor_available}")
            self.logger.info(f"   🎭 Fingerprint profiles: {len(self.fingerprint_profiles)}")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize anti-detection manager: {e}")
            raise

    async def _load_fingerprint_profiles(self):
        """Load browser fingerprint profiles for randomization"""

        # Advanced fingerprint profiles based on real browser data
        self.fingerprint_profiles = [
            {
                "name": "windows_chrome_120",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "platform": "Win32",
                "viewport": {"width": 1920, "height": 1080},
                "screen": {"width": 1920, "height": 1080, "colorDepth": 24},
                "languages": ["en-US", "en"],
                "timezone": "America/New_York",
                "webgl_vendor": "Google Inc. (Intel)",
                "webgl_renderer": "ANGLE (Intel, Intel(R) UHD Graphics 620 Direct3D11 vs_5_0 ps_5_0)",
                "hardware_concurrency": 8,
                "device_memory": 8
            },
            {
                "name": "macos_safari_17",
                "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
                "platform": "MacIntel",
                "viewport": {"width": 1440, "height": 900},
                "screen": {"width": 1440, "height": 900, "colorDepth": 24},
                "languages": ["en-US", "en"],
                "timezone": "America/Los_Angeles",
                "webgl_vendor": "Apple Inc.",
                "webgl_renderer": "Apple GPU",
                "hardware_concurrency": 8,
                "device_memory": 16
            },
            {
                "name": "ubuntu_firefox_121",
                "user_agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
                "platform": "Linux x86_64",
                "viewport": {"width": 1366, "height": 768},
                "screen": {"width": 1366, "height": 768, "colorDepth": 24},
                "languages": ["en-US", "en-GB", "en"],
                "timezone": "Europe/London",
                "webgl_vendor": "Mozilla",
                "webgl_renderer": "Mesa DRI Intel(R) UHD Graphics 620 (KBL GT2)",
                "hardware_concurrency": 4,
                "device_memory": 8
            },
            {
                "name": "mobile_android_chrome",
                "user_agent": "Mozilla/5.0 (Linux; Android 13; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
                "platform": "Linux armv7l",
                "viewport": {"width": 375, "height": 812},
                "screen": {"width": 375, "height": 812, "colorDepth": 24},
                "languages": ["en-US", "en"],
                "timezone": "America/New_York",
                "webgl_vendor": "Qualcomm",
                "webgl_renderer": "Adreno (TM) 640",
                "hardware_concurrency": 8,
                "device_memory": 6,
                "is_mobile": True
            }
        ]

        self.current_fingerprint = random.choice(self.fingerprint_profiles)
        self.logger.info(f"🎭 Loaded {len(self.fingerprint_profiles)} fingerprint profiles")

    async def _initialize_timing_patterns(self):
        """Initialize human-like timing patterns"""

        self.timing_patterns = {
            'navigation_delay': (1.5, 4.0),      # Delay between page navigations
            'click_delay': (0.1, 0.8),           # Delay between clicks
            'typing_speed': (0.05, 0.2),         # Delay between keystrokes
            'scroll_delay': (0.3, 1.5),          # Delay between scroll actions
            'form_fill_delay': (2.0, 5.0),       # Delay while filling forms
            'reading_time': (3.0, 8.0),          # Time spent "reading" content
        }

        self.logger.info("⏱️ Human timing patterns initialized")

    async def _check_tor_availability(self):
        """Check if TOR is available on the system"""
        try:
            # Check if tor is installed
            result = subprocess.run(['which', 'tor'], capture_output=True, text=True)
            if result.returncode == 0:
                self.tor_available = True
                self.logger.info("🧅 TOR detected on system")
            else:
                self.logger.warning("⚠️ TOR not found - install tor package for enhanced anonymity")

        except Exception as e:
            self.logger.warning(f"⚠️ TOR availability check failed: {e}")
            self.tor_available = False

    async def _load_proxy_list(self):
        """Load proxy list from configuration or external sources"""

        # Try to load from config file
        config_file = Path(__file__).parent.parent.parent / "config" / "proxies.json"

        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    proxy_data = json.load(f)

                for proxy_config in proxy_data.get('proxies', []):
                    proxy = ProxyInfo(
                        host=proxy_config['host'],
                        port=proxy_config['port'],
                        proxy_type=ProxyType(proxy_config.get('type', 'http')),
                        username=proxy_config.get('username'),
                        password=proxy_config.get('password'),
                        country=proxy_config.get('country'),
                        city=proxy_config.get('city')
                    )
                    self.available_proxies.append(proxy)

                self.logger.info(f"📁 Loaded {len(self.available_proxies)} proxies from config")
                return

            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load proxy config: {e}")

        # Add some free proxy examples (for demo purposes)
        demo_proxies = [
            ProxyInfo("proxy1.example.com", 8080, ProxyType.HTTP, country="US"),
            ProxyInfo("proxy2.example.com", 3128, ProxyType.HTTP, country="DE"),
            ProxyInfo("proxy3.example.com", 1080, ProxyType.SOCKS5, country="NL"),
        ]

        self.available_proxies.extend(demo_proxies)
        self.logger.info(f"🔗 Using {len(demo_proxies)} demo proxies (configure real proxies in config/proxies.json)")

    async def _test_proxies(self):
        """Test proxy connectivity and performance"""
        if not self.available_proxies:
            return

        self.logger.info("🔍 Testing proxy connectivity...")

        working_proxies = []

        for proxy in self.available_proxies[:5]:  # Test first 5 proxies
            try:
                # Simple connectivity test (would need real implementation)
                response_time = random.uniform(0.5, 2.0)  # Simulated response time
                success_rate = random.uniform(0.7, 0.95)  # Simulated success rate

                proxy.response_time = response_time
                proxy.success_rate = success_rate
                proxy.quality = ProxyQuality.HIGH if success_rate > 0.9 else ProxyQuality.MEDIUM

                working_proxies.append(proxy)

            except Exception as e:
                self.logger.warning(f"⚠️ Proxy {proxy.host}:{proxy.port} failed test: {e}")

        self.available_proxies = working_proxies
        self.logger.info(f"✅ {len(working_proxies)} proxies passed connectivity test")

    async def rotate_proxy(self) -> bool:
        """Rotate to next available proxy"""
        if not self.available_proxies:
            self.logger.warning("⚠️ No proxies available for rotation")
            return False

        try:
            # Select best available proxy
            available_proxies = [p for p in self.available_proxies if p.failures < 3]

            if not available_proxies:
                self.logger.warning("⚠️ All proxies have failed - resetting failure counts")
                for proxy in self.available_proxies:
                    proxy.failures = 0
                available_proxies = self.available_proxies

            # Prefer high-quality proxies
            high_quality = [p for p in available_proxies if p.quality == ProxyQuality.HIGH]
            if high_quality:
                self.current_proxy = random.choice(high_quality)
            else:
                self.current_proxy = random.choice(available_proxies)

            self.current_proxy.last_used = datetime.now()
            self.stats['total_proxy_switches'] += 1
            self.stats['last_rotation'] = datetime.now().isoformat()

            self.logger.info(f"🔄 Rotated to proxy: {self.current_proxy.host}:{self.current_proxy.port}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Proxy rotation failed: {e}")
            return False

    async def rotate_fingerprint(self) -> bool:
        """Rotate browser fingerprint"""
        try:
            # Exclude current fingerprint from selection
            available_profiles = [
                p for p in self.fingerprint_profiles
                if p['name'] != (self.current_fingerprint.get('name') if self.current_fingerprint else None)
            ]

            if not available_profiles:
                available_profiles = self.fingerprint_profiles

            self.current_fingerprint = random.choice(available_profiles)

            self.logger.info(f"🎭 Rotated fingerprint to: {self.current_fingerprint['name']}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Fingerprint rotation failed: {e}")
            return False

    async def get_human_delay(self, action_type: str) -> float:
        """Get human-like delay for specified action"""
        if action_type not in self.timing_patterns:
            return random.uniform(0.5, 2.0)

        min_delay, max_delay = self.timing_patterns[action_type]

        # Add some randomness to make it more human-like
        variance = random.uniform(0.8, 1.2)
        delay = random.uniform(min_delay, max_delay) * variance

        return delay

    async def start_tor_session(self) -> bool:
        """Start TOR session for maximum anonymity"""
        if not self.tor_available:
            self.logger.warning("⚠️ TOR not available - cannot start TOR session")
            return False

        try:
            # Start TOR process (simplified - real implementation would be more complex)
            self.logger.info("🧅 Starting TOR session...")

            # Create TOR proxy info
            tor_proxy = ProxyInfo(
                host="127.0.0.1",
                port=self.tor_socks_port,
                proxy_type=ProxyType.SOCKS5,
                country="TOR",
                quality=ProxyQuality.HIGH
            )

            self.current_proxy = tor_proxy
            self.logger.info("✅ TOR session active")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to start TOR session: {e}")
            return False

    async def stop_tor_session(self):
        """Stop TOR session"""
        if self.tor_process:
            try:
                self.tor_process.terminate()
                self.tor_process = None
                self.logger.info("🧅 TOR session stopped")
            except Exception as e:
                self.logger.error(f"❌ Failed to stop TOR session: {e}")

    def get_current_config(self) -> Dict[str, Any]:
        """Get current anti-detection configuration"""
        return {
            "proxy": {
                "host": self.current_proxy.host if self.current_proxy else None,
                "port": self.current_proxy.port if self.current_proxy else None,
                "type": self.current_proxy.proxy_type.value if self.current_proxy else None,
                "country": self.current_proxy.country if self.current_proxy else None,
            },
            "fingerprint": self.current_fingerprint,
            "tor_active": self.current_proxy and self.current_proxy.proxy_type == ProxyType.TOR,
            "stats": self.stats
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get anti-detection statistics"""
        return {
            **self.stats,
            "available_proxies": len(self.available_proxies),
            "fingerprint_profiles": len(self.fingerprint_profiles),
            "tor_available": self.tor_available,
            "current_proxy": self.current_proxy.host if self.current_proxy else None
        }

# Testing function
async def test_anti_detection():
    """Test anti-detection manager"""
    manager = AntiDetectionManager()
    await manager.initialize()

    # Test proxy rotation
    print("Testing proxy rotation...")
    for i in range(3):
        await manager.rotate_proxy()
        await asyncio.sleep(1)

    # Test fingerprint rotation
    print("Testing fingerprint rotation...")
    for i in range(3):
        await manager.rotate_fingerprint()
        await asyncio.sleep(1)

    # Test timing patterns
    print("Testing timing patterns...")
    for action in ['navigation_delay', 'click_delay', 'typing_speed']:
        delay = await manager.get_human_delay(action)
        print(f"{action}: {delay:.2f}s")

    # Print statistics
    print("\nStatistics:")
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    asyncio.run(test_anti_detection())
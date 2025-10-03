#!/usr/bin/env python3
"""
⏱️ Rate Limiter - Mock Version for Testing
"""

import asyncio
import time

class RateLimiter:
    """Mock rate limiter for testing"""

    def __init__(self, requests_per_minute=60, requests_per_hour=1000):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.request_times = []

    async def wait_if_needed(self):
        """Mock rate limiting - minimal delay for testing"""
        await asyncio.sleep(0.01)  # Minimal delay for testing
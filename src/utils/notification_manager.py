#!/usr/bin/env python3
"""
📱 Notification Manager - Mock Version for Testing
"""

class NotificationManager:
    """Mock notification manager for testing"""

    async def send_alert_notification(self, alert):
        """Mock alert notification"""
        print(f"ALERT: {alert.title} - {alert.message}")
        return True
#!/usr/bin/env python3
"""
⚡ Real-time Social Media Monitor - Live Investigation Tracking
LakyLuk OSINT Investigation Suite

Features:
✅ Real-time monitoring of target profiles across platforms
✅ Live activity tracking with instant notifications
✅ Automated content archiving and change detection
✅ AI-powered anomaly detection and pattern analysis
✅ Multi-threaded monitoring with intelligent rate limiting
✅ Event-driven alerts for investigation triggers

Monitoring Capabilities:
- Profile changes (bio, picture, location, etc.)
- New posts and content updates
- Connection/follower changes
- Activity pattern anomalies
- Keyword and hashtag mentions
- Geographic location updates
"""

import asyncio
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import sqlite3

from .facebook_scanner import FacebookScanner
from .instagram_scanner import InstagramScanner
from .linkedin_scanner import LinkedInScanner
from ...analytics.entity_correlation_engine import EntityProfile
from ...core.enhanced_orchestrator import EnhancedInvestigationOrchestrator
from ...utils.notification_manager import NotificationManager
from ...utils.data_archiver import DataArchiver


@dataclass
class MonitoringTarget:
    """Target profile for real-time monitoring"""
    target_id: str
    platform: str
    username: str
    profile_url: str
    monitoring_type: str  # 'profile', 'content', 'network', 'all'
    keywords: List[str] = None
    check_interval: int = 300  # seconds
    active: bool = True
    created_at: datetime = None
    last_checked: Optional[datetime] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MonitoringEvent:
    """Event detected during monitoring"""
    event_id: str
    target_id: str
    event_type: str
    timestamp: datetime
    description: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    confidence: float = 1.0
    ai_analysis: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.ai_analysis is None:
            self.ai_analysis = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MonitoringAlert:
    """Alert triggered by monitoring events"""
    alert_id: str
    target_id: str
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    title: str
    message: str
    timestamp: datetime
    events: List[MonitoringEvent]
    acknowledged: bool = False
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RealTimeMonitor:
    """Real-time social media monitoring system"""

    def __init__(self, ai_orchestrator: EnhancedInvestigationOrchestrator = None):
        self.ai_orchestrator = ai_orchestrator
        self.notification_manager = NotificationManager()
        self.data_archiver = DataArchiver()

        # Platform scanners
        self.scanners = {
            'facebook': FacebookScanner(ai_orchestrator),
            'instagram': InstagramScanner(ai_orchestrator),
            'linkedin': LinkedInScanner(ai_orchestrator)
        }

        # Monitoring state
        self.targets = {}  # target_id -> MonitoringTarget
        self.events = []  # List of MonitoringEvent
        self.alerts = []  # List of MonitoringAlert
        self.monitoring_active = False

        # Threading and queues
        self.monitor_threads = {}
        self.event_queue = Queue()
        self.alert_queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=10)

        # Event handlers
        self.event_handlers = {}
        self.alert_handlers = {}

        # Configuration
        self.config = {
            'max_concurrent_monitors': 20,
            'default_check_interval': 300,  # 5 minutes
            'batch_size': 5,
            'enable_ai_analysis': True,
            'auto_archive_events': True,
            'alert_cooldown': 3600,  # 1 hour between similar alerts
            'profile_change_threshold': 0.1,
            'anomaly_detection_threshold': 0.8
        }

        # Database for persistence
        self.db_path = "monitoring_data.db"
        self._init_database()

    async def add_monitoring_target(self, target: MonitoringTarget) -> bool:
        """
        Add a new target for real-time monitoring

        Args:
            target: MonitoringTarget to add

        Returns:
            True if target was added successfully
        """
        try:
            # Validate target
            if not await self._validate_target(target):
                return False

            # Store target
            self.targets[target.target_id] = target

            # Start monitoring thread
            if self.monitoring_active:
                await self._start_target_monitoring(target)

            # Save to database
            self._save_target_to_db(target)

            # Generate initial event
            initial_event = MonitoringEvent(
                event_id=self._generate_event_id(),
                target_id=target.target_id,
                event_type='monitoring_started',
                timestamp=datetime.now(),
                description=f"Started monitoring {target.platform} profile {target.username}"
            )
            await self._process_event(initial_event)

            return True

        except Exception as e:
            print(f"Error adding monitoring target: {e}")
            return False

    async def remove_monitoring_target(self, target_id: str) -> bool:
        """
        Remove a target from monitoring

        Args:
            target_id: ID of target to remove

        Returns:
            True if target was removed successfully
        """
        try:
            if target_id not in self.targets:
                return False

            # Stop monitoring thread
            if target_id in self.monitor_threads:
                self.monitor_threads[target_id].cancel()
                del self.monitor_threads[target_id]

            # Remove from memory
            del self.targets[target_id]

            # Remove from database
            self._remove_target_from_db(target_id)

            # Generate removal event
            removal_event = MonitoringEvent(
                event_id=self._generate_event_id(),
                target_id=target_id,
                event_type='monitoring_stopped',
                timestamp=datetime.now(),
                description=f"Stopped monitoring target {target_id}"
            )
            await self._process_event(removal_event)

            return True

        except Exception as e:
            print(f"Error removing monitoring target: {e}")
            return False

    async def start_monitoring(self) -> bool:
        """
        Start the real-time monitoring system

        Returns:
            True if monitoring started successfully
        """
        try:
            if self.monitoring_active:
                return True

            self.monitoring_active = True

            # Start monitoring threads for all targets
            for target in self.targets.values():
                if target.active:
                    await self._start_target_monitoring(target)

            # Start event processing
            self._start_event_processor()

            # Start alert processing
            self._start_alert_processor()

            print(f"Real-time monitoring started for {len(self.targets)} targets")
            return True

        except Exception as e:
            print(f"Error starting monitoring: {e}")
            return False

    async def stop_monitoring(self) -> bool:
        """
        Stop the real-time monitoring system

        Returns:
            True if monitoring stopped successfully
        """
        try:
            self.monitoring_active = False

            # Cancel all monitoring threads
            for thread in self.monitor_threads.values():
                thread.cancel()
            self.monitor_threads.clear()

            print("Real-time monitoring stopped")
            return True

        except Exception as e:
            print(f"Error stopping monitoring: {e}")
            return False

    async def get_monitoring_status(self) -> Dict[str, Any]:
        """
        Get current monitoring system status

        Returns:
            Comprehensive monitoring status
        """
        status = {
            'monitoring_active': self.monitoring_active,
            'total_targets': len(self.targets),
            'active_targets': len([t for t in self.targets.values() if t.active]),
            'total_events': len(self.events),
            'unacknowledged_alerts': len([a for a in self.alerts if not a.acknowledged]),
            'platform_distribution': {},
            'recent_events': [],
            'system_health': {}
        }

        # Platform distribution
        for target in self.targets.values():
            platform = target.platform
            status['platform_distribution'][platform] = status['platform_distribution'].get(platform, 0) + 1

        # Recent events (last 24 hours)
        recent_threshold = datetime.now() - timedelta(hours=24)
        status['recent_events'] = [
            event for event in self.events[-100:]  # Last 100 events
            if event.timestamp >= recent_threshold
        ]

        # System health
        status['system_health'] = {
            'thread_count': len(self.monitor_threads),
            'queue_sizes': {
                'events': self.event_queue.qsize(),
                'alerts': self.alert_queue.qsize()
            },
            'last_update': datetime.now().isoformat()
        }

        return status

    async def register_event_handler(self, event_type: str, handler: Callable):
        """Register custom event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    async def register_alert_handler(self, alert_type: str, handler: Callable):
        """Register custom alert handler"""
        if alert_type not in self.alert_handlers:
            self.alert_handlers[alert_type] = []
        self.alert_handlers[alert_type].append(handler)

    # Internal monitoring methods

    async def _start_target_monitoring(self, target: MonitoringTarget):
        """Start monitoring thread for a specific target"""
        if target.target_id in self.monitor_threads:
            return  # Already monitoring

        # Create monitoring task
        task = asyncio.create_task(self._monitor_target_loop(target))
        self.monitor_threads[target.target_id] = task

    async def _monitor_target_loop(self, target: MonitoringTarget):
        """Main monitoring loop for a target"""
        last_state = None

        while self.monitoring_active and target.active:
            try:
                # Get current profile state
                current_state = await self._get_target_state(target)

                if current_state:
                    # Compare with last state
                    if last_state:
                        changes = await self._detect_changes(target, last_state, current_state)
                        for change in changes:
                            await self._process_event(change)

                    last_state = current_state
                    target.last_checked = datetime.now()

                    # Archive current state if enabled
                    if self.config['auto_archive_events']:
                        await self.data_archiver.archive_profile_state(target.target_id, current_state)

                # Wait for next check
                await asyncio.sleep(target.check_interval)

            except Exception as e:
                print(f"Error monitoring target {target.target_id}: {e}")
                # Exponential backoff on error
                await asyncio.sleep(min(target.check_interval * 2, 3600))

    async def _get_target_state(self, target: MonitoringTarget) -> Optional[Dict[str, Any]]:
        """Get current state of a monitoring target"""
        try:
            scanner = self.scanners.get(target.platform)
            if not scanner:
                return None

            if target.platform == 'instagram':
                profile_data = await scanner.analyze_profile(target.username)
            elif target.platform == 'facebook':
                # Facebook requires special handling
                profile_data = await scanner.analyze_profile(target.profile_url)
            elif target.platform == 'linkedin':
                profile_data = await scanner.analyze_professional_profile(target.profile_url)
            else:
                return None

            return profile_data

        except Exception as e:
            print(f"Error getting target state for {target.target_id}: {e}")
            return None

    async def _detect_changes(self, target: MonitoringTarget, old_state: Dict, new_state: Dict) -> List[MonitoringEvent]:
        """Detect changes between two profile states"""
        events = []

        try:
            # Profile information changes
            profile_changes = self._compare_profile_info(old_state, new_state)
            for change in profile_changes:
                event = MonitoringEvent(
                    event_id=self._generate_event_id(),
                    target_id=target.target_id,
                    event_type=f"profile_{change['field']}_changed",
                    timestamp=datetime.now(),
                    description=f"Profile {change['field']} changed",
                    old_value=change['old_value'],
                    new_value=change['new_value']
                )
                events.append(event)

            # Content changes
            content_changes = self._compare_content(old_state, new_state)
            for change in content_changes:
                event = MonitoringEvent(
                    event_id=self._generate_event_id(),
                    target_id=target.target_id,
                    event_type='new_content',
                    timestamp=datetime.now(),
                    description=f"New {change['content_type']} posted",
                    new_value=change['content']
                )
                events.append(event)

            # Follower/connection changes
            network_changes = self._compare_network_metrics(old_state, new_state)
            for change in network_changes:
                event = MonitoringEvent(
                    event_id=self._generate_event_id(),
                    target_id=target.target_id,
                    event_type=f"network_{change['metric']}_changed",
                    timestamp=datetime.now(),
                    description=f"{change['metric']} count changed by {change['delta']}",
                    old_value=change['old_value'],
                    new_value=change['new_value']
                )
                events.append(event)

            # AI-powered anomaly detection
            if self.ai_orchestrator and self.config['enable_ai_analysis']:
                anomalies = await self.ai_orchestrator.detect_profile_anomalies(
                    old_state, new_state, target.metadata.get('baseline', {})
                )
                for anomaly in anomalies:
                    event = MonitoringEvent(
                        event_id=self._generate_event_id(),
                        target_id=target.target_id,
                        event_type='anomaly_detected',
                        timestamp=datetime.now(),
                        description=anomaly['description'],
                        confidence=anomaly['confidence'],
                        ai_analysis=anomaly
                    )
                    events.append(event)

        except Exception as e:
            print(f"Error detecting changes: {e}")

        return events

    async def _process_event(self, event: MonitoringEvent):
        """Process a monitoring event"""
        try:
            # Add to events list
            self.events.append(event)

            # Put in processing queue
            self.event_queue.put(event)

            # Call registered handlers
            if event.event_type in self.event_handlers:
                for handler in self.event_handlers[event.event_type]:
                    try:
                        await handler(event)
                    except Exception as e:
                        print(f"Error in event handler: {e}")

            # Check if event should trigger an alert
            alert = await self._evaluate_alert_conditions(event)
            if alert:
                await self._process_alert(alert)

            # Save to database
            self._save_event_to_db(event)

        except Exception as e:
            print(f"Error processing event: {e}")

    async def _evaluate_alert_conditions(self, event: MonitoringEvent) -> Optional[MonitoringAlert]:
        """Evaluate if event should trigger an alert"""
        try:
            # High-priority event types
            high_priority_events = [
                'profile_picture_changed',
                'profile_bio_changed',
                'profile_location_changed',
                'anomaly_detected'
            ]

            # Critical event types
            critical_events = [
                'account_deleted',
                'account_suspended',
                'privacy_settings_changed'
            ]

            alert_type = None
            severity = 'low'

            if event.event_type in critical_events:
                alert_type = 'critical_change'
                severity = 'critical'
            elif event.event_type in high_priority_events:
                alert_type = 'significant_change'
                severity = 'high'
            elif event.event_type == 'new_content':
                # Check for keyword matches
                if self._check_keyword_matches(event):
                    alert_type = 'keyword_match'
                    severity = 'medium'

            if alert_type:
                # Check cooldown period
                if not self._is_alert_on_cooldown(event.target_id, alert_type):
                    return MonitoringAlert(
                        alert_id=self._generate_alert_id(),
                        target_id=event.target_id,
                        alert_type=alert_type,
                        severity=severity,
                        title=f"Alert: {event.event_type}",
                        message=event.description,
                        timestamp=datetime.now(),
                        events=[event]
                    )

        except Exception as e:
            print(f"Error evaluating alert conditions: {e}")

        return None

    async def _process_alert(self, alert: MonitoringAlert):
        """Process a monitoring alert"""
        try:
            # Add to alerts list
            self.alerts.append(alert)

            # Put in alert queue
            self.alert_queue.put(alert)

            # Send notifications
            await self.notification_manager.send_alert_notification(alert)

            # Call registered handlers
            if alert.alert_type in self.alert_handlers:
                for handler in self.alert_handlers[alert.alert_type]:
                    try:
                        await handler(alert)
                    except Exception as e:
                        print(f"Error in alert handler: {e}")

            # Save to database
            self._save_alert_to_db(alert)

        except Exception as e:
            print(f"Error processing alert: {e}")

    def _start_event_processor(self):
        """Start event processing thread"""
        def process_events():
            while self.monitoring_active:
                try:
                    event = self.event_queue.get(timeout=1)
                    # Additional event processing can be done here
                    self.event_queue.task_done()
                except Empty:
                    continue
                except Exception as e:
                    print(f"Error in event processor: {e}")

        thread = threading.Thread(target=process_events, daemon=True)
        thread.start()

    def _start_alert_processor(self):
        """Start alert processing thread"""
        def process_alerts():
            while self.monitoring_active:
                try:
                    alert = self.alert_queue.get(timeout=1)
                    # Additional alert processing can be done here
                    self.alert_queue.task_done()
                except Empty:
                    continue
                except Exception as e:
                    print(f"Error in alert processor: {e}")

        thread = threading.Thread(target=process_alerts, daemon=True)
        thread.start()

    # Utility methods

    def _compare_profile_info(self, old_state: Dict, new_state: Dict) -> List[Dict]:
        """Compare profile information fields"""
        changes = []
        fields_to_check = ['display_name', 'bio', 'location', 'profile_picture_url']

        for field in fields_to_check:
            old_val = old_state.get('basic_info', {}).get(field)
            new_val = new_state.get('basic_info', {}).get(field)

            if old_val != new_val:
                changes.append({
                    'field': field,
                    'old_value': old_val,
                    'new_value': new_val
                })

        return changes

    def _compare_content(self, old_state: Dict, new_state: Dict) -> List[Dict]:
        """Compare content between states"""
        changes = []

        old_posts = old_state.get('content_analysis', {}).get('recent_posts', [])
        new_posts = new_state.get('content_analysis', {}).get('recent_posts', [])

        # Simple check for new posts (in real implementation, would use more sophisticated comparison)
        if len(new_posts) > len(old_posts):
            for i in range(len(old_posts), len(new_posts)):
                changes.append({
                    'content_type': 'post',
                    'content': new_posts[i]
                })

        return changes

    def _compare_network_metrics(self, old_state: Dict, new_state: Dict) -> List[Dict]:
        """Compare network metrics"""
        changes = []
        metrics_to_check = ['followers', 'following', 'connections_count']

        for metric in metrics_to_check:
            old_val = old_state.get('basic_info', {}).get(metric, 0)
            new_val = new_state.get('basic_info', {}).get(metric, 0)

            if old_val != new_val and abs(old_val - new_val) > 0:
                changes.append({
                    'metric': metric,
                    'old_value': old_val,
                    'new_value': new_val,
                    'delta': new_val - old_val
                })

        return changes

    def _check_keyword_matches(self, event: MonitoringEvent) -> bool:
        """Check if event content matches target keywords"""
        target = self.targets.get(event.target_id)
        if not target or not target.keywords:
            return False

        content = str(event.new_value).lower() if event.new_value else ""
        return any(keyword.lower() in content for keyword in target.keywords)

    def _is_alert_on_cooldown(self, target_id: str, alert_type: str) -> bool:
        """Check if alert type is on cooldown for target"""
        cooldown_period = timedelta(seconds=self.config['alert_cooldown'])
        cutoff_time = datetime.now() - cooldown_period

        for alert in self.alerts:
            if (alert.target_id == target_id and
                alert.alert_type == alert_type and
                alert.timestamp > cutoff_time):
                return True

        return False

    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        import uuid
        return str(uuid.uuid4())[:8]

    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        import uuid
        return str(uuid.uuid4())[:8]

    # Database methods

    def _init_database(self):
        """Initialize SQLite database for persistence"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS monitoring_targets (
                    target_id TEXT PRIMARY KEY,
                    platform TEXT,
                    username TEXT,
                    profile_url TEXT,
                    monitoring_type TEXT,
                    keywords TEXT,
                    check_interval INTEGER,
                    active BOOLEAN,
                    created_at TIMESTAMP,
                    metadata TEXT
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS monitoring_events (
                    event_id TEXT PRIMARY KEY,
                    target_id TEXT,
                    event_type TEXT,
                    timestamp TIMESTAMP,
                    description TEXT,
                    old_value TEXT,
                    new_value TEXT,
                    confidence REAL,
                    metadata TEXT
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS monitoring_alerts (
                    alert_id TEXT PRIMARY KEY,
                    target_id TEXT,
                    alert_type TEXT,
                    severity TEXT,
                    title TEXT,
                    message TEXT,
                    timestamp TIMESTAMP,
                    acknowledged BOOLEAN,
                    metadata TEXT
                )
            ''')

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Database initialization error: {e}")

    def _save_target_to_db(self, target: MonitoringTarget):
        """Save target to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO monitoring_targets
                (target_id, platform, username, profile_url, monitoring_type,
                 keywords, check_interval, active, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                target.target_id, target.platform, target.username, target.profile_url,
                target.monitoring_type, json.dumps(target.keywords), target.check_interval,
                target.active, target.created_at, json.dumps(target.metadata)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Error saving target to database: {e}")

    def _save_event_to_db(self, event: MonitoringEvent):
        """Save event to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO monitoring_events
                (event_id, target_id, event_type, timestamp, description,
                 old_value, new_value, confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_id, event.target_id, event.event_type, event.timestamp,
                event.description, json.dumps(event.old_value) if event.old_value else None,
                json.dumps(event.new_value) if event.new_value else None,
                event.confidence, json.dumps(event.metadata)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Error saving event to database: {e}")

    def _save_alert_to_db(self, alert: MonitoringAlert):
        """Save alert to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO monitoring_alerts
                (alert_id, target_id, alert_type, severity, title, message,
                 timestamp, acknowledged, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id, alert.target_id, alert.alert_type, alert.severity,
                alert.title, alert.message, alert.timestamp, alert.acknowledged,
                json.dumps(alert.metadata)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Error saving alert to database: {e}")

    async def _validate_target(self, target: MonitoringTarget) -> bool:
        """Validate monitoring target"""
        # Check if platform is supported
        if target.platform not in self.scanners:
            return False

        # Check if target already exists
        if target.target_id in self.targets:
            return False

        # Additional validation can be added here
        return True
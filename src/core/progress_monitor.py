#!/usr/bin/env python3
"""
📊 Progress Monitor - Real-time Investigation Progress Tracking
OSINT Desktop Suite - Phase 2 Core Engine Component
LakyLuk Enhanced Edition - 4.10.2025

Features:
- Real-time investigation progress tracking
- Multi-phase monitoring with detailed metrics
- Event-based notification system
- Performance analytics and optimization
- Live dashboard integration support
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import threading


class InvestigationPhase(Enum):
    """Investigation workflow phases"""
    INITIALIZATION = "initialization"
    INFORMATION_GATHERING = "information_gathering"
    WEB_SEARCH = "web_search"
    SOCIAL_MEDIA = "social_media"
    GOVERNMENT_DATABASES = "government_databases"
    ENTITY_CORRELATION = "entity_correlation"
    AI_ANALYSIS = "ai_analysis"
    REPORT_GENERATION = "report_generation"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatus(Enum):
    """Individual task status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProgressTask:
    """Individual task within investigation"""
    task_id: str
    phase: InvestigationPhase
    name: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    progress_percent: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'task_id': self.task_id,
            'phase': self.phase.value,
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'progress_percent': self.progress_percent,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_seconds': self.duration_seconds,
            'error_message': self.error_message,
            'metadata': self.metadata
        }


@dataclass
class PhaseProgress:
    """Progress tracking for investigation phase"""
    phase: InvestigationPhase
    status: TaskStatus = TaskStatus.PENDING
    tasks: List[ProgressTask] = field(default_factory=list)
    progress_percent: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def calculate_progress(self) -> float:
        """Calculate overall phase progress"""
        if not self.tasks:
            return 0.0

        total_progress = sum(task.progress_percent for task in self.tasks)
        return total_progress / len(self.tasks)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'phase': self.phase.value,
            'status': self.status.value,
            'progress_percent': self.calculate_progress(),
            'tasks': [task.to_dict() for task in self.tasks],
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


class ProgressMonitor:
    """
    Real-time Investigation Progress Monitor

    Tracks investigation progress across multiple phases and tasks,
    provides real-time updates, and integrates with dashboard visualization.
    """

    def __init__(self, investigation_id: str, output_dir: Optional[Path] = None):
        """
        Initialize progress monitor

        Args:
            investigation_id: Unique investigation identifier
            output_dir: Directory for progress logs (optional)
        """
        self.investigation_id = investigation_id
        self.logger = logging.getLogger(__name__)

        # Setup output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path.cwd() / "investigations" / investigation_id / "progress"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking
        self.phases: Dict[InvestigationPhase, PhaseProgress] = {}
        self.current_phase: Optional[InvestigationPhase] = None
        self.overall_status = TaskStatus.PENDING

        # Timing
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

        # Event callbacks
        self.callbacks: List[Callable] = []

        # Performance metrics
        self.metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'skipped_tasks': 0,
            'total_duration': 0.0,
            'average_task_duration': 0.0
        }

        # Thread safety
        self.lock = threading.Lock()

        self.logger.info(f"📊 Progress Monitor initialized for investigation: {investigation_id}")

    def start_investigation(self):
        """Mark investigation start"""
        with self.lock:
            self.started_at = datetime.now()
            self.overall_status = TaskStatus.IN_PROGRESS
            self._save_progress()
            self._notify_callbacks('investigation_started', {
                'investigation_id': self.investigation_id,
                'started_at': self.started_at.isoformat()
            })

    def register_phase(self, phase: InvestigationPhase, tasks: List[Dict[str, str]]):
        """
        Register investigation phase with tasks

        Args:
            phase: Investigation phase
            tasks: List of task definitions (name, description)
        """
        with self.lock:
            phase_tasks = [
                ProgressTask(
                    task_id=f"{phase.value}_{i}",
                    phase=phase,
                    name=task['name'],
                    description=task['description']
                )
                for i, task in enumerate(tasks)
            ]

            self.phases[phase] = PhaseProgress(
                phase=phase,
                tasks=phase_tasks
            )

            self.metrics['total_tasks'] += len(phase_tasks)

            self.logger.info(f"✅ Registered phase {phase.value} with {len(phase_tasks)} tasks")

    def start_phase(self, phase: InvestigationPhase):
        """Start investigation phase"""
        with self.lock:
            if phase not in self.phases:
                self.logger.warning(f"⚠️ Phase {phase.value} not registered")
                return

            self.current_phase = phase
            self.phases[phase].status = TaskStatus.IN_PROGRESS
            self.phases[phase].started_at = datetime.now()

            self._save_progress()
            self._notify_callbacks('phase_started', {
                'phase': phase.value,
                'started_at': self.phases[phase].started_at.isoformat()
            })

            self.logger.info(f"🚀 Started phase: {phase.value}")

    def start_task(self, phase: InvestigationPhase, task_id: str):
        """Start specific task within phase"""
        with self.lock:
            if phase not in self.phases:
                return

            for task in self.phases[phase].tasks:
                if task.task_id == task_id:
                    task.status = TaskStatus.IN_PROGRESS
                    task.started_at = datetime.now()
                    task.progress_percent = 0.0

                    self._save_progress()
                    self._notify_callbacks('task_started', {
                        'phase': phase.value,
                        'task_id': task_id,
                        'task_name': task.name
                    })

                    self.logger.debug(f"▶️ Started task: {task.name}")
                    break

    def update_task_progress(self, phase: InvestigationPhase, task_id: str,
                           progress_percent: float, metadata: Optional[Dict] = None):
        """Update task progress"""
        with self.lock:
            if phase not in self.phases:
                return

            for task in self.phases[phase].tasks:
                if task.task_id == task_id:
                    task.progress_percent = min(100.0, max(0.0, progress_percent))

                    if metadata:
                        task.metadata.update(metadata)

                    self._save_progress()
                    self._notify_callbacks('task_progress', {
                        'phase': phase.value,
                        'task_id': task_id,
                        'progress': task.progress_percent,
                        'metadata': metadata
                    })
                    break

    def complete_task(self, phase: InvestigationPhase, task_id: str,
                     metadata: Optional[Dict] = None):
        """Mark task as completed"""
        with self.lock:
            if phase not in self.phases:
                return

            for task in self.phases[phase].tasks:
                if task.task_id == task_id:
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                    task.progress_percent = 100.0

                    if task.started_at:
                        task.duration_seconds = (task.completed_at - task.started_at).total_seconds()

                    if metadata:
                        task.metadata.update(metadata)

                    self.metrics['completed_tasks'] += 1
                    self._update_average_duration()

                    self._save_progress()
                    self._notify_callbacks('task_completed', {
                        'phase': phase.value,
                        'task_id': task_id,
                        'task_name': task.name,
                        'duration': task.duration_seconds
                    })

                    self.logger.info(f"✅ Completed task: {task.name} ({task.duration_seconds:.2f}s)")
                    break

    def fail_task(self, phase: InvestigationPhase, task_id: str, error: str):
        """Mark task as failed"""
        with self.lock:
            if phase not in self.phases:
                return

            for task in self.phases[phase].tasks:
                if task.task_id == task_id:
                    task.status = TaskStatus.FAILED
                    task.completed_at = datetime.now()
                    task.error_message = error

                    if task.started_at:
                        task.duration_seconds = (task.completed_at - task.started_at).total_seconds()

                    self.metrics['failed_tasks'] += 1

                    self._save_progress()
                    self._notify_callbacks('task_failed', {
                        'phase': phase.value,
                        'task_id': task_id,
                        'task_name': task.name,
                        'error': error
                    })

                    self.logger.error(f"❌ Failed task: {task.name} - {error}")
                    break

    def complete_phase(self, phase: InvestigationPhase):
        """Mark phase as completed"""
        with self.lock:
            if phase not in self.phases:
                return

            self.phases[phase].status = TaskStatus.COMPLETED
            self.phases[phase].completed_at = datetime.now()

            self._save_progress()
            self._notify_callbacks('phase_completed', {
                'phase': phase.value,
                'completed_at': self.phases[phase].completed_at.isoformat()
            })

            self.logger.info(f"🎉 Completed phase: {phase.value}")

    def complete_investigation(self):
        """Mark investigation as completed"""
        with self.lock:
            self.overall_status = TaskStatus.COMPLETED
            self.completed_at = datetime.now()

            if self.started_at:
                self.metrics['total_duration'] = (self.completed_at - self.started_at).total_seconds()

            self._save_progress()
            self._notify_callbacks('investigation_completed', {
                'investigation_id': self.investigation_id,
                'completed_at': self.completed_at.isoformat(),
                'total_duration': self.metrics['total_duration']
            })

            self.logger.info(f"🎊 Investigation completed: {self.investigation_id}")

    def fail_investigation(self, error: str):
        """Mark investigation as failed"""
        with self.lock:
            self.overall_status = TaskStatus.FAILED
            self.completed_at = datetime.now()

            self._save_progress()
            self._notify_callbacks('investigation_failed', {
                'investigation_id': self.investigation_id,
                'error': error
            })

            self.logger.error(f"💥 Investigation failed: {error}")

    def get_overall_progress(self) -> float:
        """Calculate overall investigation progress"""
        if not self.phases:
            return 0.0

        total_progress = sum(phase.calculate_progress() for phase in self.phases.values())
        return total_progress / len(self.phases)

    def get_current_status(self) -> Dict[str, Any]:
        """Get current investigation status"""
        return {
            'investigation_id': self.investigation_id,
            'overall_status': self.overall_status.value,
            'overall_progress': self.get_overall_progress(),
            'current_phase': self.current_phase.value if self.current_phase else None,
            'phases': {
                phase.value: phase_data.to_dict()
                for phase, phase_data in self.phases.items()
            },
            'metrics': self.metrics,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_seconds': self.metrics.get('total_duration', 0.0)
        }

    def register_callback(self, callback: Callable):
        """Register callback for progress events"""
        self.callbacks.append(callback)
        self.logger.debug(f"📞 Registered progress callback: {callback.__name__}")

    def _notify_callbacks(self, event_type: str, data: Dict):
        """Notify all registered callbacks"""
        for callback in self.callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                self.logger.error(f"Error in callback {callback.__name__}: {e}")

    def _update_average_duration(self):
        """Update average task duration metric"""
        if self.metrics['completed_tasks'] > 0:
            total_time = sum(
                task.duration_seconds
                for phase in self.phases.values()
                for task in phase.tasks
                if task.status == TaskStatus.COMPLETED
            )
            self.metrics['average_task_duration'] = total_time / self.metrics['completed_tasks']

    def _save_progress(self):
        """Save progress to JSON file"""
        try:
            progress_file = self.output_dir / "progress.json"
            with open(progress_file, 'w') as f:
                json.dump(self.get_current_status(), f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving progress: {e}")

    def export_timeline(self) -> List[Dict]:
        """Export timeline of all events"""
        timeline = []

        for phase, phase_data in self.phases.items():
            if phase_data.started_at:
                timeline.append({
                    'timestamp': phase_data.started_at.isoformat(),
                    'event': 'phase_started',
                    'phase': phase.value
                })

            for task in phase_data.tasks:
                if task.started_at:
                    timeline.append({
                        'timestamp': task.started_at.isoformat(),
                        'event': 'task_started',
                        'phase': phase.value,
                        'task': task.name
                    })

                if task.completed_at:
                    timeline.append({
                        'timestamp': task.completed_at.isoformat(),
                        'event': 'task_completed' if task.status == TaskStatus.COMPLETED else 'task_failed',
                        'phase': phase.value,
                        'task': task.name,
                        'duration': task.duration_seconds
                    })

        # Sort by timestamp
        timeline.sort(key=lambda x: x['timestamp'])
        return timeline


# Example usage and testing
if __name__ == "__main__":
    import time

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create progress monitor
    monitor = ProgressMonitor("test_investigation_001")

    # Register callback
    def progress_callback(event_type, data):
        print(f"📢 Event: {event_type} - {data}")

    monitor.register_callback(progress_callback)

    # Start investigation
    monitor.start_investigation()

    # Register phases
    monitor.register_phase(InvestigationPhase.WEB_SEARCH, [
        {'name': 'Google Search', 'description': 'Search Google for target'},
        {'name': 'Bing Search', 'description': 'Search Bing for target'}
    ])

    monitor.register_phase(InvestigationPhase.SOCIAL_MEDIA, [
        {'name': 'Facebook Scan', 'description': 'Scan Facebook profiles'},
        {'name': 'LinkedIn Scan', 'description': 'Scan LinkedIn profiles'}
    ])

    # Simulate web search phase
    monitor.start_phase(InvestigationPhase.WEB_SEARCH)

    monitor.start_task(InvestigationPhase.WEB_SEARCH, "web_search_0")
    time.sleep(1)
    monitor.update_task_progress(InvestigationPhase.WEB_SEARCH, "web_search_0", 50.0)
    time.sleep(1)
    monitor.complete_task(InvestigationPhase.WEB_SEARCH, "web_search_0",
                         metadata={'results_found': 42})

    monitor.start_task(InvestigationPhase.WEB_SEARCH, "web_search_1")
    time.sleep(1)
    monitor.complete_task(InvestigationPhase.WEB_SEARCH, "web_search_1",
                         metadata={'results_found': 38})

    monitor.complete_phase(InvestigationPhase.WEB_SEARCH)

    # Complete investigation
    monitor.complete_investigation()

    # Print final status
    print("\n📊 Final Status:")
    print(json.dumps(monitor.get_current_status(), indent=2))

    print("\n⏱️ Timeline:")
    for event in monitor.export_timeline():
        print(f"  {event['timestamp']}: {event['event']} - {event.get('task', event.get('phase'))}")

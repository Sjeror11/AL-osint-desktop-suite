#!/usr/bin/env python3
"""
🧪 Core Engine Phase 2 - Comprehensive Test Suite
OSINT Desktop Suite - Testing
LakyLuk Enhanced Edition - 4.10.2025

Tests:
- ProgressMonitor functionality
- InvestigationWorkflow execution
- Phase coordination
- Error handling
- Result aggregation
"""

import unittest
import asyncio
import sys
from pathlib import Path
import json
import shutil
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.progress_monitor import (
    ProgressMonitor,
    InvestigationPhase,
    TaskStatus,
    ProgressTask,
    PhaseProgress
)
from core.investigation_workflow import (
    InvestigationWorkflow,
    InvestigationConfig,
    InvestigationTarget,
    InvestigationType,
    InvestigationPriority
)


class TestProgressMonitor(unittest.TestCase):
    """Test ProgressMonitor functionality"""

    def setUp(self):
        """Setup test environment"""
        self.test_dir = Path("test_investigations")
        self.test_dir.mkdir(exist_ok=True)
        self.monitor = ProgressMonitor("test_inv_001", self.test_dir)

    def tearDown(self):
        """Cleanup test environment"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test monitor initialization"""
        self.assertEqual(self.monitor.investigation_id, "test_inv_001")
        self.assertEqual(self.monitor.overall_status, TaskStatus.PENDING)
        self.assertIsNone(self.monitor.started_at)

    def test_start_investigation(self):
        """Test investigation start"""
        self.monitor.start_investigation()

        self.assertEqual(self.monitor.overall_status, TaskStatus.IN_PROGRESS)
        self.assertIsNotNone(self.monitor.started_at)

    def test_phase_registration(self):
        """Test phase registration"""
        tasks = [
            {'name': 'Task 1', 'description': 'First task'},
            {'name': 'Task 2', 'description': 'Second task'}
        ]

        self.monitor.register_phase(InvestigationPhase.WEB_SEARCH, tasks)

        self.assertIn(InvestigationPhase.WEB_SEARCH, self.monitor.phases)
        self.assertEqual(len(self.monitor.phases[InvestigationPhase.WEB_SEARCH].tasks), 2)
        self.assertEqual(self.monitor.metrics['total_tasks'], 2)

    def test_task_lifecycle(self):
        """Test complete task lifecycle"""
        # Register phase
        tasks = [{'name': 'Test Task', 'description': 'A test task'}]
        self.monitor.register_phase(InvestigationPhase.WEB_SEARCH, tasks)

        # Start phase
        self.monitor.start_phase(InvestigationPhase.WEB_SEARCH)
        self.assertEqual(
            self.monitor.phases[InvestigationPhase.WEB_SEARCH].status,
            TaskStatus.IN_PROGRESS
        )

        # Start task
        task_id = "web_search_0"
        self.monitor.start_task(InvestigationPhase.WEB_SEARCH, task_id)

        task = self.monitor.phases[InvestigationPhase.WEB_SEARCH].tasks[0]
        self.assertEqual(task.status, TaskStatus.IN_PROGRESS)
        self.assertIsNotNone(task.started_at)

        # Update progress
        self.monitor.update_task_progress(
            InvestigationPhase.WEB_SEARCH,
            task_id,
            50.0,
            {'intermediate': 'data'}
        )
        self.assertEqual(task.progress_percent, 50.0)
        self.assertEqual(task.metadata['intermediate'], 'data')

        # Complete task
        self.monitor.complete_task(
            InvestigationPhase.WEB_SEARCH,
            task_id,
            {'final': 'result'}
        )

        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertEqual(task.progress_percent, 100.0)
        self.assertIsNotNone(task.completed_at)
        self.assertGreater(task.duration_seconds, 0)
        self.assertEqual(self.monitor.metrics['completed_tasks'], 1)

    def test_task_failure(self):
        """Test task failure handling"""
        tasks = [{'name': 'Failing Task', 'description': 'This will fail'}]
        self.monitor.register_phase(InvestigationPhase.WEB_SEARCH, tasks)
        self.monitor.start_phase(InvestigationPhase.WEB_SEARCH)
        self.monitor.start_task(InvestigationPhase.WEB_SEARCH, "web_search_0")

        # Fail task
        self.monitor.fail_task(
            InvestigationPhase.WEB_SEARCH,
            "web_search_0",
            "Test error message"
        )

        task = self.monitor.phases[InvestigationPhase.WEB_SEARCH].tasks[0]
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertEqual(task.error_message, "Test error message")
        self.assertEqual(self.monitor.metrics['failed_tasks'], 1)

    def test_overall_progress_calculation(self):
        """Test overall progress calculation"""
        # Register two phases with tasks
        self.monitor.register_phase(InvestigationPhase.WEB_SEARCH, [
            {'name': 'Task 1', 'description': 'Test'}
        ])
        self.monitor.register_phase(InvestigationPhase.SOCIAL_MEDIA, [
            {'name': 'Task 2', 'description': 'Test'}
        ])

        # Complete first phase
        self.monitor.start_phase(InvestigationPhase.WEB_SEARCH)
        self.monitor.start_task(InvestigationPhase.WEB_SEARCH, "web_search_0")
        self.monitor.complete_task(InvestigationPhase.WEB_SEARCH, "web_search_0")

        # Second phase not started
        progress = self.monitor.get_overall_progress()
        self.assertGreater(progress, 0)
        self.assertLess(progress, 100)

    def test_callback_notification(self):
        """Test callback notification system"""
        events_received = []

        def test_callback(event_type, data):
            events_received.append({'event': event_type, 'data': data})

        self.monitor.register_callback(test_callback)
        self.monitor.start_investigation()

        self.assertEqual(len(events_received), 1)
        self.assertEqual(events_received[0]['event'], 'investigation_started')

    def test_progress_persistence(self):
        """Test progress saving to file"""
        self.monitor.start_investigation()

        tasks = [{'name': 'Test', 'description': 'Test'}]
        self.monitor.register_phase(InvestigationPhase.WEB_SEARCH, tasks)

        # Check that progress file exists
        progress_file = self.test_dir / "test_inv_001" / "progress" / "progress.json"
        self.assertTrue(progress_file.exists())

        # Verify content
        with open(progress_file) as f:
            data = json.load(f)

        self.assertEqual(data['investigation_id'], "test_inv_001")
        self.assertEqual(data['overall_status'], 'in_progress')

    def test_timeline_export(self):
        """Test timeline export functionality"""
        self.monitor.start_investigation()

        tasks = [{'name': 'Task 1', 'description': 'Test'}]
        self.monitor.register_phase(InvestigationPhase.WEB_SEARCH, tasks)
        self.monitor.start_phase(InvestigationPhase.WEB_SEARCH)
        self.monitor.start_task(InvestigationPhase.WEB_SEARCH, "web_search_0")
        self.monitor.complete_task(InvestigationPhase.WEB_SEARCH, "web_search_0")

        timeline = self.monitor.export_timeline()

        self.assertGreater(len(timeline), 0)
        self.assertTrue(all('timestamp' in event for event in timeline))
        self.assertTrue(all('event' in event for event in timeline))


class TestInvestigationWorkflow(unittest.TestCase):
    """Test InvestigationWorkflow functionality"""

    def setUp(self):
        """Setup test environment"""
        self.test_dir = Path("test_workflow_investigations")
        self.test_dir.mkdir(exist_ok=True)

        # Create test target
        self.target = InvestigationTarget(
            name="Test Subject",
            target_type=InvestigationType.PERSON,
            location="Prague"
        )

    def tearDown(self):
        """Cleanup test environment"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_workflow_initialization(self):
        """Test workflow initialization"""
        config = InvestigationConfig(
            target=self.target,
            output_dir=self.test_dir,
            enable_ai_enhancement=False
        )

        workflow = InvestigationWorkflow(config)

        self.assertIsNotNone(workflow.investigation_id)
        self.assertEqual(workflow.config.target.name, "Test Subject")
        self.assertIsNotNone(workflow.progress)

    def test_minimal_workflow_execution(self):
        """Test minimal workflow execution"""
        config = InvestigationConfig(
            target=self.target,
            output_dir=self.test_dir,
            enabled_phases=[InvestigationPhase.INITIALIZATION],
            enable_ai_enhancement=False
        )

        workflow = InvestigationWorkflow(config)
        results = asyncio.run(workflow.execute())

        self.assertEqual(results.status, "completed")
        self.assertIn("initialization", results.phases_completed)
        self.assertEqual(len(results.errors), 0)

    def test_multi_phase_workflow(self):
        """Test multi-phase workflow execution"""
        config = InvestigationConfig(
            target=self.target,
            output_dir=self.test_dir,
            enabled_phases=[
                InvestigationPhase.INITIALIZATION,
                InvestigationPhase.WEB_SEARCH
            ],
            enable_ai_enhancement=False
        )

        workflow = InvestigationWorkflow(config)
        results = asyncio.run(workflow.execute())

        self.assertEqual(results.status, "completed")
        self.assertGreaterEqual(len(results.phases_completed), 1)

    def test_results_persistence(self):
        """Test that results are saved to file"""
        config = InvestigationConfig(
            target=self.target,
            output_dir=self.test_dir,
            enabled_phases=[InvestigationPhase.INITIALIZATION],
            enable_ai_enhancement=False
        )

        workflow = InvestigationWorkflow(config)
        results = asyncio.run(workflow.execute())

        # Check results file exists
        results_file = workflow.output_dir / "investigation_results.json"
        self.assertTrue(results_file.exists())

        # Verify content
        with open(results_file) as f:
            data = json.load(f)

        self.assertEqual(data['investigation_id'], workflow.investigation_id)
        self.assertEqual(data['status'], 'completed')

    def test_progress_tracking(self):
        """Test progress tracking during workflow"""
        config = InvestigationConfig(
            target=self.target,
            output_dir=self.test_dir,
            enabled_phases=[InvestigationPhase.INITIALIZATION],
            enable_ai_enhancement=False
        )

        workflow = InvestigationWorkflow(config)

        # Execute workflow
        asyncio.run(workflow.execute())

        # Get progress
        progress = workflow.get_progress()

        self.assertEqual(progress['overall_status'], 'completed')
        self.assertGreater(progress['overall_progress'], 0)
        self.assertGreater(progress['metrics']['completed_tasks'], 0)

    def test_error_handling(self):
        """Test error handling in workflow"""
        # This would test error scenarios
        # For now, just verify workflow handles missing tools gracefully
        config = InvestigationConfig(
            target=self.target,
            output_dir=self.test_dir,
            enabled_phases=[
                InvestigationPhase.INITIALIZATION,
                InvestigationPhase.GOVERNMENT_DATABASES  # Likely not available
            ],
            enable_ai_enhancement=False
        )

        workflow = InvestigationWorkflow(config)
        results = asyncio.run(workflow.execute())

        # Should complete even if some phases are skipped
        self.assertEqual(results.status, "completed")


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow"""

    def setUp(self):
        """Setup integration test environment"""
        self.test_dir = Path("test_integration")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        """Cleanup"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_end_to_end_investigation(self):
        """Test complete investigation from start to finish"""
        target = InvestigationTarget(
            name="Integration Test Target",
            target_type=InvestigationType.PERSON,
            location="Czech Republic",
            investigation_scope="comprehensive"
        )

        config = InvestigationConfig(
            target=target,
            output_dir=self.test_dir,
            enabled_phases=[
                InvestigationPhase.INITIALIZATION,
                InvestigationPhase.WEB_SEARCH,
                InvestigationPhase.ENTITY_CORRELATION
            ],
            enable_ai_enhancement=False,
            timeout_minutes=5
        )

        workflow = InvestigationWorkflow(config)
        results = asyncio.run(workflow.execute())

        # Verify results
        self.assertEqual(results.status, "completed")
        self.assertIsNotNone(results.investigation_id)
        self.assertGreater(results.duration_seconds, 0)
        self.assertGreaterEqual(len(results.phases_completed), 1)

        # Verify files created
        self.assertTrue((workflow.output_dir / "investigation_results.json").exists())
        self.assertTrue((workflow.output_dir / "progress" / "progress.json").exists())


def run_tests():
    """Run all tests"""
    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,  # Suppress INFO logs during testing
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestProgressMonitor))
    suite.addTests(loader.loadTestsFromTestCase(TestInvestigationWorkflow))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*80)
    print("🧪 CORE ENGINE PHASE 2 TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"✅ Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"💥 Errors: {len(result.errors)}")
    print(f"⏭️  Skipped: {len(result.skipped)}")
    print("="*80)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

"""
Deterministic Execution Order Management for SMPS.

Ensures that all operations execute in a deterministic order regardless of
how they are called, preventing execution order dependencies in results.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable, Set
from collections import defaultdict, deque
from enum import Enum

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExecutionPhase(Enum):
    """Phases of SMPS execution in dependency order."""

    # Data loading and preprocessing
    DATA_LOADING = "data_loading"
    DATA_VALIDATION = "data_validation"
    DATA_PREPROCESSING = "data_preprocessing"

    # Feature engineering
    FEATURE_ENGINEERING = "feature_engineering"
    FEATURE_SELECTION = "feature_selection"

    # Model training
    MODEL_TRAINING = "model_training"
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"

    # Validation and testing
    PHYSICS_VALIDATION = "physics_validation"
    CROSS_VALIDATION = "cross_validation"
    MODEL_EVALUATION = "model_evaluation"

    # Results and reporting
    RESULTS_AGGREGATION = "results_aggregation"
    REPORT_GENERATION = "report_generation"


@dataclass
class ExecutionTask:
    """Represents a single executable task."""

    name: str
    phase: ExecutionPhase
    function: Callable
    dependencies: List[str]  # Names of tasks this depends on
    inputs: List[str]  # Input data/artifacts required
    outputs: List[str]  # Output data/artifacts produced
    priority: int = 0  # Higher priority tasks run first within phase

    def __post_init__(self):
        if not self.dependencies:
            self.dependencies = []


class ExecutionOrderManager:
    """
    Manages deterministic execution order for all SMPS operations.

    Uses dependency graphs to ensure operations execute in the correct order
    regardless of how they are invoked.
    """

    def __init__(self):
        self.tasks: Dict[str, ExecutionTask] = {}
        if NETWORKX_AVAILABLE:
            self.execution_graph = nx.DiGraph()
        else:
            self.execution_graph = None  # Fallback mode
        self.completed_tasks: Set[str] = set()
        self.task_results: Dict[str, Any] = {}

    def register_task(self, task: ExecutionTask) -> None:
        """Register a task with the execution manager."""
        if task.name in self.tasks:
            logger.warning(
                f"Task '{task.name}' already registered, overwriting")

        self.tasks[task.name] = task

        if NETWORKX_AVAILABLE and self.execution_graph is not None:
            self.execution_graph.add_node(
                task.name, phase=task.phase, priority=task.priority)

            # Add dependency edges
            for dep in task.dependencies:
                self.execution_graph.add_edge(dep, task.name)

        logger.debug(
            f"Registered task: {task.name} in phase {task.phase.value}")

    def execute_deterministic(self, target_tasks: List[str] = None,
                              max_parallel: int = 1) -> Dict[str, Any]:
        """
        Execute tasks in deterministic order based on dependencies.

        Args:
            target_tasks: Specific tasks to execute (None = all tasks)
            max_parallel: Maximum parallel execution (currently not implemented)

        Returns:
            Dictionary of task results
        """
        # Determine which tasks to execute
        if target_tasks:
            tasks_to_execute = set(target_tasks)
        else:
            tasks_to_execute = set(self.tasks.keys())

        # Resolve dependencies and get execution order
        execution_order = self._resolve_execution_order(tasks_to_execute)

        logger.info(
            f"Executing {len(execution_order)} tasks in deterministic order")

        # Execute tasks in order
        results = {}
        for task_name in execution_order:
            if task_name not in tasks_to_execute:
                continue

            task = self.tasks[task_name]
            logger.info(f"Executing task: {task.name}")

            try:
                # Check if dependencies are satisfied
                if not self._check_dependencies_satisfied(task):
                    logger.error(
                        f"Dependencies not satisfied for task: {task.name}")
                    continue

                # Execute the task
                result = self._execute_task(task)
                results[task_name] = result
                self.task_results[task_name] = result
                self.completed_tasks.add(task_name)

                logger.info(f"Completed task: {task.name}")

            except Exception as e:
                logger.error(f"Failed to execute task {task.name}: {e}")
                # Continue with other tasks (don't fail completely)

        return results

    def get_execution_plan(self, target_tasks: List[str] = None) -> List[str]:
        """Get the planned execution order without executing."""
        if target_tasks:
            tasks_to_execute = set(target_tasks)
        else:
            tasks_to_execute = set(self.tasks.keys())

        return self._resolve_execution_order(tasks_to_execute)

    def get_task_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered tasks."""
        status = {}
        for name, task in self.tasks.items():
            status[name] = {
                'phase': task.phase.value,
                'completed': name in self.completed_tasks,
                'dependencies_satisfied': self._check_dependencies_satisfied(task),
                'has_result': name in self.task_results
            }
        return status

    def reset_execution_state(self) -> None:
        """Reset execution state (for re-running pipelines)."""
        self.completed_tasks.clear()
        self.task_results.clear()
        logger.info("Execution state reset")

    def _resolve_execution_order(self, target_tasks: Set[str]) -> List[str]:
        """Resolve the execution order based on dependencies."""
        if not NETWORKX_AVAILABLE or self.execution_graph is None:
            # Fallback to simple phase-based ordering
            return self._fallback_execution_order(target_tasks)

        # Create subgraph of target tasks and their dependencies
        relevant_tasks = set()
        to_visit = list(target_tasks)

        while to_visit:
            task_name = to_visit.pop()
            if task_name in relevant_tasks:
                continue

            relevant_tasks.add(task_name)
            if task_name in self.tasks:
                # Add dependencies
                to_visit.extend(self.tasks[task_name].dependencies)

        # Create subgraph
        subgraph = self.execution_graph.subgraph(relevant_tasks)

        # Topological sort with phase and priority ordering
        try:
            # Sort by phase order first, then by priority within phase
            phase_order = {phase: i for i, phase in enumerate(ExecutionPhase)}
            sorted_nodes = sorted(
                subgraph.nodes(data=True),
                key=lambda x: (phase_order[x[1]['phase']], -x[1]['priority'])
            )

            # Get topological order respecting dependencies
            ordered_tasks = []
            visited = set()

            def visit(node):
                if node in visited:
                    return
                visited.add(node)

                # Visit dependencies first
                for pred in subgraph.predecessors(node):
                    visit(pred)

                ordered_tasks.append(node)

            for node, _ in sorted_nodes:
                visit(node)

            return ordered_tasks

        except nx.NetworkXError as e:
            logger.error(f"Dependency resolution failed: {e}")
            # Fallback to simple phase-based ordering
            return self._fallback_execution_order(relevant_tasks)

    def _fallback_execution_order(self, tasks: Set[str]) -> List[str]:
        """Fallback execution order when topological sort fails."""
        # Group by phase
        phase_groups = defaultdict(list)
        for task_name in tasks:
            if task_name in self.tasks:
                phase_groups[self.tasks[task_name].phase].append(task_name)

        # Sort within each phase by priority
        ordered_tasks = []
        for phase in ExecutionPhase:
            if phase in phase_groups:
                phase_tasks = phase_groups[phase]
                # Sort by priority (higher first)
                phase_tasks.sort(
                    key=lambda x: self.tasks[x].priority, reverse=True)
                ordered_tasks.extend(phase_tasks)

        return ordered_tasks

    def _check_dependencies_satisfied(self, task: ExecutionTask) -> bool:
        """Check if all dependencies of a task are satisfied."""
        for dep in task.dependencies:
            if dep not in self.completed_tasks:
                return False
        return True

    def _execute_task(self, task: ExecutionTask) -> Any:
        """Execute a single task."""
        try:
            # Prepare inputs from previous task results
            inputs = {}
            for input_name in task.inputs:
                if input_name in self.task_results:
                    inputs[input_name] = self.task_results[input_name]

            # Execute the task function
            if inputs:
                result = task.function(**inputs)
            else:
                result = task.function()

            return result

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            raise


# Global execution order manager instance
execution_manager = ExecutionOrderManager()


def register_task(name: str, phase: ExecutionPhase, function: Callable,
                  dependencies: List[str] = None, inputs: List[str] = None,
                  outputs: List[str] = None, priority: int = 0) -> None:
    """Convenience function to register a task."""
    task = ExecutionTask(
        name=name,
        phase=phase,
        function=function,
        dependencies=dependencies or [],
        inputs=inputs or [],
        outputs=outputs or [],
        priority=priority
    )
    execution_manager.register_task(task)


def execute_pipeline(target_tasks: List[str] = None) -> Dict[str, Any]:
    """Convenience function to execute the pipeline."""
    return execution_manager.execute_deterministic(target_tasks)

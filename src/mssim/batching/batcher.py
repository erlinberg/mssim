from mssim.batching.parser import MSSIMParser
from mssim.runner import run_simulation
import logging
import math
import sys
import subprocess
import itertools
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class MSSIMBatcher:
    def __init__(self, config_path: str = "settings.json") -> None:
        self.config_path = config_path
        self.parser = MSSIMParser(config_path=config_path)

        # Identify parent condition keys
        condition_keys = {
            key for var in self.parser.active_variables for key in var.conditions.keys()
        }

        # Categorize variables for readability
        self.parent_vars = [
            v for v in self.parser.active_variables if v.name in condition_keys
        ]

        self.dependent_vars = [
            v for v in self.parser.active_variables if v.conditions]
        
        self.shared_vars = [
            v
            for v in self.parser.active_variables
            if v not in self.parent_vars and v not in self.dependent_vars
        ]

    def _generate_parent_states(self) -> Iterator[dict[str, Any]]:
        """Yields every possible combination of parent variable values as a dictionary."""
        if not self.parent_vars:
            yield {}  # Yield an empty state if no dependencies exist
            return

        keys = [v.name for v in self.parent_vars]
        values = [v.value for v in self.parent_vars]

        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))

    def _get_active_variables_for(self, parent_state: dict[str, Any]) -> list[Any]:
        """Returns all shared and dependent variables that are active in the current state."""
        active = list(self.shared_vars)
        for var in self.dependent_vars:
            if var.is_active_under(parent_state):
                active.append(var)
        return active

    def _decode_config(
        self,
        local_id: int,
        parent_state: dict[str, Any],
        active_vars: list[Any],
    ) -> dict[str, Any]:
        """Performs the decoding of the specific parameters based on the local_id in a segment."""
        config = parent_state.copy()
        remainder = local_id

        for var in reversed(active_vars):
            length = var.cardinality
            config[var.name] = var.value[remainder % length]
            remainder //= length

        return config

    def get_total_tasks(self) -> int:
        """Calculates the total number of correct batch configurations."""
        total = 0
        for parent_state in self._generate_parent_states():
            active_vars = self._get_active_variables_for(parent_state)
            total += math.prod(v.cardinality for v in active_vars)
        return total

    def get_batch_config_by_id(self, task_id: int) -> dict[str, Any]:
        """Given a global task_id, determines the corresponding batch configuration by iterating through the parent states and their active variable segments."""
        current_offset = 0

        for parent_state in self._generate_parent_states():
            active_vars = self._get_active_variables_for(parent_state)
            segment_size = math.prod(v.cardinality for v in active_vars)

            # Check if the target task_id falls inside this mathematical block
            if current_offset <= task_id < current_offset + segment_size:
                local_id = task_id - current_offset
                raw_config = self._decode_config(
                    local_id, parent_state, active_vars)
                # Nest kwargs if needed
                return self.parser.post_process(raw_config=raw_config, task_id=task_id)

            current_offset += segment_size

        raise ValueError(f"task_id {task_id} is out of range")


class MSSIMJobSubmitter:
    def __init__(self, config_path: str = "settings.json") -> None:
        self.batcher = MSSIMBatcher(config_path=config_path)

    def submit_parallel(self) -> None:
        """Submits batch jobs to SLURM using sbatch with an array job. Each task ID corresponds to a unique batch configuration."""
        total_tasks = self.batcher.get_total_tasks()

        if total_tasks == 0:
            logger.error("Zero batch configurations detected.")
            sys.exit(1)

        sbatch_command = [
            "sbatch",
            f"--array=0-{total_tasks - 1}",
            "run_SLURM_batch.sh",
            self.batcher.config_path,
        ]

        logger.info("Submitting %d batch task(s) to SLURM", total_tasks)
        try:
            result = subprocess.run(
                sbatch_command, capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError as exc:
            logger.error("SLURM submission failed: %s", exc.stderr.strip())
            sys.exit(1)

        if result.stdout.strip():
            logger.info(result.stdout.strip())

    def submit_sequential(self) -> None:
        """Runs batch jobs sequentially in the current process."""
        total_tasks = self.batcher.get_total_tasks()

        if total_tasks == 0:
            logger.error("Zero batch configurations detected.")
            sys.exit(1)

        for task_id in range(total_tasks):
            batch_cfg = self.batcher.get_batch_config_by_id(task_id)
            logger.info("Running Task ID %d with config: %s",
                        task_id, batch_cfg)
            run_simulation(batch_cfg)

    def run_task(self, task_id: int) -> None:
        """Executes the batch job corresponding to the given task ID."""
        batch_cfg = self.batcher.get_batch_config_by_id(task_id)
        logger.info("Running Task ID %d with config: %s",
                    task_id, batch_cfg)
        run_simulation(batch_cfg)

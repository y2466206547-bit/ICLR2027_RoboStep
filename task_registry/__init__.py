"""Access to the self-contained 75-task Frame-VLM reward registry."""

from .loader import find_task, list_tasks, load_program, load_task
from .reward_machine import TaskRewardMachine, load_reward_machine

__all__ = [
    "find_task",
    "list_tasks",
    "load_task",
    "load_program",
    "TaskRewardMachine",
    "load_reward_machine",
]

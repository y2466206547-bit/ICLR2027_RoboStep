"""Print one of the 75 package-local Frame-VLM reward machines."""

from __future__ import annotations

import argparse

from task_registry import load_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--benchmark")
    args = parser.parse_args()
    task = load_task(args.task, args.benchmark)
    print(f"{task['benchmark']} / {task['task_id']}  K={task['K']}  transitions={task['transitions']}")
    print(f"program source: {task['reward_program']['source']}")
    for index, stage in enumerate(task["reward_program"]["stages"]):
        terms = ", ".join(f"{term['name']}[{term['direction']}]" for term in stage["potential_terms"])
        print(f"  {index}: {stage['stage_id']}  {stage['semantic_objective']}  [{terms}]")


if __name__ == "__main__":
    main()

"""Static final-suite inventory and matched PPO defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskManifest:
    benchmark: str
    task_id: str
    horizon: int
    family: str
    stages: tuple[str, ...]
    task_type: str
    long_horizon: bool = False


METAWORLD_TASKS = (
    "assembly-v3", "basketball-v3", "bin-picking-v3", "box-close-v3",
    "button-press-topdown-v3", "button-press-topdown-wall-v3", "button-press-v3", "button-press-wall-v3",
    "coffee-button-v3", "coffee-pull-v3", "coffee-push-v3", "dial-turn-v3", "disassemble-v3",
    "door-close-v3", "door-lock-v3", "door-open-v3", "door-unlock-v3", "drawer-close-v3", "drawer-open-v3",
    "faucet-close-v3", "faucet-open-v3", "hammer-v3", "hand-insert-v3", "handle-press-side-v3",
    "handle-press-v3", "handle-pull-side-v3", "handle-pull-v3", "lever-pull-v3", "peg-insert-side-v3",
    "peg-unplug-side-v3", "pick-out-of-hole-v3", "pick-place-v3", "pick-place-wall-v3",
    "plate-slide-back-side-v3", "plate-slide-back-v3", "plate-slide-side-v3", "plate-slide-v3",
    "push-back-v3", "push-v3", "push-wall-v3", "reach-v3", "reach-wall-v3", "shelf-place-v3", "soccer-v3",
    "stick-pull-v3", "stick-push-v3", "sweep-into-v3", "sweep-v3", "window-close-v3", "window-open-v3",
)

MANISKILL_TASKS = (
    "PickCube-v1", "PushCube-v1", "PullCube-v1", "LiftPegUpright-v1", "PegInsertionSide-v1",
    "PlugCharger-v1", "PokeCube-v1", "PullCubeTool-v1", "PushT-v1", "RollBall-v1", "StackCube-v1",
    "StackPyramid-v1", "PickSingleYCB-v1", "PlaceSphere-v1", "FMBAssembly1Easy-v1", "RotateValveLevel4-v1",
    "TwoRobotStackCube-v1", "DrawTriangle-v1", "TriFingerRotateCubeLevel4-v1",
)

SOFTGYM_TASKS = ("ClothDrop", "ClothFlatten", "ClothFold", "RopeFlatten", "PassWater", "PourWater")


def _mw_stages(task: str) -> tuple[str, ...]:
    if any(token in task for token in ("button", "door", "dial", "drawer", "faucet", "lever", "window")):
        return ("approach_control", "contact_control", "verify_task_outcome")
    if any(token in task for token in ("assembly", "basketball", "bin-picking", "box-close", "coffee-pull", "disassemble", "pick-place", "shelf-place", "stick-pull")):
        return ("approach_object", "capture_object", "fixture_clearance", "transport_or_extract", "verify_task_outcome")
    if any(token in task for token in ("hammer", "hand-insert", "peg", "pick-out-of-hole", "plate-slide", "stick-push", "sweep")):
        return ("approach_object", "contact_object", "transport_or_extract", "verify_task_outcome")
    return ("approach_object", "contact_object", "verify_task_outcome")


def _ms_stages(task: str) -> tuple[str, ...]:
    known = {
        "StackPyramid-v1": ("grasp_red", "place_red_beside_green", "release_red", "grasp_blue", "align_blue", "stack_blue", "release_blue"),
        "FMBAssembly1Easy-v1": ("reach_bridge", "grasp_bridge", "reorient_with_fixture", "regrasp_bridge", "align_with_board", "place_bridge"),
        "PlugCharger-v1": ("reach_charger", "grasp_charger", "orient", "align_with_receptacle", "insert"),
    }
    if task in known:
        return known[task]
    if task in {"StackCube-v1", "TwoRobotStackCube-v1"}:
        return ("reach_assigned_cubes", "grasp_cube", "align_above_base", "place", "release")
    return ("reach_object", "establish_contact", "transport_and_settle")


def _type(task: str, benchmark: str) -> str:
    if benchmark == "SoftGym":
        return "Fluid" if task in {"PassWater", "PourWater"} else "Deform."
    if any(x in task.lower() for x in ("button", "door", "drawer", "faucet", "lever", "window", "valve", "dial")):
        return "Artic."
    if any(x in task.lower() for x in ("tool", "hammer", "poke", "draw", "stick", "sweep")):
        return "Tool"
    if any(x in task.lower() for x in ("insert", "plug", "triangle", "finger", "peg", "pyramid", "assembly", "push-t")):
        return "Precise"
    return "Rigid"


def final_suite() -> list[TaskManifest]:
    result: list[TaskManifest] = []
    for task in METAWORLD_TASKS:
        stages = _mw_stages(task)
        result.append(TaskManifest("Meta-World", task, 500 if len(stages) >= 5 else 200, "native_mt50", stages, _type(task, "Meta-World"), len(stages) >= 5))
    for task in MANISKILL_TASKS:
        stages = _ms_stages(task)
        horizon = 500 if task == "FMBAssembly1Easy-v1" else 250 if task == "StackPyramid-v1" else 300 if "Triangle" in task or "Valve" in task else 100 if "Tool" in task or task == "PushT-v1" else 50
        result.append(TaskManifest("ManiSkill", task, horizon, "maniskill19", stages, _type(task, "ManiSkill"), len(stages) >= 5))
    for task in SOFTGYM_TASKS:
        stages = ("approach_or_transport", "semantic_completion")
        result.append(TaskManifest("SoftGym", task, 100, "softgym6", stages, _type(task, "SoftGym"), False))
    return result


PPO_DEFAULTS: dict[str, dict[str, Any]] = {
    "Meta-World": {"actor_critic": "256-256-128 ELU", "rollout": "32x64", "epochs": 8, "minibatches": 8, "lr_schedule": "adaptive_kl", "gamma": 0.99, "gae_lambda": 0.95, "clip": 0.20},
    "ManiSkill": {"actor_critic": "3x256 Tanh", "rollout": "16x50", "epochs": 4, "minibatches": 32, "lr": 3e-4, "lr_schedule": "constant", "gamma": 0.80, "gae_lambda": 0.90, "clip": 0.20},
    "SoftGym": {"actor_critic": "256-256-128 ELU", "rollout": "256 steps/update", "epochs": 4, "minibatches": 4, "lr": 3e-4, "lr_schedule": "constant", "gamma": 0.99, "gae_lambda": 0.95, "clip": 0.20},
}


def task_index() -> dict[str, TaskManifest]:
    return {f"{item.benchmark}:{item.task_id}": item for item in final_suite()}

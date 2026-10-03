"""Fixed-budget Frame-VLM reward compilation/repair loop."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

from .schema import RewardProgram
from .vlm import CompileRequest, FrameVLMCompiler


@dataclass(frozen=True)
class CompilationRound:
    round_index: int
    task_id: str
    success_before_repair: float | None
    success_after_repair: float | None
    program: RewardProgram
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class CompilationResult:
    final_program: RewardProgram
    rounds: tuple[CompilationRound, ...]
    target_reached: bool


class RewardCompilationLoop:
    """Run the paper's bounded reward repair protocol outside PPO.

    `evaluate` is optional and should report validation diagnostics only.  The
    loop never inspects held-out test episodes to choose a reward program.
    After this function returns, the caller serializes `final_program` and
    freezes it before policy learning.
    """

    def __init__(self, compiler: FrameVLMCompiler, *, max_rounds: int = 5, target_success: float = 0.5) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        self.compiler = compiler
        self.max_rounds = int(max_rounds)
        self.target_success = float(target_success)

    def run(
        self,
        request: CompileRequest,
        *,
        evaluate: Callable[[RewardProgram, int], Mapping[str, Any]] | None = None,
    ) -> CompilationResult:
        previous: RewardProgram | None = None
        rounds: list[CompilationRound] = []
        reached = False
        before_score: float | None = None
        training_diagnostics: Mapping[str, Any] = {}
        gate_diagnostics: Mapping[str, Any] = {}
        for round_index in range(self.max_rounds):
            current_request = replace(
                request,
                previous_program=None if previous is None else previous.to_dict(),
                repair_round=round_index,
                training_diagnostics=training_diagnostics,
                gate_diagnostics=gate_diagnostics,
            )
            program = self.compiler.compile(current_request)
            diagnostics = dict(evaluate(program, round_index) if evaluate is not None else {})
            after_score = diagnostics.get("validation_success")
            after_score = None if after_score is None else float(after_score)
            rounds.append(CompilationRound(round_index, request.task_id, before_score, after_score, program, diagnostics))
            training_diagnostics = diagnostics.get("training_diagnostics", diagnostics)
            gate_diagnostics = diagnostics.get("gate_diagnostics", {})
            previous = program
            if after_score is not None and after_score >= self.target_success:
                reached = True
                break
            before_score = after_score
        if previous is None:  # defensive; max_rounds is checked above
            raise RuntimeError("reward compiler produced no program")
        return CompilationResult(previous, tuple(rounds), reached)

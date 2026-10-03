import tempfile
import time
import unittest

import numpy as np

from robostep.observations import ablation_config, pack_policy_observation
from robostep.reward import ActiveStageReward
from robostep.schema import RewardProgram, StageSpec
from robostep.vlm import FrozenAsyncVLM, GateRequest
from robostep.ppo import PPOConfig
from robostep.adapters import StageAwareRunner
from robostep.compilation import RewardCompilationLoop
from robostep.vlm import CompileRequest, FrameVLMCompiler


def program():
    return RewardProgram("test", "test", (StageSpec("a", "a", dwell_steps=2, transition_bonus=1.0), StageSpec("b", "b", dwell_steps=1, transition_bonus=2.0)))


class CoreTests(unittest.TestCase):
    def test_transition_uses_old_stage_dense_and_one_bonus(self):
        machine = ActiveStageReward(program(), batch_size=1, shaping_gamma=1.0, delta_max=10.0)
        machine.reset(np.array([[0.0, 0.0]]))
        first = machine.step(np.array([[0.4, 0.0]]), np.array([[True, False]]))
        self.assertFalse(first.transition[0])
        second = machine.step(np.array([[0.8, 0.0]]), np.array([[True, False]]))
        self.assertTrue(second.transition[0])
        self.assertEqual(second.event_stage[0], 0)
        self.assertAlmostEqual(float(second.transition_bonus[0]), 1.0)
        self.assertEqual(int(machine.stage[0]), 1)
        third = machine.step(np.array([[0.8, 0.3]]), np.array([[False, False]]))
        self.assertEqual(float(third.transition_bonus[0]), 0.0)

    def test_observation_ablation_dimensions(self):
        state = np.zeros((2, 4), dtype=np.float32)
        ours = ablation_config("ours", base_state_dim=4, stage_count=3)
        blind = ablation_config("actor_blind_stage", base_state_dim=4, stage_count=3)
        self.assertEqual(pack_policy_observation(state, [0, 3], ours).shape[1], 8)
        self.assertEqual(pack_policy_observation(state, [0, 3], blind).shape[1], 4)
        self.assertEqual(pack_policy_observation(state, [0, 3], blind, role="critic").shape[1], 8)

    def test_frozen_cache_abstains_on_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FrozenAsyncVLM(f"{directory}/cache.jsonl", call_backend=lambda _: "accept")
            request = GateRequest("id", "task", 1, 0, 2, "task", {})
            client.submit(request)
            client.flush()
            self.assertEqual(client.poll()[0][1], "accept")
            client.freeze()
            missing = GateRequest("new", "task", 1, 0, 3, "task", {})
            client.submit(missing)
            client.flush()
            self.assertEqual(client.poll()[0][1], "abstain")
            client.close()

            reloaded = FrozenAsyncVLM(f"{directory}/cache.jsonl")
            reloaded.submit(request)
            reloaded.flush()
            self.assertEqual(reloaded.poll()[0][1], "accept")
            reloaded.close()

    def test_benchmark_ppo_defaults_keep_three_maniskill_layers(self):
        self.assertEqual(PPOConfig.for_benchmark("ManiSkill").hidden_sizes, (256, 256, 256))

    def test_terminal_only_requires_official_success(self):
        machine = ActiveStageReward(program(), batch_size=1, shaping_gamma=1.0, delta_max=10.0)
        machine.reset(np.array([[0.0, 0.0]]))
        result = machine.step(np.array([[1.0, 1.0]]), np.array([[True, True]]), terminal_success=np.array([False]), reward_mode="terminal_only")
        self.assertEqual(float(result.reward[0]), 0.0)
        result = machine.step(np.array([[1.0, 1.0]]), np.array([[True, True]]), terminal_success=np.array([True]), reward_mode="terminal_only")
        self.assertEqual(float(result.reward[0]), 1.0)

    def test_nonreset_modes_keep_one_baseline_per_stage(self):
        machine = ActiveStageReward(program(), batch_size=1, shaping_gamma=1.0, delta_max=10.0)
        machine.reset(np.array([[0.0, 0.0]]))
        first = machine.step(
            np.array([[1.0, 0.0]]),
            np.array([[True, False]]),
            reward_mode="all_dense_with_stage_actor",
        )
        self.assertAlmostEqual(float(first.reward[0]), 1.0)
        transition = machine.step(
            np.array([[1.0, 0.5]]),
            np.array([[True, False]]),
            reward_mode="all_dense_with_stage_actor",
        )
        self.assertTrue(transition.transition[0])
        self.assertAlmostEqual(float(transition.reward[0]), 1.5)
        next_stage = machine.step(
            np.array([[1.0, 0.8]]),
            np.array([[False, False]]),
            reward_mode="all_dense_with_stage_actor",
        )
        self.assertAlmostEqual(float(next_stage.dense[0]), 0.3)

    def test_per_subtask_is_unit_transition_reward(self):
        machine = ActiveStageReward(program(), batch_size=1, shaping_gamma=1.0, delta_max=10.0)
        machine.reset(np.array([[0.0, 0.0]]))
        first = machine.step(
            np.array([[0.0, 0.0]]),
            np.array([[True, False]]),
            reward_mode="per_subtask",
        )
        self.assertFalse(first.transition[0])
        result = machine.step(
            np.array([[0.0, 0.0]]),
            np.array([[True, False]]),
            reward_mode="per_subtask",
        )
        self.assertTrue(result.transition[0])
        self.assertAlmostEqual(float(result.reward[0]), 1.0)

    def test_gate_response_must_match_request_step(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FrozenAsyncVLM(f"{directory}/cache.jsonl", call_backend=lambda _: "accept")
            request = GateRequest("id", "task", 1, 0, 2, "task", {})
            client.submit(request); client.flush()
            self.assertEqual(client.poll(current_episode=1, current_stage=0, current_request_step=3), [])
            client.close()

    def test_pending_vlm_cannot_fall_back_to_rule_transition(self):
        class Adapter:
            benchmark = "test"
            task_id = "test"
            def reset(self, seed=None): return np.zeros(2, dtype=np.float32)
            def step(self, action): return np.zeros(2, dtype=np.float32), {}
            def reward_inputs(self, observation, info): return np.array([1.0, 0.0]), np.array([True, False]), np.zeros(2), np.array(0.0)
            def official_success(self, observation, info): return False
            def close(self): pass
        with tempfile.TemporaryDirectory() as directory:
            from robostep.gates import GateVLM
            client = FrozenAsyncVLM(f"{directory}/cache.jsonl")
            runner = StageAwareRunner(Adapter(), program(), gate_vlm=GateVLM(client))
            runner.reset()
            _, _, _, diagnostics = runner.step(np.zeros(1))
            self.assertEqual(int(diagnostics["transition"][0]), 0)
            client.close()

    def test_reward_compilation_is_bounded_and_validation_only(self):
        seen = []
        def backend(payload):
            seen.append(payload)
            return program().to_dict()
        compiler = FrameVLMCompiler(backend)
        loop = RewardCompilationLoop(compiler, max_rounds=5, target_success=0.5)
        result = loop.run(CompileRequest("test", "test", {}), evaluate=lambda _, i: {"validation_success": 0.3 + 0.3 * i, "training_diagnostics": {"stagnation": bool(i == 0)}})
        self.assertEqual(len(result.rounds), 2)
        self.assertTrue(result.target_reached)
        self.assertTrue(seen[1]["training_diagnostics"]["stagnation"])


if __name__ == "__main__":
    unittest.main()

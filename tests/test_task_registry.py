import unittest

from task_registry import find_task, list_tasks, load_reward_machine, load_task


class TaskRegistryTests(unittest.TestCase):
    def test_final_suite_counts(self):
        self.assertEqual(len(list_tasks()), 75)
        self.assertEqual(len(list_tasks("MetaWorld")), 50)
        self.assertEqual(len(list_tasks("ManiSkill")), 19)
        self.assertEqual(len(list_tasks("SoftGym")), 6)

    def test_recipe_matches_transition_count(self):
        task = load_task("StackPyramid-v1", "ManiSkill")
        self.assertEqual(task["K"], 7)
        self.assertEqual(task["transitions"], 6)
        self.assertEqual(len(task["stages"]), task["K"])
        self.assertEqual(find_task("PassWater", "SoftGym")["K"], 2)

    def test_reward_machine_is_callable_from_package(self):
        machine = load_reward_machine("PassWater", "SoftGym")
        self.assertEqual(machine.stage_count, 2)
        machine.reset({"transport_progress": 0.2, "retained_water": 0.9, "target_closeness": 0.0, "spill_penalty": 0.0})
        result = machine.step(
            {"transport_progress": 0.6, "retained_water": 0.9, "target_closeness": 0.2, "spill_penalty": 0.0},
            candidates=[True, False],
        )
        self.assertEqual(result.reward.shape, (1,))


if __name__ == "__main__":
    unittest.main()

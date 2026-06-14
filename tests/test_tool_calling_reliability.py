from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_EXPERIMENTS_DIR = PROJECT_ROOT / "paper_experiments"
sys.path.insert(0, str(PAPER_EXPERIMENTS_DIR))

from tool_calling_reliability import selected_tool_names
from tool_calling_reliability_temperature_sweep import build_success_table
from tool_calling_reliability_model_sweep_ablated import (
    MAX_AGENT_TRIES as ABLATED_MAX_AGENT_TRIES,
    RETRY_ON_TOOL_FAILURE,
    SYSTEM_INSTRUCTIONS_ENABLED,
    build_ablated_messages,
    build_failure_table as build_ablated_failure_table,
)


class ToolCallingReliabilityTests(unittest.TestCase):
    def test_extracts_tool_names_from_dictionary_response(self):
        response = {
            "message": {
                "tool_calls": [
                    {"function": {"name": "solve_allocation", "arguments": {}}},
                    {"function": {"name": "other_tool", "arguments": {}}},
                ]
            }
        }

        self.assertEqual(selected_tool_names(response), ["solve_allocation", "other_tool"])

    def test_extracts_tool_names_from_typed_ollama_response_objects(self):
        response = SimpleNamespace(
            message=SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(
                        function=SimpleNamespace(name="solve_allocation", arguments={})
                    )
                ]
            )
        )

        self.assertEqual(selected_tool_names(response), ["solve_allocation"])

    def test_extracts_tool_names_from_mixed_mapping_and_object_response(self):
        response = {
            "message": SimpleNamespace(
                tool_calls=[
                    {"function": SimpleNamespace(name="solve_allocation", arguments={})}
                ]
            )
        }

        self.assertEqual(selected_tool_names(response), ["solve_allocation"])

    def test_returns_empty_list_when_no_tool_calls_are_present(self):
        self.assertEqual(selected_tool_names({"message": {}}), [])
        self.assertEqual(selected_tool_names(SimpleNamespace(message=SimpleNamespace())), [])

    def test_build_success_table_reports_success_proportions_by_temperature(self):
        results = {
            "0.0": {
                "focus_cost": [
                    {"correct_tool_called": True},
                    {"correct_tool_called": False},
                ],
                "focus_energy": [{"correct_tool_called": True}],
            },
            "0.25": {
                "focus_cost": [{"correct_tool_called": True}],
                "focus_energy": [
                    {"correct_tool_called": False},
                    {"correct_tool_called": False},
                ],
            },
        }

        table = build_success_table(results, [0.0, 0.25])

        self.assertEqual(table[0]["scenario"], "Cost")
        self.assertEqual(table[0]["0.0"], "0.5000")
        self.assertEqual(table[0]["0.25"], "1.0000")
        self.assertEqual(table[1]["scenario"], "Energy")
        self.assertEqual(table[1]["0.0"], "1.0000")
        self.assertEqual(table[1]["0.25"], "0.0000")

    def test_ablated_messages_contain_only_the_user_prompt(self):
        messages = build_ablated_messages("Optimise this workload")

        self.assertEqual(messages, [{"role": "user", "content": "Optimise this workload"}])
        self.assertFalse(any(message["role"] == "system" for message in messages))
        self.assertEqual(ABLATED_MAX_AGENT_TRIES, 1)
        self.assertFalse(SYSTEM_INSTRUCTIONS_ENABLED)
        self.assertFalse(RETRY_ON_TOOL_FAILURE)

    def test_ablated_failure_table_reports_failure_proportions_by_model(self):
        results = {
            "model-a": {
                "focus_cost": [
                    {"tool_call_failure": True},
                    {"tool_call_failure": False},
                ],
                "focus_energy": [{"tool_call_failure": True}],
            },
            "model-b": {
                "focus_cost": [{"tool_call_failure": False}],
                "focus_energy": [
                    {"tool_call_failure": False},
                    {"tool_call_failure": False},
                ],
            },
        }

        table = build_ablated_failure_table(results, ["model-a", "model-b"])

        self.assertEqual(table[0]["scenario"], "Cost")
        self.assertEqual(table[0]["model-a"], "0.5000")
        self.assertEqual(table[0]["model-b"], "0.0000")
        self.assertEqual(table[1]["scenario"], "Energy")
        self.assertEqual(table[1]["model-a"], "1.0000")
        self.assertEqual(table[1]["model-b"], "0.0000")


if __name__ == "__main__":
    unittest.main()

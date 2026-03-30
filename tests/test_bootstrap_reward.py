import os
import sys

import numpy as np


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPT_DIR = os.path.join(ROOT_DIR, "optimization")
if OPT_DIR not in sys.path:
    sys.path.insert(0, OPT_DIR)

import reward  # noqa: E402
from bootstrap_reward_utils import (  # noqa: E402
    EMPTY_LABEL,
    EXEC_ERROR_LABEL,
    TIMEOUT_LABEL,
    canonicalize_execution_output,
)


def test_canonicalize_execution_output_maps_special_labels():
    assert canonicalize_execution_output("Timeout Error") == TIMEOUT_LABEL
    assert canonicalize_execution_output("Execution Error: queue empty") == EXEC_ERROR_LABEL
    assert canonicalize_execution_output("error: bad output") == EXEC_ERROR_LABEL
    assert canonicalize_execution_output(" \n ") == EMPTY_LABEL
    assert canonicalize_execution_output("1  2\n3") == "1 2 3"


def test_build_novelty_info_marks_duplicate_empty_and_public_duplicates():
    info = reward.build_novelty_info(
        generated_inputs=["1\n", "1\n", "", "9\n"],
        generated_outputs=["2\n", "2\n", "", "10\n"],
        public_inputs=["9\n"],
        public_outputs=["10\n"],
        duplicate_public_penalty=0.5,
    )
    assert np.isclose(info["novelty"][0], 0.0)
    assert np.isclose(info["novelty"][1], 0.0)
    assert np.isclose(info["novelty"][2], 0.0)
    assert np.isclose(info["novelty"][3], 0.5)
    assert info["duplicate_mask"][0]
    assert info["empty_mask"][2]
    assert info["public_duplicate_mask"][3]


def test_analyze_problem_supports_no_ground_truth_path():
    problem = {
        "question": "Return a number.",
        "num_ground_truth_test": 0,
        "all_case_input": ["1\n", "2\n", ""],
        "all_case_output": ["1\n", "2\n", ""],
        "all_case_bool_table": [
            [True, True, False],
            [True, True, False],
            [False, False, False],
        ],
        "all_case_exe_results": [
            ["1\n", "2\n", "Execution Error: bad"],
            ["1\n", "2\n", "Timeout Error"],
            ["0\n", "3\n", ""],
        ],
        "public_example_bool_table": [[True], [True], [False]],
        "example_input": ["3\n"],
        "example_output": ["3\n"],
        "generated_code": ["print(1)", "print(1)", "print(0)"],
        "full_code_generation": ["```python\nprint(1)\n```"] * 3,
        "code_response_length": [12, 12, 12],
        "full_case_generation": [
            "**Test Input:**\n```1```\n**Test Output:**\n```1```",
            "**Test Input:**\n```2```\n**Test Output:**\n```2```",
            "**Test Input:**\n``````\n**Test Output:**\n``````",
        ],
        "case_response_length": [20, 20, 20],
        "code_generation_prompt": "code prompt",
        "case_generation_prompt": "case prompt",
    }
    analysis = reward.analyze_problem(problem)
    assert analysis["summary"]["generated_tests"] == 3
    assert analysis["summary"]["anchor_available"] is True
    assert analysis["code_group"] is not None
    assert analysis["case_group"] is not None
    assert analysis["code_corr"] is None
    assert analysis["case_corr"] is None

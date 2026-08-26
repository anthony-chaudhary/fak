#!/usr/bin/env python3
"""Process and result-contract tests for the Python sweep orchestrator."""

import sys
import tempfile
from pathlib import Path

import run_sweep as sweep


def test_run_command_merges_environment_and_honors_working_directory():
    with tempfile.TemporaryDirectory() as td:
        result = sweep.run_command(
            [
                sys.executable,
                "-c",
                "import os; print(os.environ['SWEEP_TEST']); print(os.getcwd())",
            ],
            cwd=Path(td),
            env={"SWEEP_TEST": "present"},
            timeout=10,
        )
        lines = result.stdout.strip().splitlines()
        assert result.returncode == 0
        assert lines[0] == "present"
        assert Path(lines[1]).resolve() == Path(td).resolve()


def test_api_model_returns_explicit_unimplemented_failure():
    model = sweep.ModelConfig(name="api-model", provider="example", enabled=True)
    workload = sweep.WorkloadConfig(max_turns=2, trials=1, timeout_s=10)
    result = sweep.run_api_model(model, workload, Path("out"), trial=3)
    assert result.model_name == "api-model" and result.trial == 3
    assert result.success is False
    assert result.error == "Not yet implemented"

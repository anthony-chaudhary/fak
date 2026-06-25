import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "shared_task_contract.py"


class SharedTaskContractCLITest(unittest.TestCase):
    def run_ok(self, *args: str) -> str:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stdout

    def test_validate_doc(self) -> None:
        out = self.run_ok("validate-doc", "docs/shared-task-record-contract.md")
        self.assertIn("OK", out)

    def test_validate_sequence(self) -> None:
        out = self.run_ok("validate-sequence", "examples/shared-task-record")
        self.assertIn("shared sequence", out)

    def test_validate_verdicts(self) -> None:
        out = self.run_ok("validate-verdicts", "examples/shared-task-record-verdicts")
        self.assertIn("collaboration verdicts", out)


if __name__ == "__main__":
    unittest.main()

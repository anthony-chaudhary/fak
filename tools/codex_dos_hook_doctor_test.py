"""Hermetic tests for tools/codex_dos_hook_doctor.py."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "codex_dos_hook_doctor.py"


def load():
    spec = importlib.util.spec_from_file_location("codex_dos_hook_doctor", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_manifest(home: Path) -> Path:
    root = home / "plugins" / "cache" / "dos" / "dos-kernel" / "0.28.0"
    launcher = root / "bin" / "dos-hook"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text("# launcher\n", encoding="utf-8")
    (launcher.parent / "dos-hook.ps1").write_text("# launcher\n", encoding="utf-8")
    manifest = root / "hooks" / "hooks.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "shell": "powershell",
                                    "command": "$py = Get-Command python; & $py.Source -m dos.cli hook pretool --workspace . --dialect codex; if ($LASTEXITCODE -ne 0) { exit 0 }",
                                }
                            ]
                        }
                    ],
                    "UserPromptSubmit": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "shell": "powershell",
                                    "command": "$py = Get-Command python; & $py.Source -m dos.cli hook marker --reset --workspace .; if ($LASTEXITCODE -ne 0) { exit 0 }",
                                }
                            ]
                        }
                    ],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


class CodexDosHookDoctorTest(unittest.TestCase):
    def test_dry_run_reports_replacements_without_command_bodies(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "codex-home"
            write_manifest(home)

            report = mod.build_report(home, apply=False, target_shell="bash")

            self.assertEqual(report["status"], "WARN")
            self.assertFalse(report["applied"])
            self.assertEqual(report["summary"]["replacements_available"], 2)
            self.assertEqual(report["summary"]["codex_replacements_available"], 1)
            self.assertEqual(report["summary"]["command_modes"], {"python_cli": 2})
            self.assertEqual(report["summary"]["codex_command_modes"], {"python_cli": 1})
            self.assertEqual(report["summary"]["projected_command_modes"], {"native_launcher": 2})
            self.assertEqual(report["summary"]["projected_codex_command_modes"], {"native_launcher": 1})
            self.assertEqual(report["summary"]["unrepairable_python_cli_hooks"], 0)
            self.assertEqual(report["summary"]["codex_unrepairable_python_cli_hooks"], 0)
            encoded = json.dumps(report)
            self.assertNotIn("-m dos.cli hook", encoded)
            self.assertNotIn(str(home), encoded)

    def test_apply_rewrites_manifest_and_creates_backup(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "codex-home"
            manifest = write_manifest(home)

            report = mod.build_report(home, apply=True, target_shell="bash")

            self.assertEqual(report["status"], "CHANGED")
            self.assertTrue(report["applied"])
            self.assertEqual(report["summary"]["applied_manifests"], 1)
            backup = manifest.with_name("hooks.json.before-native-dos-hook.bak")
            self.assertTrue(backup.exists())
            rewritten = json.loads(manifest.read_text(encoding="utf-8"))
            commands = [
                hook["command"]
                for entries in rewritten["hooks"].values()
                for entry in entries
                for hook in entry["hooks"]
            ]
            shells = [
                hook.get("shell")
                for entries in rewritten["hooks"].values()
                for entry in entries
                for hook in entry["hooks"]
            ]
            self.assertTrue(all("bin/dos-hook" in command for command in commands))
            self.assertNotIn("dos-hook.ps1", json.dumps(rewritten))
            self.assertEqual(shells, ["bash", "bash"])

            second = mod.build_report(home, apply=False, target_shell="bash")
            self.assertEqual(second["status"], "PASS")
            self.assertEqual(second["summary"]["command_modes"], {"native_launcher": 2})
            self.assertEqual(second["summary"]["codex_command_modes"], {"native_launcher": 1})
            self.assertEqual(second["summary"]["projected_command_modes"], {"native_launcher": 2})
            self.assertEqual(second["summary"]["projected_codex_command_modes"], {"native_launcher": 1})

    def test_apply_rewrites_powershell_native_launcher_to_bash_native(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "codex-home"
            manifest = write_manifest(home)
            data = json.loads(manifest.read_text(encoding="utf-8"))
            data["hooks"]["PreToolUse"][0]["hooks"][0]["command"] = (
                "$dosHook = 'C:\\Users\\USER\\.codex\\plugins\\cache\\dos\\dos-kernel\\0.28.0\\bin\\dos-hook.ps1'; "
                "if (Test-Path $dosHook) { & $dosHook pretool --workspace . --dialect codex; $dosExit = $LASTEXITCODE }"
            )
            manifest.write_text(json.dumps(data) + "\n", encoding="utf-8")

            report = mod.build_report(home, apply=False, target_shell="bash")

            self.assertEqual(report["status"], "WARN")
            self.assertEqual(report["summary"]["command_modes"], {"powershell_native_launcher": 1, "python_cli": 1})
            self.assertEqual(report["summary"]["projected_command_modes"], {"native_launcher": 2})

            applied = mod.build_report(home, apply=True, target_shell="bash")
            self.assertEqual(applied["status"], "CHANGED")
            rewritten = json.loads(manifest.read_text(encoding="utf-8"))
            hook = rewritten["hooks"]["PreToolUse"][0]["hooks"][0]
            self.assertEqual(hook["shell"], "bash")
            self.assertIn("bin/dos-hook", hook["command"])
            self.assertNotIn("dos-hook.ps1", hook["command"])

    def test_powershell_target_rewrites_bash_native_launcher_for_windows_codex(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "codex-home"
            manifest = write_manifest(home)
            mod.build_report(home, apply=True, target_shell="bash")

            dry = mod.build_report(home, apply=False, target_shell="powershell")
            self.assertEqual(dry["status"], "WARN")
            self.assertEqual(dry["target_command_mode"], "powershell_native_launcher")
            self.assertEqual(dry["summary"]["command_modes"], {"native_launcher": 2})
            self.assertEqual(dry["summary"]["codex_command_modes"], {"native_launcher": 1})
            self.assertEqual(dry["summary"]["projected_command_modes"], {"powershell_native_launcher": 2})
            self.assertEqual(dry["summary"]["projected_codex_command_modes"], {"powershell_native_launcher": 1})

            applied = mod.build_report(home, apply=True, target_shell="powershell")
            self.assertEqual(applied["status"], "CHANGED")
            rewritten = json.loads(manifest.read_text(encoding="utf-8"))
            commands = [
                hook["command"]
                for entries in rewritten["hooks"].values()
                for entry in entries
                for hook in entry["hooks"]
            ]
            shells = [
                hook.get("shell")
                for entries in rewritten["hooks"].values()
                for entry in entries
                for hook in entry["hooks"]
            ]
            self.assertEqual(shells, ["powershell", "powershell"])
            self.assertTrue(all("dos-hook.ps1" in command for command in commands))
            self.assertTrue(all("dos.cli hook" in command for command in commands))
            self.assertFalse(any("'2>/dev/null'" in command or '"2>/dev/null"' in command for command in commands))

            second = mod.build_report(home, apply=False, target_shell="powershell")
            self.assertEqual(second["status"], "PASS")
            self.assertEqual(second["summary"]["command_modes"], {"powershell_native_launcher": 2})
            self.assertEqual(second["summary"]["codex_command_modes"], {"powershell_native_launcher": 1})

    def test_powershell_target_repairs_quoted_redirect_argument(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "codex-home"
            manifest = write_manifest(home)
            data = json.loads(manifest.read_text(encoding="utf-8"))
            data["hooks"]["PreToolUse"][0]["hooks"][0]["shell"] = "powershell"
            data["hooks"]["PreToolUse"][0]["hooks"][0]["command"] = (
                "$dosHook = 'C:\\Users\\USER\\.codex\\plugins\\cache\\dos\\dos-kernel\\0.28.0\\bin\\dos-hook.ps1'; "
                "& $dosHook 'pretool' '--workspace' '.' '--dialect' 'codex' '2>/dev/null' 2>$null; exit 0"
            )
            manifest.write_text(json.dumps(data) + "\n", encoding="utf-8")

            dry = mod.build_report(home, apply=False, target_shell="powershell")
            self.assertEqual(dry["status"], "WARN")
            self.assertEqual(dry["summary"]["codex_replacements_available"], 1)

            applied = mod.build_report(home, apply=True, target_shell="powershell")
            self.assertEqual(applied["status"], "CHANGED")
            rewritten = json.loads(manifest.read_text(encoding="utf-8"))
            command = rewritten["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
            self.assertIn("dos-hook.ps1", command)
            self.assertNotIn("'2>/dev/null'", command)

    def test_unparseable_python_hook_stays_unrepairable(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "codex-home"
            manifest = write_manifest(home)
            data = json.loads(manifest.read_text(encoding="utf-8"))
            data["hooks"]["PreToolUse"][0]["hooks"][0]["command"] = (
                "$py = Get-Command python; & $py.Source -m dos.cli hook; # --dialect codex"
            )
            manifest.write_text(json.dumps(data) + "\n", encoding="utf-8")

            report = mod.build_report(home, apply=False, target_shell="bash")

            self.assertEqual(report["status"], "WARN")
            self.assertEqual(report["summary"]["replacements_available"], 1)
            self.assertEqual(report["summary"]["codex_replacements_available"], 0)
            self.assertEqual(report["summary"]["unrepairable_python_cli_hooks"], 1)
            self.assertEqual(report["summary"]["codex_unrepairable_python_cli_hooks"], 1)
            self.assertEqual(report["summary"]["projected_command_modes"], {"native_launcher": 1, "python_cli": 1})
            self.assertEqual(report["summary"]["projected_codex_command_modes"], {"python_cli": 1})

    def test_missing_manifest_is_unknown(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            report = mod.build_report(Path(td) / "codex-home", apply=False, target_shell="bash")
            self.assertEqual(report["status"], "UNKNOWN")
            self.assertEqual(report["manifest_count"], 0)


if __name__ == "__main__":
    unittest.main()

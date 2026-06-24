"""Hermetic tests for the vLLM adjudication-tax witness.

Pure stdlib (no pytest, no network beyond a loopback in-process server, no real
``fak serve``, no GPU). Runnable as ``python tools/vllm_tax_witness_test.py``.

Two in-process ThreadingHTTPServers stand in for the raw vLLM endpoint and the
fak gateway. Each serves /v1/models + /v1/chat/completions with a deterministic,
DIFFERING latency (a small sleep), so the gateway leg is reproducibly slower and
the latency tax is a real, non-zero delta -- exactly the by-design tax the
witness exists to measure. The witness is driven with --base-url at the "raw"
fake and --gateway-url at the slower "gateway" fake, mirroring how
glm52_serving_witness_test avoids spawning a real gateway.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import vllm_tax_witness as witness  # noqa: E402


def make_handler(sleep_s: float, decode_tps: float):
    """A fake OpenAI handler with a fixed per-request latency + decode_tps."""

    class FakeOpenAIHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            return

        def _write(self, status: int, payload: dict) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/v1/models":
                self._write(200, {"data": [{"id": "Qwen/Qwen2.5-7B-Instruct"}]})
                return
            self._write(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(length)
            if sleep_s > 0:
                time.sleep(sleep_s)
            self._write(200, {
                "choices": [{"message": {"role": "assistant", "content": "VLLM_TAX_OK"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 6, "total_tokens": 14},
                # An explicit decode tps so the decode-tps tax is deterministic too.
                "timings": {"predicted_per_second": decode_tps},
            })

    return FakeOpenAIHandler


def serve(test: unittest.TestCase, sleep_s: float, decode_tps: float) -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(sleep_s, decode_tps))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    test.addCleanup(server.server_close)
    test.addCleanup(server.shutdown)
    return int(server.server_port)


class DryRunTest(unittest.TestCase):
    def test_dry_run_planned_records_command_and_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "report.json"
            with redirect_stdout(StringIO()):
                code = witness.main([
                    "--base-url", "http://127.0.0.1:8000/v1",
                    "--model", "qwen",
                    "--dry-run",
                    "--engine-cache-engine", "vllm",
                    "--out", str(out),
                ])
            self.assertEqual(code, 0)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["vllm_adjudication_tax_witness"], "PLANNED")
            command = report["gateway_process"]["command"]
            self.assertEqual(command[:2], ["go", "run"])
            self.assertNotIn("-C", command)
            self.assertIn("--engine-cache-engine", command)
            self.assertIn("vllm", command)
            self.assertIn("raw_vllm", report["planned_endpoints"])
            self.assertIn("gateway", report["planned_endpoints"])
            # No traffic => no measured tax.
            self.assertIsNone(report["tax"]["latency_tax"])

    def test_dry_run_default_fak_command_is_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "report.json"
            with redirect_stdout(StringIO()):
                code = witness.main([
                    "--base-url", "http://127.0.0.1:8000/v1",
                    "--dry-run", "--out", str(out),
                ])
            self.assertEqual(code, 0)
            report = json.loads(out.read_text(encoding="utf-8"))
            command = report["gateway_process"]["command"]
            self.assertEqual(command[:2], ["go", "run"])
            self.assertNotIn("fak", command[:3])


class HeadToHeadTest(unittest.TestCase):
    def test_gateway_trails_raw_with_measured_tax(self) -> None:
        # Raw is fast (no sleep, high tps); gateway is slow (sleep, low tps).
        raw_port = serve(self, sleep_s=0.0, decode_tps=100.0)
        gw_port = serve(self, sleep_s=0.05, decode_tps=40.0)
        base = f"http://127.0.0.1:{raw_port}/v1"
        gateway = f"http://127.0.0.1:{gw_port}"
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "report.json"
            md = Path(td) / "report.md"
            bfile = Path(td) / "baseline.json"
            with redirect_stdout(StringIO()):
                code = witness.main([
                    "--base-url", base,
                    "--gateway-url", gateway,
                    "--model", "Qwen/Qwen2.5-7B-Instruct",
                    "--count", "3",
                    "--baseline-file", str(bfile),
                    "--record",
                    "--out", str(out),
                    "--markdown", str(md),
                ])
            # No baseline yet => KEEP (recorded) => exit 0.
            self.assertEqual(code, 0)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["vllm_adjudication_tax_witness"], "PASS")
            tax = report["tax"]
            # Gateway is reproducibly slower: latency tax > 1, decode-tps tax > 1.
            self.assertIsNotNone(tax["latency_tax"])
            self.assertGreater(tax["latency_tax"], 1.0)
            self.assertIsNotNone(tax["decode_tps_tax"])
            self.assertGreater(tax["decode_tps_tax"], 1.0)
            self.assertEqual(report["raw_vllm"]["ok_samples"], 3)
            self.assertEqual(report["gateway"]["ok_samples"], 3)
            self.assertEqual(report["gate"]["verdict"], witness.KEEP)
            # The baseline got written with the just-measured tax.
            doc = json.loads(bfile.read_text(encoding="utf-8"))
            key = "Qwen/Qwen2.5-7B-Instruct::vllm"
            self.assertIn(key, doc["baselines"])
            # Markdown carries the honest tax framing + the acceptance table.
            text = md.read_text(encoding="utf-8")
            self.assertIn("| check | status | detail |", text)
            self.assertIn("adjudication tax", text)

    def test_honest_framing_in_summary(self) -> None:
        raw_port = serve(self, sleep_s=0.0, decode_tps=100.0)
        gw_port = serve(self, sleep_s=0.02, decode_tps=50.0)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "report.json"
            bfile = Path(td) / "baseline.json"
            with redirect_stdout(StringIO()):
                witness.main([
                    "--base-url", f"http://127.0.0.1:{raw_port}/v1",
                    "--gateway-url", f"http://127.0.0.1:{gw_port}",
                    "--count", "2",
                    "--baseline-file", str(bfile),
                    "--out", str(out),
                ])
            report = json.loads(out.read_text(encoding="utf-8"))
            framing = report["summary"]["framing"].lower()
            self.assertIn("tax", framing)
            self.assertIn("trails", framing)
            self.assertIn("not raw tok/s", framing)
            self.assertIn("governance", framing)
            self.assertEqual(report["acceptance"]["honest_framing"]["status"], "PASS")


class GateVerdictTest(unittest.TestCase):
    """The KEEP / REVERT branches via a crafted baseline + measured tax."""

    def _gate(self, *, measured: float | None, baseline: float, tol: float, record: bool, bfile: Path):
        tax = {"latency_tax": measured}
        return witness.gate(
            tax,
            baseline_latency_tax=baseline,
            baseline_file=bfile,
            key="m::vllm",
            tolerance_pct=tol,
            model="m",
            record=record,
        )

    def test_keep_when_within_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bfile = Path(td) / "b.json"
            # baseline 1.30, tol 25% -> ceiling 1.625; measured 1.50 <= ceiling -> KEEP.
            g = self._gate(measured=1.50, baseline=1.30, tol=25.0, record=False, bfile=bfile)
            self.assertEqual(g["verdict"], witness.KEEP)
            self.assertTrue(g["ok"])
            self.assertEqual(witness.EXIT[g["verdict"]], 0)

    def test_revert_when_regressed_past_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bfile = Path(td) / "b.json"
            # baseline 1.30, tol 25% -> ceiling 1.625; measured 2.00 > ceiling -> REVERT.
            g = self._gate(measured=2.00, baseline=1.30, tol=25.0, record=False, bfile=bfile)
            self.assertEqual(g["verdict"], witness.REVERT)
            self.assertFalse(g["ok"])
            self.assertEqual(witness.EXIT[g["verdict"]], 3)
            self.assertGreater(g["regression_pct"], 25.0)

    def test_no_measurement_is_no_bench_not_silent_keep(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bfile = Path(td) / "b.json"
            g = self._gate(measured=None, baseline=1.30, tol=25.0, record=False, bfile=bfile)
            self.assertEqual(g["verdict"], witness.NO_BENCH)
            self.assertFalse(g["ok"])
            self.assertEqual(witness.EXIT[g["verdict"]], 4)

    def test_no_baseline_records_on_keep(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bfile = Path(td) / "b.json"
            g = witness.gate({"latency_tax": 1.42}, baseline_latency_tax=None,
                             baseline_file=bfile, key="m::vllm", tolerance_pct=25.0,
                             model="m", record=True)
            self.assertEqual(g["verdict"], witness.KEEP)
            self.assertTrue(g["baseline_recorded"])
            self.assertEqual(witness.load_baseline(bfile, "m::vllm"), 1.42)

    def test_no_baseline_without_record_does_not_write(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bfile = Path(td) / "b.json"
            g = witness.gate({"latency_tax": 1.42}, baseline_latency_tax=None,
                             baseline_file=bfile, key="m::vllm", tolerance_pct=25.0,
                             model="m", record=False)
            self.assertEqual(g["verdict"], witness.KEEP)
            self.assertFalse(g["baseline_recorded"])
            self.assertFalse(bfile.exists())


class ComputeTaxTest(unittest.TestCase):
    def test_delta_directions(self) -> None:
        raw = {"median_latency_s": 0.10, "median_decode_tps": 100.0}
        gateway = {"median_latency_s": 0.15, "median_decode_tps": 40.0}
        tax = witness.compute_tax(raw, gateway)
        # latency tax = gateway/raw = 1.5 (fak slower).
        self.assertAlmostEqual(tax["latency_tax"], 1.5, places=4)
        # decode tps tax = raw/gateway = 2.5 (fak slower).
        self.assertAlmostEqual(tax["decode_tps_tax"], 2.5, places=4)

    def test_missing_legs_yield_none(self) -> None:
        tax = witness.compute_tax({}, {})
        self.assertIsNone(tax["latency_tax"])
        self.assertIsNone(tax["decode_tps_tax"])


if __name__ == "__main__":
    unittest.main()

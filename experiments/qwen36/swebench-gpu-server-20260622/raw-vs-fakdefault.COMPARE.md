# SWE-bench Verified — fak-gateway vs raw-SGLang (overall completion)

**Model:** `Qwen/Qwen3.6-27B` served as `qwen36-27b` (SGLang TP=8, bf16) · **Selection:** astropy__astropy-12907 · **Host:** dgx-a100.example.lab

Both arms drive the identical `mini-swe-agent` against the SAME SGLang weights; the only difference is whether each tool turn is routed through the `fak serve` adjudication gateway (`:8080`) or hits raw SGLang (`:30000`) directly. *Overall completion* = the agent ran to the end and emitted a non-empty patch; *resolved* = the official harness (`swebench.harness.run_evaluation`) PASS_TO_PASS + FAIL_TO_PASS grade.

| arm | endpoint | instances | completed | patch bytes | agent time | resolved | resolve% | grade time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw-sglang | `http://127.0.0.1:30000/v1` | 1 | 1 | 504 | 222.8s | 1/500 | 0.2% | 255.8s |
| fak-gateway | `http://127.0.0.1:8080/v1` | 1 | 0 | 0 | 410.0s | 0/500 | 0.0% | 128.5s |

## Verdict
- **raw-SGLang:** completed 1/1, resolved 1/500.
- **fak-gateway:** completed 0/1, resolved 0/500.
- Overall completion through the fak gateway DIFFERS from raw SGLang on this selection — routing every tool turn through fak's adjudication plane did change the resolve outcome.

# Prompt-Vault harness and Qwen3.8 fleet comparison

**Verdict:** The frozen Prompt-Vault Color Palette task now has a deterministic 12-check browser fixture. The unchanged fixture accepts both captured strong-model artifacts at 12/12 and rejects empty (2/12) and static-but-wrong (6/12) pages. The two accepted arms remain unranked because fak native and Ultracode still lack artifacts and complete telemetry.

## Value frame

- **For:** engineers choosing a coding harness and deciding whether to spend one strong-model session or a local Qwen3.8 fleet.
- **Problem:** existing model-serving probes and orchestration plans do not prove that a harness can turn the same engineering brief into an accepted artifact.
- **Today:** Prompt-Vault provides concrete standalone application briefs, while fak has exact Qwen3.8 serving witnesses and Ultracode plan/launch evidence on separate surfaces.
- **Better because:** this contract freezes one prompt, one acceptance boundary, every model/harness setting, and the cost/outcome fields needed for an honest comparison.
- **Witness:** [`matrix.json`](prompt-vault/matrix.json), the pinned upstream revision, the [Codex](prompt-vault/witnesses/codex.json) and [Claude](prompt-vault/witnesses/claude.json) grade receipts, their captured renders, and the existing Qwen3.8/Ultracode witnesses linked below.

Problem centrality is **Core**. It directly tests integrated agent operation. It also checks managed context (same frozen prompt), net-true efficiency (accepted result plus tokens/cost), bounded adaptation (declared settings), and integrated operations (artifact plus terminal receipt).

## What Prompt-Vault contributes

Observed at 2026-08-21T01:06:57Z at [`w512/Prompt-Vault@19f2d492`](https://github.com/w512/Prompt-Vault/tree/19f2d492c985559ac4aedd98348581ac55d0d4d9); the pinned commit event is 2026-08-19T17:18:26Z:

- 14 structured application briefs across Easy, Medium, Hard, and Advanced directories;
- explicit UI, behavior, persistence, and technology constraints suitable for end-to-end artifact work;
- no executable grader, reference artifacts, releases, issues, or pull requests at the pinned revision;
- no declared repository license, so fak records paths, hashes, and provenance instead of copying prompt bodies.

The source was checked as a pinned shipped tree across its task content, repository history, releases/tags, issues, pull requests, and license/provenance surfaces. With no declared license and no upstream grader, the disposition is **INSPIRE-ONLY** for source reuse and **RECIPE** for fak's independently authored browser grader. The pin has no refresh trigger; a different upstream revision is a new corpus version, not silent drift in this one.

This makes Prompt-Vault a **task source**, not benchmark authority. A passing result requires a fak-owned acceptance fixture that directly tests the selected brief. Visual quality requires captured renders; file existence or an agent's final message is not a pass.

The spine selects `Easy/Color_Palette_Generator.md` (`sha256:4e095b57…a439d`) because it is a bounded single-file task with observable requirements: five equal swatches, random generation, per-swatch locking, hex labels, copy feedback, contrast-aware controls, and Space-key regeneration. Expanding to harder tasks comes only after this task has a complete four-arm run.

## Comparison contract

[`matrix.json`](prompt-vault/matrix.json) is the machine-readable contract. Every arm receives the exact pinned task bytes plus only this neutral execution suffix:

> Work only in the current directory. Implement the specification as index.html. Do not read parent directories or repository instructions. Do not explain; finish the artifact and then report the files created.

The arms are:

| Arm | Frozen identity and settings | Current evidence |
|---|---|---|
| fak native baseline | `qwen38:27b`; Q4_K_M; native in-kernel Metal; 4,096-token context; session state off | Qwen text/JSON/tool acceptance exists, but there is no native coding tool loop that can produce `index.html`; task result pending. |
| Codex | `codex-cli@0.148.0`; `gpt-5.6-sol`; `xhigh`; ephemeral isolated directory | Produced `index.html` in 93,631 ms. Receipt: 38,003 input tokens, 0 cached input tokens, 3,513 output tokens, exit 0; billed cost, tool calls, and retries remain `UNKNOWN`. Artifact SHA-256 `061e44d4…1a7d9`. **PASS 12/12:** [grade](prompt-vault/witnesses/codex.json) · [desktop render](prompt-vault/witnesses/codex.png). |
| Claude Code | `claude-code@2.1.237`; `claude-opus-5`; high effort; no session persistence; isolated directory | Produced `index.html` in 56,051 ms. Receipt: 51,147 input tokens, 2,396 output tokens, $0.315635, two turns; tool calls and retries remain `UNKNOWN`. Artifact SHA-256 `154bf52b…eebc4`. **PASS 12/12:** [grade](prompt-vault/witnesses/claude.json) · [desktop render](prompt-vault/witnesses/claude.png). |
| fak Ultracode fleet | `profile=ultracode`; four workers; `ultra` route; implementer/verifier/reviewer roles | Resolver and worker-launch evidence exist, but no owned executor/reconciliation receipt can run and accept the task; result pending. |

Do not rank the arms until all four have `terminal_verdict`, the same acceptance fixture, and complete token/cost fields. Harness overhead is part of the result: the Claude run loaded its configured harness context, and the reported 51,147 input tokens must not be silently subtracted.

## Existing proof points this joins

- [`docs/supported/qwen38-27b.md`](../supported/qwen38-27b.md) pins the Qwen3.8 artifacts and launch identities.
- [`docs/_witnesses/qwen38-27b-2026-08-19/`](../_witnesses/qwen38-27b-2026-08-19/) proves Qwen3.8-27B FP8 TP2 and CUDA GGUF text/JSON/tool acceptance.
- [`docs/_witnesses/qwen38-27b-2026-08-20/`](../_witnesses/qwen38-27b-2026-08-20/) records the exact native-Metal baseline settings represented in `matrix.json`: text/JSON/tool acceptance passed on `internal/model@r448+g8145dc0bea` and `internal/agent@r320+g6bb3c3dd55`.
- [`docs/notes/ULTRACODE-NATIVE-RUNTIME-AUDIT-2026-08-20.md`](../notes/ULTRACODE-NATIVE-RUNTIME-AUDIT-2026-08-20.md) proves Ultracode's current boundary: deterministic plans and real worker launch, but no accepted workflow outcome.

The comparison therefore separates two questions that are often conflated:

1. **Can Qwen3.8 serve correct text/JSON/tool outputs on the declared backend?** Existing model acceptance says yes for the witnessed configurations.
2. **Can the harness turn a Prompt-Vault brief into an independently accepted engineering artifact, and at what total cost?** Not yet proven for any comparable four-arm matrix.

## Acceptance before score

[Issue #8323](https://github.com/anthony-chaudhary/fak/issues/8323) added [`color-palette.spec.js`](prompt-vault/color-palette.spec.js), the exact Playwright lock, and [`verify-color-palette.ps1`](prompt-vault/verify-color-palette.ps1). The runner seeds palette generation, serves every input from the same loopback origin, uses Chrome at 1440×900, and applies the same 12 checks without artifact-specific selectors. Run it from the repository root:

```powershell
npm ci --prefix docs/benchmarks/prompt-vault
.\docs\benchmarks\prompt-vault\verify-color-palette.ps1 `
  -CodexArtifact <captured-codex-index.html> `
  -ClaudeArtifact <captured-claude-index.html>
```

The committed negative receipts prove an [empty page fails at 2/12](prompt-vault/witnesses/fail-empty.json) and a [static five-swatch impostor fails at 6/12](prompt-vault/witnesses/fail-wrong.json). The captured artifacts are deliberately not copied into the public tree because the pinned source has no declared license; their full SHA-256 values bind the reports to the previously captured bytes. Every known result field listed by `required_measurements` is present in `matrix.json`; unavailable telemetry is `UNKNOWN`, never zero.

Run order:

1. ~~Freeze the acceptance fixture and prove it fails on an empty/wrong page.~~ Complete: 2/12 and 6/12 FAIL receipts; Codex and Claude each PASS 12/12.
2. Run the single Qwen3.8 native baseline through the owned coding executor.
3. Run Codex and Claude once each in fresh isolated directories.
4. Run the Qwen3.8 Ultracode profile with the same prompt and acceptance target.
5. Grade the remaining artifacts with the same fixture, capture renders, and publish accepted outcome, wall time, total tokens, cache tokens, spend, retries, and witness acceptance.

Until the native and Ultracode rows exist, the honest result is **two arms accepted; four-arm head-to-head result pending**.



## Qwen3.8 one-L4 paired verdict (2026-08-21)

The exact Qwen3.8 checkpoint now reaches readiness on one NVIDIA L4 when the streamed loader runs with `GOMEMLIMIT=20GiB`; that is a hardware-placement result, not a quality win. The normalized single arm produced a valid artifact but passed only 2/12 checks. The evaluator-guided micro-context arm exhausted its 1,536-token completion budget before closing the forced `Write` call, so it produced no gradeable artifact.

`fak ultracode bench --pair witnesses/qwen38-paired-run.json --json` returned **`ABSTAIN`**: `acceptance outcome is not equal and passing`. Therefore this run does not beat either accepted 12/12 baseline and does not prove Ultracode gain. The next bounded device-residency work is tracked in [#8377](https://github.com/anthony-chaudhary/fak/issues/8377).

Evidence: [`qwen38-single.json`](prompt-vault/witnesses/qwen38-single.json), [`qwen38-single.png`](prompt-vault/witnesses/qwen38-single.png), and [`qwen38-paired-run.json`](prompt-vault/witnesses/qwen38-paired-run.json).
# Prompt-Vault harness and Qwen3.8 fleet comparison

**Verdict:** Prompt-Vault is now a pinned external task source for a same-task comparison of fak native, Codex, Claude Code, and fak Ultracode. The first Claude arm produced an artifact, but no arm has a comparable score yet. Fak native and Ultracode still lack the coding executor required to run this task, and Prompt-Vault supplies specifications—not a grader.

## Value frame

- **For:** engineers choosing a coding harness and deciding whether to spend one strong-model session or a local Qwen3.8 fleet.
- **Problem:** existing model-serving probes and orchestration plans do not prove that a harness can turn the same engineering brief into an accepted artifact.
- **Today:** Prompt-Vault provides concrete standalone application briefs, while fak has exact Qwen3.8 serving witnesses and Ultracode plan/launch evidence on separate surfaces.
- **Better because:** this contract freezes one prompt, one acceptance boundary, every model/harness setting, and the cost/outcome fields needed for an honest comparison.
- **Witness:** [`matrix.json`](prompt-vault/matrix.json), the pinned upstream revision, the captured Claude receipt below, and the existing Qwen3.8/Ultracode witnesses linked below.

Problem centrality is **Core**. It directly tests integrated agent operation. It also checks managed context (same frozen prompt), net-true efficiency (accepted result plus tokens/cost), bounded adaptation (declared settings), and integrated operations (artifact plus terminal receipt).

## What Prompt-Vault contributes

Studied on 2026-08-20 at [`w512/Prompt-Vault@19f2d492`](https://github.com/w512/Prompt-Vault/tree/19f2d492c985559ac4aedd98348581ac55d0d4d9):

- 14 structured application briefs across Easy, Medium, Hard, and Advanced directories;
- explicit UI, behavior, persistence, and technology constraints suitable for end-to-end artifact work;
- no executable grader, reference artifacts, releases, issues, or pull requests at the pinned revision;
- no declared repository license, so fak records paths, hashes, and provenance instead of copying prompt bodies.

This makes Prompt-Vault a **task source**, not benchmark authority. A passing result requires a fak-owned acceptance fixture that directly tests the selected brief. Visual quality requires captured renders; file existence or an agent's final message is not a pass.

The spine selects `Easy/Color_Palette_Generator.md` (`sha256:4e095b57…a439d`) because it is a bounded single-file task with observable requirements: five equal swatches, random generation, per-swatch locking, hex labels, copy feedback, contrast-aware controls, and Space-key regeneration. Expanding to harder tasks comes only after this task has a complete four-arm run.

## Comparison contract

[`matrix.json`](prompt-vault/matrix.json) is the machine-readable contract. Every arm receives the exact pinned task bytes plus only this neutral execution suffix:

> Work only in the current directory. Implement the specification as index.html. Do not read parent directories or repository instructions. Do not explain; finish the artifact and then report the files created.

The arms are:

| Arm | Frozen identity and settings | Current evidence |
|---|---|---|
| fak native baseline | `qwen38:27b`; Q4_K_M; native in-kernel Metal; 4,096-token context; session state off | Qwen text/JSON/tool acceptance exists, but there is no native coding tool loop that can produce `index.html`; task result pending. |
| Codex | `codex-cli@0.148.0`; `gpt-5.6-sol`; `xhigh`; ephemeral isolated directory | Produced `index.html` in 93,631 ms. Receipt: 38,003 input tokens, 0 cached input tokens, 3,513 output tokens, exit 0. Artifact SHA-256 `061e44d4…1a7d9`. **Ungraded:** this is throughput evidence, not quality proof. |
| Claude Code | `claude-code@2.1.237`; `claude-opus-5`; high effort; no session persistence; isolated directory | Produced `index.html` in 56,051 ms. Receipt: 51,147 input tokens, 2,396 output tokens, $0.315635, two turns. Artifact SHA-256 `154bf52b…eebc4`. **Ungraded:** this is throughput evidence, not quality proof. |
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

[Issue #8323](https://github.com/anthony-chaudhary/fak/issues/8323) tracks the deterministic browser fixture that the next run must add that checks every explicit Color Palette requirement and captures a desktop render. It then executes unchanged against every arm's `index.html`. The result row must include all fields listed by `required_measurements` in `matrix.json`; missing cost or cache telemetry is `UNKNOWN`, never zero.

Run order:

1. Freeze the acceptance fixture and prove it fails on an empty/wrong page.
2. Run the single Qwen3.8 native baseline through the owned coding executor.
3. Run Codex and Claude once each in fresh isolated directories.
4. Run the Qwen3.8 Ultracode profile with the same prompt and acceptance target.
5. Grade all artifacts with the same fixture, capture renders, and publish accepted outcome, wall time, total tokens, cache tokens, spend, retries, and witness acceptance.

Until those rows exist, the honest result is **comparison spine shipped; head-to-head result pending**.




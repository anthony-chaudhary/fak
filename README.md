<p align="center">
  <picture><source media="(prefers-color-scheme: dark)" srcset="visuals/brand/fak-logo.svg"><img src="visuals/brand/fak-logo-ink.svg" alt="fak logo" width="320"></picture>
</p>

# fak — configure your agents for the task at hand

> Try the [native inference goal](docs/native-inference-goal.md): fak-native is the product and performance path.
> It is intended to beat llama.cpp in matched, quality-constrained envelopes. llama.cpp remains
> an explicit reference for benchmarks and diagnosis. It also supports interoperability and
> borrowing, but never acts as a silent fallback.

AI agents can do more work than one fixed set of prompts, tools, permissions, and model settings
can handle well. fak gives each task a small configuration. One boundary then manages context,
models, tools, and the record of what happened.

## Start here

Wrap the agent you already use with one command. fak forwards your existing subscription
credential and applies a default-deny policy floor, with no separate API key:

```bash
fak guard -- codex  # -> launches Codex behind the policy floor
```

For the quickest proof, run the offline demo below. For security details, inspect the
default-deny policy. For performance, follow the witnessed native-model results.
Claims and limitations link to the evidence behind them.

<!-- native-status: 2026-08-26 -->
### Native-model status — 2026-08-26

This compact readout is designed for frequent refreshes; every row links its authority and says what is *not* proven.

| Lane | Latest witnessed result | Current hold |
|---|---|---|
<!-- qwen38-frontdoor:begin -->
| [Qwen3.8-27B](docs/benchmarks/QWEN-PERFORMANCE-INDEX.md) | Accepted fak-native Metal Q4_K_M: **2.3-2.9 decode tok/s**, with functional `PASS` in the frozen M3 Pro full-run envelope. ([accepted receipt](docs/_witnesses/qwen38-27b-2026-08-20/metal-native-run-summary.json)) Closest near-matched observation: **3.3 vs 6.966061 tok/s (~47%)** on the same M3 Pro and artifact. This is approximate, not accepted parity: native used P31/T64 versus P32/T64 and no joint quality-complete receipt exists. ([#8697](https://github.com/anthony-chaudhary/fak/issues/8697)) | Separate A100 cache-restore diagnostic: **~0.2 tok/s with 0/5 exact**. Failed quality makes it diagnostic only, never the parity headline. ([cache attribution](docs/_witnesses/issue-8819-qwen38-cache-attribution/README.md)) |
<!-- qwen38-frontdoor:end -->
| Ultracode / microagents | A `qwen2.5:0.5b` small-model scout/writer campaign preserved the accepted output through width 8 while reading 13,126 scoped tokens vs 32,760 in the full-context counterfactual. This is a context-access result, not a Qwen3.8 or billed-cost claim. ([access frontier](docs/_witnesses/issue-8624-ultracode-smallmodel/README.md)) | ABSTAIN on the GPT-5.6 Ultracode pair: activation and billed-token/spend accounting were not independently verified, and the observed fleet run was slower (0.47× concurrency speedup). ([paired witness](docs/_witnesses/issue-8168-ultracode-live/README.md)) |

Refresh contract: update the dated marker, both rows, linked committed witnesses, and each row's explicit hold; then run `python tools/readme_freshness_audit.py --json`.

<!-- project-status: 2026-08-25 -->
### Project status — 2026-08-25

These labels are strict for outside readers. Shipped means merged and checkable. In flight
links to open work. Goal describes direction, not a promise. Limitation names what the
current evidence does not support.

- Shipped: agent queues now have a crash-safe desired-state reconciliation spine
  ([#8875](https://github.com/anthony-chaudhary/fak/issues/8875),
  [#8876](https://github.com/anthony-chaudhary/fak/issues/8876)); self-update now avoids
  partial in-place binary replacement ([#8865](https://github.com/anthony-chaudhary/fak/issues/8865));
  and installed launch paths are visible through one command-front-door matrix
  ([#5808](https://github.com/anthony-chaudhary/fak/issues/5808)).
- In flight: the queue is being extended with guarded worker launch and landing
  ([#8889](https://github.com/anthony-chaudhary/fak/issues/8889)), status and saturation
  metrics ([#8890](https://github.com/anthony-chaudhary/fak/issues/8890)), and operator
  controls ([#8899](https://github.com/anthony-chaudhary/fak/issues/8899)). Native Qwen3.8
  performance work continues in [#8923](https://github.com/anthony-chaudhary/fak/issues/8923).
- Goal: make fak the simplest boundary for long-lived agent work. Configure the job once.
  Then let the kernel manage context, models, and tools. It also manages queues and evidence.
  Improve fak-native inference toward matched quality and speed. See
  [the problems fak solves](docs/problems-we-solve.md) and
  [the native-inference goal](docs/native-inference-goal.md).
- Limitation: the queue work above is not yet the complete operator product, and the
  native Qwen3.8 and Ultracode results remain inside the holds in the table above. For a
  subsystem-by-subsystem account of real, simulated, and planned behavior, use
  [STATUS.md](STATUS.md) and [the feature matrix](docs/supported/features.md).

Refresh contract: reconcile shipped entries with merged commits or closed issues. Confirm
that in-flight links are still open, keep goals non-promissory, and state current limitations.
Update the marker, then run `python tools/readme_freshness_audit.py --json`.

## Run your agents

Install fak, then keep using the Codex login and models you already have:

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh

# Any host with Go 1.26+
go install github.com/anthony-chaudhary/fak/cmd/fak@latest

# See the shipped configuration levels (no key or model call required)
fak agent profiles
```

For a small coding change, ask Ponytail to push toward the smallest correct implementation and
Caveman to keep the answer compact:

```bash
fak manage --output-profile caveman:medium --work-profile ponytail:high -- codex \
  "Remove the duplicate cache without adding a dependency."
```

For a risky investigation, use the same Codex harness with lighter simplicity pressure and room
for a fuller explanation:

```bash
fak manage --output-profile caveman:low --work-profile ponytail:low -- codex \
  "Trace the intermittent checkout failure and explain the evidence."
```

That is the idea: your agent, configured for this task. Ponytail controls how strongly the
agent resists avoidable code and machinery. Caveman controls how compactly it reports back. Neither profile removes explicit requirements or safety checks. Tests, diagnostics, and
evidence also remain in place.

`fak manage -- codex` uses the balanced `ponytail:medium` + `caveman:medium` defaults. Change
either axis per run, or use `standard` / `full` to turn that axis off. `fak guard` remains the
compatible name for `fak manage`.

### Next, make the harness yours

When a per-run configuration is not enough, generate a small harness whose identity,
instructions, and tools you own:

```bash
fak harness init --dir ./my-agent --module example.com/my-agent
cd ./my-agent
go run ./cmd/microharnessdemo --selfcheck
go run ./cmd/microharnessdemo
```

The generated product uses fak's managed boundary without making you assemble an agent loop from
scratch. Start with the [harness guide](docs/harness-init.md) when you are ready to customize it.

## More details

| If you want to… | Start here |
|---|---|
| Choose Ponytail and Caveman levels | [Work profiles](docs/work-profiles.md) · [response profiles](docs/response-profiles.md) |
| Inspect or change console settings | [Settings quickstart](docs/cli-reference.md#console-settings) |
| Connect another agent or model | [Codex](docs/integrations/openai-codex.md) · [Claude Code](docs/integrations/claude.md) · [all integrations](docs/integrations/) |
| Understand what fak manages | [Architecture](docs/architecture.md) · [capability map](docs/CAPABILITIES.md) |
| Improve or compare local inference | [Fak-native inference doctrine](docs/native-inference-goal.md) · [performance routes](docs/performance.md) |
| Check the proof and limits | [Claims](CLAIMS.md) · [benchmark authority](BENCHMARK-AUTHORITY.md) · [security](SECURITY.md) |
| Build on fak | [Go API](pkg/) · [harness contract](docs/harness-kit-contract.md) · [contributing](CONTRIBUTING.md) |
| Explore everything else | [Documentation index](docs/index.md) · [interactive showcase](docs/showcase.html) · [front-page archive](docs/README-legacy.md) |

Apache-2.0 licensed.

<!-- readme-verified: 2026-08-26 vs VERSION 0.45.0 + BENCHMARK-AUTHORITY · appeal-verified: 2026-08-26 99.2/100 · process: tools/readme_freshness_audit.py + tools/doc_appeal_scorecard.py -->

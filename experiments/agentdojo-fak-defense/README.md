# `fak_gateway` — fak's structural defense for the AgentDojo harness (#1064)

This directory packages fak's strongest WITNESSED asset — its **default-deny
tool-call admission gate** (capability floor + IFC source-stamp / sink-gate +
result quarantine) — as a *registered, non-model* defense for the upstream
[`ethz-spylab/agentdojo`](https://github.com/ethz-spylab/agentdojo) harness, so the
structural prompt-injection floor is measured on a comparison surface fak does
**not** author.

It is the external-entry counterpart that the now-closed
[#869](https://github.com/anthony-chaudhary/fak/issues/869) (local, model-free
floor) deliberately excluded — see `BENCHMARK-AUTHORITY.md`: *"LOCAL structural
floor only — not an official external AgentDojo leaderboard result or raw-model
arm."*

## Files

| File | What it is |
|---|---|
| `fak_gateway_defense.py` | The defense: `FakGatewayDefense(BasePipelineElement)` + `FakInjectionDetector(PromptInjectionDetector)`, a faithful port of `internal/ifc/ifc.go`. Imports the real AgentDojo base classes inside a fork; shims them otherwise so the core is unit-testable with no install. |
| `test_fak_gateway_defense.py` | The unit test (acceptance criterion 1): proves the module **loads** and **intercepts a tool call**. `python3 test_fak_gateway_defense.py` → 26 checks, exit 0. |

## The lane (honest seat)

fak's honest seat is a **tool-call admission gate** — a capability floor — which is
a *different class* from the four non-model defense rows AgentDojo publishes for
`gpt-4o-2024-05-13`:

| Defense | class | util | util-under-attack | targeted ASR |
|---|---|---:|---:|---:|
| Tool Filter | tool-set transform | 72.16 | 56.28 | 6.84 |
| Spotlighting-with-Delimiting | content transform | 72.16 | 55.64 | 41.65 |
| Transformers PI Detector | content classifier | 41.24 | 21.14 | 7.95 |
| Repeat User Prompt | prompt transform | 84.54 | 67.25 | 27.82 |
| **`fak_gateway` (this module)** | **tool-call admission gate (IFC)** | _OBSERVED, needs key_ | _OBSERVED, needs key_ | **WITNESSED ~0** |

The ceiling tier is the **formal-isolation** defenses (CaMeL drives ASR to 0; MELON
0.0–2.4%). fak's structural floor is **co-equal** with that tier — a **PLACE in the
~0-ASR tier at a measured utility cost**, never a "win", never a leaderboard rank.

## Mechanism (why the ASR column is WITNESSED)

A faithful port of the two seams in `internal/ifc/ifc.go`:

1. **Source-stamp (data plane).** Every tool *result* is stamped by provenance. A
   read of untrusted external content — the channel an AgentDojo
   `important_instructions` injection arrives through — raises a per-conversation
   taint high-water mark. It never blocks; it only annotates.
2. **Sink-gate (control plane).** A tool *call* to a sensitive sink (external
   egress, code-exec, irreversible mutation) is **refused** when tainted data is in
   flight. It defers on every non-sink / untainted call, so it only ever *adds*
   restriction.

Detection keys on *content* (evadable by paraphrase). IFC keys on *provenance* (a
paraphrase cannot launder it), so the harmful sink is barred regardless of phrasing
— the property that holds the local battery at ASR 0.000. Classification order is
security-load-bearing: an external destination in the args is egress even for a
SafeSink-named tool (the `transfer_to_human_agents`-launder bypass) and even under
an unlisted arg key (the `{"server": "attacker..."}` bypass).

Local intercept witness (reproducible, no model / GPU / network):

```bash
python3 fak_gateway_defense.py --json
# targeted ASR (full-stack) = 0.000  (0/7 attacks land their sink)
# benign completion         = 1.000  (2/2 benign controls complete)
# gate = PASS
```

## Wiring into a fork of `ethz-spylab/agentdojo`

Two paths. Both place `FakGatewayDefense` in the `ToolsExecutionLoop` **before** the
`ToolsExecutor` so a denied sink never executes.

### Path A — `--module-to-load` (no core fork edit)

```bash
# from the agentdojo fork root, with this directory importable on PYTHONPATH:
python -m agentdojo.scripts.benchmark \
  --module-to-load experiments.agentdojo_fak_defense.fak_gateway_defense \
  --defense fak_gateway \
  --model gpt-4o-2024-05-13 \
  --attack important_instructions \
  -s workspace -s slack -s travel -s banking
```

On import the module's `register()` appends `fak_gateway` to the `DEFENSES`
constant. (Depending on the harness pin you may also wire the
`from_config` branch as in Path B — the registration hook is intentionally small
and self-documenting.)

### Path B — direct fork edit (one branch in `from_config`)

In `src/agentdojo/agent_pipeline/agent_pipeline.py`:

```python
from fak_gateway_defense import make_pipeline_elements  # vendored into the fork

DEFENSES = [..., "fak_gateway"]

# inside AgentPipeline.from_config(...), where the tools loop is built:
if config.defense == "fak_gateway":
    tools_loop = ToolsExecutionLoop(
        [*make_pipeline_elements(), ToolsExecutor(), <model>]
    )
```

## Running the real entry (the model-gated arms)

Per <https://agentdojo.spylab.ai/results/> this is a **PR into a fork**, *not* a
JSON-drop leaderboard ("NOT a leaderboard"). The full cross-product:

- **Security suite — 629 cases** = Workspace 240 + Slack 105 + Travel 140 +
  Banking 144 (sum over the four suites of `user_tasks × injection_tasks`), under a
  named attack (canonical: `important_instructions`).
- **Utility suite — 97 user tasks** (benign + under-attack).

```bash
# security cross-product (targeted ASR + utility-under-attack)
python -m agentdojo.scripts.benchmark --model gpt-4o-2024-05-13 \
  --defense fak_gateway --attack important_instructions \
  -s workspace -s slack -s travel -s banking
# benign utility arm (no attack)
python -m agentdojo.scripts.benchmark --model gpt-4o-2024-05-13 --defense fak_gateway
# results table
python util_scripts/create_results_table.py
```

**Cost (sourced paper estimate):** the 629-case security run on GPT-4o ≈ **US$35**;
the 97-case utility arm ≈ **US$4**. The ASR/refusal arm is model-light; the utility
arms require the paid fronted model.

## Honesty fence (the trap this entry must not cross)

1. **ASR and utility are different provenance.** `targeted ASR` is fak-authored
   (WITNESSED). `benign utility` / `utility-under-attack` are the **fronted model's**
   capability (OBSERVED, relayed). A tool-admission gate that refuses calls
   **depresses benign utility** — report fak's utility drop honestly; never post
   ASR≈0 while hiding what it cost.
2. **The internal 0/38 is not a 629 number.** `go run ./cmd/agentdojoredteam` is a
   fixed, fak-authored 38-case corpus. The only honest *public* claim is a fresh
   629-case run submitted upstream.
3. **The ceiling is PLACE, never WIN.** CaMeL/MELON already sit at ~0 ASR via formal
   isolation; fak's structural floor is co-equal with that tier, not superior.
4. **It is not a leaderboard.** Call it a published defense *row*, cited with the
   fronted-model caveat — not a rank or a "win".

## Status

- **Built + WITNESSED here:** the defense module, its `--module-to-load` /
  `from_config` wiring, and the unit test proving it loads and intercepts a tool
  call (the load-bearing #1064 work). The `targeted ASR` mechanism is reproducible
  locally with no key.
- **Operator-gated (the remaining step):** the paid 629-case + 97-case run behind a
  real fronted model, and opening the PR into a fork of `ethz-spylab/agentdojo`
  (shares the `OPENAI_BASE_URL` seam with [#873](https://github.com/anthony-chaudhary/fak/issues/873)).
  Tracked as the recorded blocker in the entry artifact under
  `experiments/agent-live/agentdojo-fak-gateway-defense-entry-20260627.json`.

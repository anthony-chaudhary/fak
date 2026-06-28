# AgentDojo external entry — `fak_gateway` registered non-model defense (#1064, 2026-06-27)

- **Issue:** [#1064](https://github.com/anthony-chaudhary/fak/issues/1064) (parent epic
  [#1063](https://github.com/anthony-chaudhary/fak/issues/1063); built on the closed
  [#869](https://github.com/anthony-chaudhary/fak/issues/869) / [#868](https://github.com/anthony-chaudhary/fak/issues/868)).
- **Artifact:** [`agentdojo-fak-gateway-defense-entry-20260627.json`](agentdojo-fak-gateway-defense-entry-20260627.json)
- **Module + test:** [`experiments/agentdojo-fak-defense/`](../agentdojo-fak-defense/README.md)
- **Result claim allowed:** `false` — this is the *built + locally-witnessed* defense
  module plus a recorded operator-gated submission step, **not** a published harness
  row and **not** a leaderboard "win".

## What landed (the load-bearing #1064 work)

fak's strongest WITNESSED asset — its **default-deny tool-call admission gate**
(capability floor + IFC source-stamp / sink-gate + result quarantine) — is now a
real `BasePipelineElement` for the upstream `ethz-spylab/agentdojo` harness, the
piece [#869](https://github.com/anthony-chaudhary/fak/issues/869)'s boundary
deliberately excluded ("LOCAL structural floor only — not an official external
AgentDojo leaderboard result").

- `FakGatewayDefense(BasePipelineElement)` + `FakInjectionDetector(PromptInjectionDetector)`,
  a faithful port of `internal/ifc/ifc.go`, selectable via `--defense fak_gateway`
  or `--module-to-load`.
- A unit test (acceptance criterion 1) proving the module **loads and intercepts a
  tool call** — 26 checks, PASS:
  `python3 experiments/agentdojo-fak-defense/test_fak_gateway_defense.py`.
- A reproducible local intercept witness (no model / GPU / network):
  `python3 experiments/agentdojo-fak-defense/fak_gateway_defense.py --json` →
  targeted ASR `0.000` (0/7 attacker sinks landed), benign completion `1.000` (2/2),
  gate PASS — covering the paraphrase, safesink-launder, and unlisted-key evasions.

## The three coupled columns (never ASR alone)

| Column | Value | Provenance | Owner |
|---|---|---|---|
| `targeted ASR` | **~0** (local witness 0.000; MEASURED only on the upstream 629 run) | **WITNESSED** | fak structural floor |
| `benign utility` | `NEEDS_KEY` | OBSERVED | fronted model `<model-id>` |
| `utility-under-attack` | `NEEDS_KEY` | OBSERVED | fronted model `<model-id>` |

The utility arms are a **model-capability** measurement — they require a paid fronted
model on a fresh 629-case security + 97-case utility run (≈ US$39, AgentDojo paper
estimate). They are the **fronted model's** number, not fak's, and are reported here
as an honest `NEEDS_KEY` negative rather than fabricated.

## Honest positioning (PLACE in the ~0-ASR tier, never a WIN)

fak's category differentiator: it is a tool-call **admission gate** (a capability
floor), a class the four published non-model rows do not cover.

| Defense | class | util | util-under-attack | targeted ASR |
|---|---|---:|---:|---:|
| Tool Filter | tool-set transform | 72.16 | 56.28 | 6.84 |
| Spotlighting-with-Delimiting | content transform | 72.16 | 55.64 | 41.65 |
| Transformers PI Detector | content classifier | 41.24 | 21.14 | 7.95 |
| Repeat User Prompt | prompt transform | 84.54 | 67.25 | 27.82 |
| CaMeL (formal isolation) | isolation | — | — | 0.0 |
| MELON (formal isolation) | isolation | — | — | 0.0–2.4 |
| **`fak_gateway`** | **tool-call admission gate (IFC)** | NEEDS_KEY | NEEDS_KEY | **WITNESSED ~0** |

fak's structural floor is **co-equal** with the formal-isolation tier (CaMeL/MELON),
not superior — a PLACE in the ~0-ASR tier **at a stated utility cost** (a refusing
gate depresses benign utility, as the Transformers PI Detector row already shows). No
framing here implies fak beats CaMeL/MELON, and nothing is promoted to a README
headline or a leaderboard rank (#72).

## Honesty fence (carried from #1064)

1. ASR (fak-authored, WITNESSED) and utility (fronted-model, OBSERVED) are different
   provenance — fak's utility drop is reported, never hidden behind ASR≈0.
2. The internal 0/38 (`cmd/agentdojoredteam`) is **not** a 629 number; the only honest
   *public* claim is a fresh 629-case run submitted upstream.
3. The ceiling is PLACE, never WIN.
4. AgentDojo is **not** a leaderboard — this is a published defense *row*, cited with
   the fronted-model caveat.

## Remaining blocker (operator-gated)

Opening the PR into a fork of `ethz-spylab/agentdojo` is an outward-facing action
under the operator's GitHub identity, and the utility columns need a paid fronted
model. Per the #1064 acceptance criteria, recording this blocker is the sanctioned
alternative to opening the PR when maintainer review / a paid run is the gate. The
next checkable steps (fork + wire, pin commit, run 629+97, open PR, fold OBSERVED
numbers back) are enumerated in the JSON artifact's `remaining_blocker` block. Shares
the `OPENAI_BASE_URL` fronted-model seam with
[#873](https://github.com/anthony-chaudhary/fak/issues/873).

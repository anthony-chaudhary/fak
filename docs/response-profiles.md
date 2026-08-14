# Response profiles: concise, Caveman-compatible, and composable

Response profiles are an **opt-in presentation control** for fak's owned agent loop. They let a user ask for shorter answers without changing tool authorization, work scope, tests, diagnostics, or evidence requirements.

## Start here

List the supported selections and their status:

```bash
fak agent profiles
fak agent profiles --json
```

Run with the recommended Caveman-compatible balance:

```bash
fak agent --output-style caveman:medium --task "Explain this failure and propose the smallest safe fix"
```

Disable all response shaping:

```bash
fak agent --output-style full
```

`full` is the default. Nothing enables itself.

## Choosing an intensity

| Selection | Use it when | What changes |
|---|---|---|
| `caveman:low` | You want normal explanations with less filler | Trims restatement and connective prose |
| `caveman:medium` | You want concise, directly actionable answers | Removes preamble/recap and keeps only needed explanation |
| `caveman:high` | You want highly compressed operational output | Essential content only; shortest safe response shape |
| `native:low|medium|high` | You want the same intensity scale without the Caveman compatibility label | Uses fak's native profile family |
| `full` | You want no steering, or need maximum narrative detail | Adds no response-profile segment |

The friendly `caveman:medium` spelling canonicalizes to `caveman:native:medium`. Captures keep the implementation slot so users can distinguish fak's independently authored behavior from a future attributed upstream adapter.

## Native versus original

- **Native** means fak-authored profile text, enforced through fak's cache-stable system-prompt MMU and preservation contract. It is shipped now.
- **Original** means exact or adapted upstream-authored profile material at a pinned revision, with attribution and a content digest. `caveman:original:*` is reserved but deliberately returns an error until that adapter passes its provenance and safety gates ([#6706](https://github.com/anthony-chaudhary/fak/issues/6706)).

fak never silently substitutes native behavior when a user asks for `original`.

## What a response profile cannot change

A response profile cannot remove or weaken:

- policy decisions, authorization checks, or safety warnings;
- commands, code, identifiers, paths, errors, diagnostics, or requested evidence;
- uncertainty and provenance qualifiers;
- explicit user formatting instructions;
- repository instructions or required next actions.

Unknown families, implementations, and intensities fail before the agent run. The selected canonical style, family, and intensity are recorded by the owned system-block seam, while the resident cache prefix stays unchanged.

## Response shape is not work policy

Caveman response shape controls **how the answer is written**. Ponytail controls **how implementation work is scoped**. They are intentionally separate so users can eventually mix them:

```text
response=caveman:native:medium
work=ponytail:native:high
```

The Ponytail work profile is tracked in [#6700](https://github.com/anthony-chaudhary/fak/issues/6700); general repeated-profile composition is tracked in [#6707](https://github.com/anthony-chaudhary/fak/issues/6707). Until those ship, `fak agent --output-style` affects response shape only.

## Current boundary

The shipped flag drives `fak agent`, the owned in-process loop. External harness propagation through `fak guard` is not claimed yet. Use `fak agent profiles` as the machine-readable source of currently selectable behavior rather than inferring support from planned issue names.


## Troubleshooting

- **`invalid --output-style`** — run `fak agent profiles`; the requested family, implementation, or intensity is not shipped.
- **Asked for `original`** — this is an intentional refusal, not an installation problem. Track #6706 or choose `caveman:medium` for the safe native implementation.
- **Output is still detailed** — explicit user/repository requirements and required diagnostics outrank the profile. Try `caveman:high`, but required content remains.
- **Using Claude Code, Codex, Cursor, or another external harness through `fak guard`** — the current profile flag does not propagate there yet. Do not set unrelated environment variables and assume coverage; use the owned `fak agent` loop until the guard integration ships.

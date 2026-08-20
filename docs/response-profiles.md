---
title: "Response profiles: concise and composable output"
description: "How fak response profiles control answer compactness independently from tool authority, work scope, diagnostics, and implementation policy."
---

# Response profiles: concise, Caveman-compatible, and composable

Response profiles are a **presentation control** for fak's owned agent loop; `caveman:medium` is default-on and `full` is its explicit ablation. They let a user ask for shorter answers without changing tool authorization, work scope, tests, diagnostics, or evidence requirements.
They compose independently with [work profiles](work-profiles.md): Caveman can shape the response while Ponytail shapes implementation decisions, and neither selection implies the other.

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
fak agent --output-style full  # ablate the default caveman:medium profile
```

`caveman:medium` is the default for the owned `fak agent` loop. `full` is the explicit ablation arm; harnesses that do not launch through `fak agent` remain unchanged.

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

- **Native** means fak-authored profile text, enforced through fak's cache-stable system-prompt MMU and preservation contract.
- **Original** means the vendored upstream-authored material from `juliusbrussee/caveman@c72984e4392c7a154e55c11dbf445f01ce5c35d4`, preserved under its MIT license. Select `caveman:original:{low|medium|high}`; the intensities map to upstream `lite|full|ultra`.

The original-profile readout records source revision and digest, activation source, and the `set FAK_STYLE=full` disable command. Unknown original intensities fail closed. System safety, explicit user formatting, and preservation rules override imported instructions. fak never silently substitutes native behavior when a user asks for `original`.

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

`fak agent` and guarded Claude compose these axes independently. Both default to the native medium profiles; `--output-style full` / `--output-profile full` disables response shaping, while `--work-profile standard` disables Ponytail work policy.

## External guarded harnesses

`fak guard -- claude ...` defaults to `caveman:medium` plus `ponytail:medium` and injects one composed governed fragment through Claude's owned `--append-system-prompt` seam. This matches the owned `fak agent` posture without requiring configuration in the repository being worked on.

The axes remain independent: `--output-profile full` disables Caveman and `--work-profile standard` disables Ponytail. Explicit profile selections resolve before launch. The capture records both canonical profiles and digests, the composite digest, harness, activation seam, whether activation came from defaults, and the disable command.

Claude is currently the only witnessed external injection seam. Default launches of Codex, Cursor, and other wrapped commands remain byte-identical rather than claiming activation. Explicit non-off selections on an unsupported harness fail before child launch with `PROFILE_UNSUPPORTED_HARNESS`. Unknown response or work profiles likewise fail before launch.

## Troubleshooting

- **`invalid --output-style`** � run `fak agent profiles`; the requested family, implementation, or intensity is not shipped.
- **Asked for `original`** � this is an intentional refusal, not an installation problem. Track #6706 or choose `caveman:medium` for the safe native implementation.
- **Output is still detailed** � explicit user/repository requirements and required diagnostics outrank the profile. Try `caveman:high`, but required content remains.
- **Using Claude Code, Codex, Cursor, or another external harness through `fak guard`** � the current profile flag does not propagate there yet. Do not set unrelated environment variables and assume coverage; use the owned `fak agent` loop until the guard integration ships.

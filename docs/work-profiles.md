# Work profiles: choose how fak approaches implementation

Work profiles are **implementation policy** for the owned `fak agent` loop. They answer
"how strongly should the agent challenge avoidable complexity?" Response profiles answer a
different question: "how compact should the final prose be?" You can select either axis independently, and explicitly disable either one.

```bash
fak agent profiles
fak agent --work-profile ponytail:medium --output-style caveman:low --offline
```

The default is `ponytail:medium`; every fak-owned agent loop therefore receives the medium
simplicity policy unless the operator overrides it. Disable it explicitly with `--work-profile standard`.

## Low, medium, or high?

| Selection | Best for | What changes |
|---|---|---|
| `ponytail:low` | Existing projects where you want a light simplicity check | Briefly checks whether existing code, configuration, or deletion is enough. |
| `ponytail:medium` | Most feature and maintenance work | Walks a simplicity ladder and stops at the first complete, correct option. |
| `ponytail:high` | Complexity-sensitive or dependency-sensitive work | Actively demands justification before new machinery or abstractions. |

Friendly names expand to attributed canonical names, for example
`ponytail:medium` becomes `ponytail:native:medium`. `native` means the instruction bytes are
fak-authored and safety-hardened, inspired by the pinned Ponytail study. It does **not** claim to
be Ponytail's original prompt. `ponytail:original:*` is reserved and fails closed until a pinned,
attributed adapter is implemented.

## What a work profile may not change

Intensity controls only how aggressively the agent searches for a smaller correct implementation.
It never permits the agent to:

- drop explicit user scope or repository instructions;
- weaken policy, security, correctness, compatibility, or migrations;
- skip tests, diagnostics, uncertainty reporting, or required proof;
- call fewer lines or fewer dependencies a success when the task is incomplete.

Precedence is fixed:

```text
system policy and explicit user requirements
> repository instructions
> work profile
> response profile
```

## Mix and match

```bash
# Simpler implementation decisions, normal explanatory output
fak agent --work-profile ponytail:medium --output-style full --offline

# Light simplicity pressure, highly compressed response
fak agent --work-profile ponytail:low --output-style caveman:high --offline

# Disable both axes
fak agent --work-profile standard --output-style full --offline
```

The selected family, implementation, intensity, and fragment digest are captured independently in
the owned system-block readout. Both overlays live after the stable resident cache prefix; changing a
profile does not rewrite the resident policy bytes.

## Current boundary and troubleshooting

This release drives the in-process `fak agent` system block. It does not yet prove activation in an
external Claude Code, Codex, Cursor, or other `fak guard` harness; that witnessed transport is #6787.

- **`invalid --work-profile`**: run `fak agent profiles`; unsupported and `original` forms fail closed.
- **Too much implementation pressure**: move from high to medium or low; use `standard` to disable.
- **Output is still verbose**: work policy does not change prose; choose `--output-style` separately.
- **Concern that a requirement was removed**: that violates the profile contract, regardless of
  intensity. Preserve the requirement and use the repository's normal tests/evidence gate.

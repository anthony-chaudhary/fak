# Study: Claude Code Concise output style (2026-08-20)

## Verdict

**Borrow the product lesson, not the prompt or name.** Claude Code v2.1.237 added a built-in
`Concise` output style that leads with results, removes preamble and narration, and promises to keep
the work equally thorough. That response-shaping behavior is already **PRESENT** in FAK: the owned
agent loop defaults to `caveman:medium`, resolves it through the closed `syspromptmmu` profile
registry, injects a fak-authored result-first fragment, preserves correctness and safety detail, and
records the canonical selection plus digest.

The useful gap is narrower and operator-facing. Claude Code makes style selection discoverable and
persistent through `/config` and `outputStyle`; FAK currently exposes `fak agent profiles` and a
per-run `--output-style` flag. Persistent selection is therefore **PARTIAL** and is filed as
[#8288](https://github.com/anthony-chaudhary/fak/issues/8288). No second style engine, borrowed prompt
bytes, or `concise` alias should be added.

Assignment: **DEFAULT** for the existing FAK-native `caveman:medium` behavior; **RECIPE** for
Claude Code's native project setting, now selected in `.claude/settings.json` by
`c55ab2355058`; **OPTIONAL-MODULE** for the persisted FAK operator preference in #8288;
**EXCLUDE** Claude-specific prompt bytes and naming.

## Source and acquisition

Observed on 2026-08-20:

- Claude Code release
  [`v2.1.237`](https://github.com/anthropics/claude-code/releases/tag/v2.1.237), published
  2026-08-20T00:54:41Z, pins to
  [`anthropics/claude-code@770933ea1ad2fa7b858191e397a65e6644771c64`](https://github.com/anthropics/claude-code/commit/770933ea1ad2fa7b858191e397a65e6644771c64).
  Its release note says the built-in style leads with results, skips preamble and narration, and
  does the work just as thoroughly; selection is under **Output style** in `/config`.
- The
  [Claude Code settings reference](https://code.claude.com/docs/en/settings) identifies
  `outputStyle` as a system-prompt setting and says system-prompt settings are rebuilt on `/clear`
  or restart.
- The
  [output-styles guide](https://code.claude.com/docs/en/output-styles) describes the menu as writing
  `.claude/settings.local.json` at project scope and documents direct `outputStyle` configuration.
  At capture time this page still listed Default, Proactive, Explanatory, and Learning rather than
  Concise, so the v2.1.237 release note is the authoritative evidence for the new built-in while the
  guide is evidence only for the established selection/persistence mechanism.
- The local Claude Code binary was initially `2.1.235`, then upgraded with `claude update` and
  verified as `2.1.237`. The project setting parses as JSON and is now committed, but this study does
  not treat a model response sample as proof of broad style effectiveness.

The public repository exposes changelog, release, docs, plugins, and issue surfaces, but not the
proprietary runtime implementation or exact built-in Concise prompt. This is therefore a
behavior-and-interface study, not an implementation port.

## What the feature actually is

Concise is a **response-shape policy in the system prompt**, not a cheaper model, reasoning mode,
permission mode, task-planning policy, or output-token hard cap. Three design choices matter:

1. **Lead with the outcome.** The first useful sentence carries the result rather than announcing
   intent or narrating tool use.
2. **Delete ceremony, preserve substance.** Preamble and narration are removable; thorough work is
   not. This separates visible prose length from implementation effort and correctness.
3. **Make the policy selectable and persistent.** A named built-in appears in the normal config
   surface and has a direct settings representation. Users need not maintain custom prompt text or
   remember a launch flag.

The third choice is the main new evidence for FAK. The first two independently converged with FAK's
existing response profiles and `signal-first` skill before Claude Code v2.1.237 shipped.

## FAK comparison

| Claude Code v2.1.237 capability | FAK evidence | Status | Decision |
|---|---|---|---|
| Named built-in response shape | `fak agent profiles --json` lists closed, canonical selections and statuses | PRESENT | Keep the existing registry; no alias required |
| Lead with results; skip preamble/narration | `internal/syspromptmmu/style.go` medium/high fragments and `.claude/skills/signal-first/SKILL.md` | PRESENT | Keep `caveman:medium` as the owned-loop default |
| Stay thorough despite shorter prose | native profile safety/correctness carve-outs plus signal-first's preserve-substance checklist | PRESENT | Preserve; never convert style into an output cap |
| Per-run selection and explicit disable | `fak agent --output-style ...`; `full` disables shaping | PRESENT | Keep CLI override and off path |
| Discoverable catalog | `fak agent profiles` and `--json` | PRESENT | Keep as the machine/human catalog |
| Persistent project/user preference | no persisted response-style preference; invocation flag only | PARTIAL | Ship #8288 through existing config/TUI primitives |
| Canonical selection and provenance witness | run report records response profile and fragment digest | PRESENT | Extend source provenance when persistence lands |
| Mid-session activation semantics | FAK selects when starting its owned loop; no `/clear` analogue is needed for the spine | ABSENT | EXCLUDE until a real long-lived reconfiguration need is witnessed |
| Exact Claude built-in prompt | proprietary/not published in the inspected source | ABSENT | EXCLUDE; use independently authored bytes |

Current machine witness:

```text
$ fak agent profiles --json
...
{"selection":"caveman:medium","canonical":"caveman:native:medium",
 "family":"caveman","implementation":"native","intensity":"medium",
 "status":"default","meaning":"Default concise response shape with correctness carve-outs."}
```

The relevant current module versions were
`.claude/skills/signal-first@r1+ge530b19e6e`,
`internal/syspromptmmu@r16+g079ceadcaf`, and `cmd/fak@r3121+g9ac3cd9713` at inspection time.
Module revisions are observational current-state evidence; the pinned commits above establish that
FAK's result-first machinery predates the Claude Code release.

## Why no code change belongs in this study

Adding a `concise` synonym or a second prompt fragment would make FAK worse:

- it would duplicate a closed registry capability already selected by default;
- it would conflate Claude's product name with independently authored FAK behavior;
- it would add migration and documentation surface without changing an operator outcome;
- it would hide the actual gap, which is durable selection and provenance.

For Claude Code itself, the smallest complete adoption was configuration: commit `c55ab2355058` adds
`"outputStyle": "Concise"` to the existing project settings while preserving the repo guard hook.
That uses Anthropic's owned mechanism and adds no prompt machinery. For FAK's independent owned-agent
loop, the smallest complete improvement remains the filed persistence spine, not another response
profile. #8288 requires precedence `CLI > persisted preference > shipped default`, validation through
`syspromptmmu.ResolveStyle`, effective-source diagnostics, an explicit `full` off switch, and an
end-to-end owned-loop witness.

## Problem frame

Centrality: **Enabling** — durable selection makes FAK's Core managed-context efficiency easier to
operate; it does not create a new Core behavior.

- **P1 managed context — advanced:** one bounded named fragment replaces repeated launch text.
- **P2 net-true efficiency — advanced:** choose once and reuse the existing registry; do not add
  prompt bytes or a parallel engine.
- **P3 bounded adaptation — preserved:** closed values fail before a run and `full` remains the off
  switch.
- **P4 integrated operations — advanced:** effective configuration and run metadata expose canonical
  value, source, and digest.

For an operator who wants concise owned-agent sessions, the problem today is remembering a flag on
every launch. A persisted preference is better because it is selected once, inspectable, overridable,
and reversible. The witness is a real owned run that receives the persisted canonical fragment and
records it, plus override, invalid-value, and disable cases.

## Honest limits

- The release claim promises equal thoroughness, but Anthropic did not publish an evaluation or
  before/after token distribution with v2.1.237. This study does not convert marketing language into
  a measured efficiency claim.
- FAK's existing profile mechanics are witnessed; broad model-quality effectiveness remains bounded
  by the mixed single-trial audit in `docs/ponytail-default-medium-audit.md` and is not re-proven here.
- Claude's exact style bytes and runtime precedence were not available in the public repository.
- The docs page lagged the release at observation time. No inference about Concise's eventual place
  among the documented built-ins is needed for the FAK decision.

## Reproduction

```powershell
# Primary upstream evidence
gh api repos/anthropics/claude-code/releases/tags/v2.1.237 `
  --jq '{tag_name,published_at,html_url,body}'
gh api repos/anthropics/claude-code/git/ref/tags/v2.1.237 --jq .object.sha

# Current FAK classification witness
fak agent profiles --json
fak version modules --json

# Local harness and committed project selection
claude --version  # 2.1.237
git show c55ab2355058:.claude/settings.json
```

No additional follow-up was found beyond #8288; exact-prompt cloning and mid-session switching are
deliberately excluded rather than silently deferred. The Claude Code project configuration is shipped
in `c55ab2355058`; #8288 remains open because it concerns FAK's separate persisted preference.



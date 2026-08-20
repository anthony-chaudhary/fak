# Codex `exec_command` guard recovery (2026-08-19)

Tracking: [#8192](https://github.com/anthony-chaudhary/fak/issues/8192)

## Verdict

The terminal in Codex session `01a01d6a-f05c-7211-aab1-b980a55e6022` did not
fail to launch. The guard refused every call because Codex advertised the shell
as `exec_command`, while the built-in floor still recognized only
`shell_command` and `functions.shell_command`.

The narrow repair is a **DEFAULT**: admit `exec_command` as another spelling of
the existing cross-platform shell capability and attach the same POSIX and
PowerShell danger rules to its actual `cmd` argument. A benign
`git status --short` is admitted; recursive forced deletion remains
`POLICY_BLOCK`.

## Captured incident

The session's hash-chain-verified guard journal,
`interactive-58292-16daf1ca84d5.jsonl`, contains 25 decisions. Sixteen are
`exec_command` / `DEFAULT_DENY` / `TERMINAL`; control-plane calls such as
`get_goal`, `update_plan`, and MCP resource discovery were admitted. The one
recovery attempt selected the same denied terminal tool. With no filesystem or
git MCP server available, the session could not reach the repository and
eventually reported blocked after 238,441 tokens.

That sequence distinguishes policy refusal from process-launch failure: no
terminal process was started. It also explains why retrying, asking the agent to
continue, or using the guard's existing recovery attempt could not help.

## Root causes

- The Codex host renamed the shell surface, but the embedded policy and harness
  profile were static alias lists. Their anti-drift test could only check names
  already present in that list, so it certified the old dialect.
- `DEFAULT_DENY` had no reason-specific recovery note. The wrapped agent saw a
  terminal verdict without a copyable, bounded operator action.
- The allow-overlay watcher hashed user, repository, and environment layers but
  omitted the per-launch session layer that reload consumed. Consequently,
  `fak guard allow --session TOOL` could write a grant after launch that the
  active guard never noticed.
- The documented TTL example placed flags after the positional tool. Go's flag
  parser stops at the first positional argument, so the example did not
  reliably request an expiry.
- The guard RSI route treated a well-formed reason code as healthy. It did not
  flag 16 repeated denials of a first-class harness tool as a semantic policy
  regression.

## Recovery choices

After this repair, current Codex sessions need no overlay for `exec_command`.
The built-in floor admits it and still evaluates the command argument against
the existing danger rules.

For a future unknown tool spelling, an operator in another terminal can apply a
bounded live workaround:

```text
fak guard allow --ttl 15m TOOL
```

The repository overlay is watched and reloaded by the running guard. The
refusal result and guard TUI now surface that exact option. A deliberate durable
policy change remains the right route for a real custom tool; an overlay is not
a substitute for adding a standard harness surface to the built-in profile.

## Default admission record

1. **Indication:** guarded Codex users whose standard terminal surface is named
   `exec_command` need ordinary repository commands to reach the gateway.
2. **Comparator and non-intervention:** retaining `DEFAULT_DENY` blocks all
   terminal work. A per-user or per-repository overlay works only after the
   operator diagnoses the rename and broadens local configuration.
3. **Benefit:** the observed session's 16 terminal refusals become one admitted
   capability, immediately restoring the normal coding path. This is an
   incident measurement, not a population-wide rate estimate.
4. **Harms and interactions:** adding an exact tool name expands the default
   capability set. The material harm would be bypassing shell danger checks;
   copying both shell dialect rule sets prevents that known interaction.
5. **Uncertainty:** future Codex tool names and argument schemas can drift again,
   and the single captured session does not measure prevalence. Review on the
   next Codex tool-dialect change or any `exec_command` danger-rule bypass.
6. **Contraindications:** do not admit an unknown prefix, skip argument
   inspection, or treat destructive shell operations as ordinary terminal use.
7. **Dose and safeguards:** add one exact alias, inherit both existing shell
   rule families, pin a benign allow and POSIX/PowerShell denies, and retain the
   fail-closed default for all other unknown tools.
8. **Consent and control:** activation is visible in the embedded policy;
   operators can tighten it with an explicit policy or remove the alias. The
   TTL overlay remains a separate, inspectable temporary choice.
9. **Surveillance and movement rules:** monitor `exec_command` default-deny
   counts and destructive-command allow incidents. Remove or narrow the alias
   on any danger-rule bypass; expand the profile only after another advertised
   standard name has an allow-and-deny witness.

**Exposure verdict: DEFAULT.** This is the minimum effective dose for a
first-class Codex capability, while dangerous arguments and unknown tools stay
fail-closed.

## Witnesses

The policy spine is reproducible without launching an agent:

```text
fak preflight --policy cmd/fak/guard-default-policy.json --tool exec_command --args '{"cmd":"git status --short"}'
# ALLOW (NONE)

fak preflight --policy cmd/fak/guard-default-policy.json --tool exec_command --args '{"cmd":"rm -rf /tmp/x"}'
# DENY (POLICY_BLOCK)
```

The regression suite additionally pins both PowerShell and POSIX danger shapes,
the actionable refusal text, the TUI action, and live reload of a session-scoped
allow layer.

## Remaining hardening

The static alias registry can still lag a future harness release, and the RSI
router cannot yet distinguish a legitimate unknown-tool denial from a repeated
denial of an advertised first-class tool. Those are separate prevention and
detection units; they should be tracked rather than weakening the unknown-tool
default.

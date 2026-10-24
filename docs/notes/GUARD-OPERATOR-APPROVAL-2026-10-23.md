---
title: "Guard operator approval: a human may override one refused call"
description: "fak guard allow --approve <ap-id> lets an attending operator admit one exact call the capability floor refused (POLICY_BLOCK rule, DEFAULT_DENY, or the reversibility confirm hold), on by default, while the agent cannot approve itself."
date: 2026-10-23
---

# Guard operator approval

## Problem

Before this change a human could override the `fak guard` floor only for a
DEFAULT_DENY, and only by widening a tool name in the allow overlay. A danger
arg rule (`rm -rf`, `git push`, `sudo`, `Remove-Item -Recurse`, ...) could not
be lifted by any channel, even when the attending operator had asked for exactly
that command. The `operator-escalation` amendment channel
(`internal/policy` `ChannelOperatorEscalation`, `internal/journal`
`GrantChannelOperatorEscalation`) was declared but had no caller.

## Mechanism

1. Every refusal an approval can lift carries `operator_approval_id`
   (`ap-<12 hex>`) in its verdict Meta and wire Detail. The id hashes the tool
   plus its effect-bearing arguments with the confirm token's canonical view
   (`argsForToken`), so a re-proposal that only rewords the description or the
   command spacing keeps its id.
2. The refusal note shows the id to the agent and to the human reading the
   transcript. The runnable command rides only the out-of-band operator remedy
   (`OperatorRemedyCommand`, the `X-Fak-Operator-Remedy` header), as #11504
   requires.
3. The operator approves from their own shell, for example Claude Code's `!`
   prompt inside the guarded session:

   ```bash
   fak guard allow --approve ap-0123456789ab
   ```

   Defaults: a 15 minute TTL, and the session scope when run inside a guarded
   session (`$FAK_GUARD_SESSION_ID`). The running guard's allow watcher applies it
   within about a second; the reload is journaled as a capability grant.
4. The agent re-proposes the identical call. The adjudicator admits it with
   `posture=operator-approved`, `would_deny=<lifted reasons>` and the id in Meta.

`fak guard allow --approve --remove <id>` revokes an approval early.

## What an approval lifts

| Refusal | Lifted for the approved call? |
| --- | --- |
| POLICY_BLOCK arg rule (rm -rf, git push, sudo, ...) | yes |
| POLICY_BLOCK name deny | yes |
| DEFAULT_DENY | yes |
| Reversibility preview-confirm hold | answered: the human approval is the confirmation |
| Credential / host-config path block | no |
| SELF_MODIFY rungs (including the guard's own config) | no |
| SECRET_EXFIL | no |
| Hardwired egress SSRF floor | no |

The FROZEN amendment invariants are unchanged: no policy knob weakens a danger
rule for every call. An approval names one call.

## Agent cannot approve itself

The default floor refuses `fak ... guard ... --approve` from every shell tool
(Bash, PowerShell, shell_command, exec_command) with POLICY_BLOCK, and the
overlay files are already self-modify protected (`protectGuardPolicyConfig`).
A human command typed at the harness prompt (`!` in Claude Code) is not a model
tool call, so it is not adjudicated.

## Witness

- `internal/adjudicator/operator_approval_test.go`
- `cmd/fak/guard_allow_approve_test.go` (end to end on the shipped default floor)
- `internal/gateway/operator_approval_note_test.go`
- `internal/policy` amendment registry and direction tests

## Open follow-ups

- `fak guard allow <tool>` (plain name widening) is still runnable from the
  agent's shell; only `--approve` is refused there.
- Capability-grant rows for an approval are journaled on the live-reload
  channel; tagging them `operator-escalation` needs the reload diff to know
  which knob moved.
- `--approve last` (approve the most recent refusal without copying the id).

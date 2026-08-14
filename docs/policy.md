---
title: "Choose, author, and test a fak policy"
description: "Operator route for selecting a policy mode, validating a capability-floor manifest, and testing representative tool calls before deployment."
---

# Choose, author, and test a fak policy

**Audience:** the operator who owns which tools an agent may call.

This page's primary workflow is a reviewed manifest. First apply the routing rule below.
If the manifest route applies, author `policy.json` and take one checkable next action: run
`fak policy --check policy.json`. After that check passes, the testing section gives the
allow/deny witness to capture. Neither validation nor preflight needs a model or server.

A fak policy is a JSON capability floor. A tool is admitted only when the selected mode and
manifest permit it; explicit denials and argument constraints remain structural checks rather
than model judgments.

## Choose one of the two manifest postures

For this page's manifest workflow, there are exactly two postures: `fail_closed` (the default)
and `admit_and_log` (a temporary unattended observation posture). `complain` is not a third
posture; it is an optional trial modifier used only with `fail_closed`.

The following exception route is outside the manifest workflow and outside the posture table:

- **Host-local interactive exception only:** use the `fak manage` operator allow overlay instead
  of this manifest workflow.
  The shipped coding-tool floor remains in force; `fak manage allow TOOL` adds a host-local
  exception for a `DEFAULT_DENY`. Explicit danger denials and argument rules are unchanged.
  This route is available only to the operator of that guarded host session. Run
  `fak manage allow TOOL`, verify it with `fak manage allow --list`, and stop; the manifest
  workflow and its next check do not apply.

If that exception-only case does not apply, continue with the manifest workflow. The manifest
is loadable by `agent`, `run`, and `serve`; `preflight` reads the same manifest to test one
call without loading a runtime. Select exactly one posture:

| Manifest task | Selection | What happens | Lifecycle rule |
|---|---|---|---|
| Deploy a reviewed least-privilege floor | `posture: "fail_closed"` | Exact and prefix allows pass; explicit denials fail; every other tool gets `DEFAULT_DENY`. | Keep this for normal interactive, served, and production agents. |
| Keep an unattended read-heavy batch moving while measuring gaps | `posture: "admit_and_log"` | Unlisted read-shaped tools are admitted and recorded as `would_deny=DEFAULT_DENY`; write-shaped and explicitly denied tools still fail. | Use only for a bounded observation window; restore `fail_closed` after reviewing the evidence. |

After selecting `fail_closed`, and only then, you may add `complain: ["TOOL"]` to trial one
suspected default-deny false positive. It admits and logs that named tool's **default deny
only**; explicit denials, self-modification checks, and argument rules still fail. Remove the
trial after review and promote a proven required tool into `allow`. Do not add `complain` to an
`admit_and_log` manifest.

The overlay is not a policy mode and is never merged into a manifest. It completes the current
host-local exception task. If that exception later becomes a durable deployment requirement,
begin a separate manifest workflow: add the reviewed tool to `allow`, choose exactly
`fail_closed` or
`admit_and_log`. The overlay and manifest are different lifecycle artifacts: the overlay
records this host operator's interactive exceptions, while the manifest is the reviewed deployment artifact.
That separate manifest review supersedes the host exception for the deployment; do not copy or
load the overlay as a manifest, and do not leave a durable decision in the overlay or trial mode.

## Author the reviewed manifest

For the normal `fail_closed` path:

```sh
# Generate a current, editable starting point from this fak binary.
fak policy --dump > policy.json

# Edit policy.json: retain only required tools and constraints.
# Then validate the schema and refusal-reason vocabulary.
fak policy --check policy.json
```

`--dump` is a starting-point generator, not a policy tailored to your agent. At minimum, keep
the schema version, select the posture, and make the role's allow/deny boundary concrete:

```json
{
  "version": "fak-policy/v1",
  "posture": "fail_closed",
  "allow": ["search_kb"],
  "deny": {"refund_payment": "POLICY_BLOCK"}
}
```

Replace those example tool names with the real harness names. Remove capabilities the role does
not need, add narrow exact or prefix allows, retain explicit denials for named irreversible
operations, and constrain risky arguments. A different schema major is refused, while a
`fak-policy/v1.x` minor is forward-accepted by a binary that supports v1.

Use these deeper routes while editing:

- [Manifest schema and refusal vocabulary](../POLICY.md) is the normative field reference.
- [Policy authoring guide](fak/policy-guide.md) has complete coding, support, DevOps, research,
  and batch examples plus argument, rate, and production patterns.
- [Browser policy editor](policy-editor.html) is an optional local authoring surface; the CLI
  check remains the validation step.

## Test the decision before loading it

Test calls that represent the role boundary, not just JSON syntax:

```sh
# Expected allow: a capability the role needs.
fak preflight --policy policy.json --tool search_kb --args '{}'

# Expected deny: a capability the role must not have.
fak preflight --policy policy.json --tool refund_payment --args '{}'
```

A useful witness contains both outcomes: the needed call returns `ALLOW`, and the prohibited
call returns `DENY` with the expected reason (`POLICY_BLOCK`, `DEFAULT_DENY`, or the applicable
argument-rule reason). Add boundary cases for every argument rule. The [worked testing
sequence](fak/policy-guide.md#testing-policies) covers table-driven checks and live gateway
parity; [policy proofs](proofs/policy.md) state the narrower loader and matching properties the
repository verifies.

If `--check` fails, fix the reported unknown field, unsupported schema major, invalid regex, or
unknown refusal reason before deployment. If `preflight` disagrees with your intent, change the
manifest and rerun both the allow and deny witness; do not compensate by prompting the model.

## Load and operate the policy

Load the validated file on the surface you actually operate:

```sh
fak agent --policy policy.json --offline
fak run --policy policy.json --trace trace.json
fak serve --policy policy.json --addr 127.0.0.1:8080
```

A long-lived `serve` process can reload its configured file through
`POST /v1/fak/policy/reload`; authenticated deployments require the same bearer token as the
other `/v1/fak/*` routes. Treat the manifest as a reviewed deployment artifact: version it,
review widening changes, rerun representative preflights, deploy, then inspect denials and
`would_deny` observations. Remove `complain` entries and `admit_and_log` after their evidence
window; promote only proven required tools into `allow`.

## Support and authority boundaries

- **Supported now:** `fak-policy/v1`, `policy --dump`, `policy --check`, per-call `preflight`,
  policy loading on `agent`/`run`/`serve`, guard's operator allow overlay, and served reload.
- **Authority:** [the root policy reference](../POLICY.md) defines manifest semantics; the
  [CLI reference](cli-reference.md) defines command syntax. This page selects the operator task
  and next action rather than replacing either reference.
- **Scope:** the capability floor bounds tool-call admission at fak's checkpoint. It does not
  prove that an allowed tool is semantically safe, sandbox arbitrary code, or replace human
  approval for consequential effects.
- **Route lifecycle:** this page is the current `gen/now` operator route. A successful
  independent read-back is its promotion evidence; a newer linked authority that supersedes
  these choices is demotion evidence. Runtime implementation gates remain independent of this
  documentation route.

**Next check:** save the reviewed manifest as `policy.json` and run
`fak policy --check policy.json`; proceed to the allow/deny preflight witness only after it
returns `OK`.

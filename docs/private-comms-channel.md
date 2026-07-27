---
title: "Private control channel"
description: "The public route for authorized operators who need the private control bridge to fak's lab GPU servers."
---

# Reach the private control channel

**Audience:** an authorized fak operator who needs to run or read back work on a lab GPU
server. Public readers can use this page to understand the boundary; the credentials,
node map, commands, and transcript remain in the private companion repository.

**Next action:** after updating your authorized `fak-private` checkout through its normal
`main`-branch workflow, open `../fak-private/tools/dgxbridge/README.md` from the `fak`
clone root and follow its discovery and readback procedure. Access to that private repository
is the authorization check. If the sibling or your access is absent, stop and ask a fak lab
maintainer for access; do not reconstruct the route from public notes.

## What this route controls

The channel is fak's private, out-of-band control bridge to the lab GPU servers. An
authorized operator submits a command through the bridge, a persistent server session runs
it, and the operator reads the result from the private transcript. Use it for hardware-gated
work such as real GPU-kernel witnesses and throughput runs.

This public page is a **route, not an operating runbook**. The current commands, session
selection rules, node map, credentials, and recovery procedure are maintained together in
the private README named above. Following that runbook is the supported choice; copying an
old command from a public note is not.

## Current boundary and support

| Context | Current contract |
|---|---|
| Mode | Private operator control; it is separate from the public `fak serve` data path. |
| Generation | Current public `fak` plus the updated `main` version of the adjacent authorized `fak-private` checkout. The private README, not this stub or a dated public note, is the command authority. |
| Lifecycle | Active lab-operator route. Public notes and benchmark records are evidence or history, not bridge instructions. |
| Support | Maintained for authorized lab operators. A public-only checkout intentionally provides no live access. |

The public repository may contain scrubbed outcomes in generic “GPU server” language. It
must not contain a host, endpoint, channel identifier, token, account identifier, raw
transcript, or staged private bridge source. The commit-time admission and public-leak gates
enforce that boundary; see the
[GPU-server private boundary](gpu-server-private-boundary.md) for ownership and gate details.

## When a readback fails: which class, which fix

A failed readback is not one condition. The bridge client separates the classes because
they have different fixes, and until #5103 they all surfaced under a single
`READBACK_WEDGED` token — which sent operators to restart sessions that were never the
problem. Read the class before acting.

| Class | What it proves | Operator action |
|---|---|---|
| `READBACK_WEDGED: sentinel_missing` | The hub answered this session's tail request and the client parsed the reply, but the command's completion marker was not in it. The transport is healthy; the session's shell is wedged. | Restart the control bridge for that session on the server. This is the only class that warrants it, and it is server-side: an authorized operator must do it from the server console, because the boundary allows no inbound login. |
| `HUB_UNRESPONSIVE: hub_timeout` | The poll itself deadlined or errored, so no reply was ever inspected. A slow or unreachable control transport. | Retry in a quieter window with a longer timeout. Do not restart anything; the session is not implicated. The transport oscillates, so a failure here is not durable. |
| `HUB_UNRESPONSIVE: no_tail_reply` | Every poll succeeded and the hub answered nothing at all. | Check that the hub process is running and still joined to the control channel. Again a hub-side condition, not a session one. |
| `HUB_UNRESPONSIVE: tail_reply_unrecognized` | The hub replied, but no reply matched the shape this client parses — client/hub protocol drift. | Update the client's reply parser in the private tree. Restarting sessions cannot fix a parser mismatch. |

Two consequences worth keeping. First, a spent probe budget never reports a wedged shell:
if the loop never polled, it cannot have observed a missing marker, and it says so.
Second, the same session can produce different classes minutes apart — a short fail-fast
probe can deadline (`hub_timeout`) while a full-budget check on the same session reaches
the hub and reports `sentinel_missing`. That is not a contradiction; it means the shell is
wedged *and* the transport is slow, and only the first needs an operator.

Run the client's `selftest` verb before committing to a long command. It uses the full
timeout instead of the short fail-fast probe budget, so it is the cheapest way to learn
which class you are in.

## Prove a readback is restored

The class table tells you which fix to apply. It does not tell you the fix worked, and a
class that clears itself between two probes is exactly the failure mode above. Before
declaring a session usable again — and before a hardware-gated task depends on it — take a
fresh round-trip witness in this order.

1. `dgxbridge doctor` — network-free. It resolves a token and a control channel and stops
   there. An unauthorized or public-only checkout reports `NOT READY` with both checks
   missing; that is the boundary working as designed, not a bridge fault, and no later step
   can run from that host.
2. `dgxbridge doctor -probe` — adds the live round-trip against the resolved session. The
   `readback` sub-check is the gate: green means a command's output actually came back, not
   merely that the session answered a control verb.
3. A trivial captured command through the normal run path — echo a fixed token, or any
   single cheap read-only command — and confirm that token appears in the returned output.
   Predictable output is the whole point: it separates "the bridge returned something" from
   "the bridge returned *this command's* result".

Step 3 is the witness worth recording. Steps 1 and 2 can both pass against a session whose
shell is healthy but whose transport is slow, and `hub_timeout` is not durable, so a single
green probe is weaker evidence than one captured round-trip of known content. Re-run step 3
rather than trusting an earlier green when the answer is close to a decision bar.

Steps 2 and 3 need live credentials. A host without them cannot reach any session — `doctor`
reports `NOT READY` with both checks missing and stops there — so the round-trip witness is
unavailable from that host, but the class logic is still checkable. The bridge package's own
tests are network-free and pin which condition maps to which class, so building the client
from the private snapshot into a scratch Go module and running that package's tests confirms
that a slow transport reports a hub class and never a wedged-shell one. Use that when a
client change needs checking and no session is reachable from your host; it settles which
class a condition maps to, and never whether a given session is healthy right now.

Run that check from a scratch Go module built on the private snapshot. The public tree
carries no bridge package at all, so the package path only resolves once the private client
is staged into such a module; there is no in-repo lane gate to run instead. The checks that
settle the class table above are the readback classification tests, and it is worth knowing
their names: `TestClassifyReadbackSeparatesFailureClasses` pins each condition to its class,
while `TestPreflightReportsSlowHubNotAWedgedSession`,
`TestPreflightReportsWedgedSessionWhenTailRepliesArrive`, `TestPreflightClassSplitsHubFromSession`,
and `TestHubSideHintsDoNotSayRestart` pin the two properties an operator acts on — that a slow
transport never reports a wedged shell, and that a hub-side class never tells anyone to restart
a session. Selecting them with `-run` covering those three prefixes is the fast, unambiguous
answer; run the package whole afterwards to catch the rest.

One caveat when you run it whole. The bg-artifact prune check shells out to the host's
`bash`, and its guard only asks whether a `bash` resolves on `PATH` — not whether that shell
can work in the test directory. On a native Windows host the WSL launcher at
`System32\bash.exe` resolves, then cannot change into a Windows temp path: it reports
`chdir(...) failed` on stderr and starts at the filesystem root instead. The prune command it
runs discards stderr and ends in `; true`, so the shell still exits `0`, every fixture
survives, and the check fails as though the reap logic had regressed. The signature is
narrow — that one check failing, with every expected-reaped artifact reported as surviving,
while the classification test passes. Until the guard probes whether the shell can work in
the test directory rather than merely exist, read a lone prune failure on a Windows host as
host noise. Any other failure, and the classification test above all, is a real regression.

Keep a captured transcript in the private repository whenever it names a host, node, or
channel; a public note carries the outcome in generic terms only.

## Public readback

When lab capacity is being considered for dev-worker dispatch, follow the scrub and schema
rules in [`fleet.md`](fleet.md), then derive the public `fak.lab_readiness/v1` record with
`fak lab readiness --from-reports --write-default --json`. Only
`READY_FOR_DEV_WORK` admits lab-backed dispatch. The other public-safe outcomes—
`WAIT_PRIVATE_RECOVERY`, `GATEWAY_UNREACHABLE`, `AUTH_OR_CHANNEL_BLOCKED`, and
`INDETERMINATE`—hold dispatch while an authorized operator follows the private recovery
runbook.

## Related routes

- [Fleet compute nodes](fleet-compute-nodes.md) — choose the sanctioned compute target for a
  hardware-gated task.
- [GPU-server private boundary](gpu-server-private-boundary.md) — decide what may cross from
  private control into public evidence.
- [Lab development loop](fak/lab-dev-loop.md) — return a scrubbed hardware witness to the
  public development workflow.

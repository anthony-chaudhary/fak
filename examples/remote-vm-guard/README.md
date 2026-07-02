# `fak guard` on a random VM — the network-egress floor

Move your coding agent onto an ephemeral cloud VM and the human steps away. Nobody is
left to click "approve" on a tool call. The attack that turns that convenience into a
breach is an SSRF to the **cloud-instance metadata endpoint** — `169.254.169.254` and
its peers. One GET there returns the VM's IAM role credentials, and a prompt-injected
agent walks off the box with them.

`fak guard` carries a structural **egress rung** into the VM. A tool call that reaches
the metadata / link-local family is refused *by shape* — `EGRESS_BLOCK`, with no model
and no human in the loop. This is why running the guard on a throwaway VM is useful the
moment it boots: the capability floor travels with the agent, and the one destination
that hands out the box's credentials is closed.

## Run it

```bash
examples/remote-vm-guard/run.sh
```

Needs only Go (to build `fak`) — **no model, key, GPU, server, or network**. Each
witness is a pure function of the destination, so the result is identical on every run
and finishes in a second or two after the one-time `go build`.
Expected runtime: the witness run completes in seconds after the build, and the verdicts
are deterministic for the same destinations.

The demo uses `fak egress check`, which runs the **same kernel floor** a guarded
session enforces, so what it shows is what `fak guard -- claude` would do to the same
tool call.

## What it proves

**Blocked — the credential-theft SSRF class.** Every address and name that reaches an
instance metadata service is refused, including the ones a naive IP block would miss:

| Destination | Why it is blocked |
|---|---|
| `http://169.254.169.254/latest/meta-data/iam/…` | AWS / GCP / Azure / DO / Oracle IMDS |
| `http://metadata.google.internal/…` | GCP metadata **by DNS name** (blocking the IP alone is bypassable) |
| `http://169.254.170.2/v2/credentials/` | AWS ECS task-metadata address |
| `http://100.100.100.100/…` | Alibaba Cloud metadata |
| `http://[fd00:ec2::254]/…` | AWS IMDSv6 |
| `curl 169.254.169.254` (no scheme, in a Bash command) | bare IP scanned out of the command line |
| any `169.254.0.0/16` or `fe80::/10` host | link-local is never a legitimate agent destination |

**Allowed — what a real session needs.** The provider API, a public `git clone`, an
ordinary public host, and even a private RFC1918 host (`10.0.0.5`) are *not* blocked —
the floor refuses the metadata/link-local class, not your traffic.

## How it works

The block is a mandatory rung in the in-process reference monitor
(`internal/adjudicator`), backed by the pure classifier in `internal/egressfloor`. It
runs ahead of the affirmative allow, fires under every policy (it needs no opt-in), and
cannot be elided by a rung profile — a security floor narrows, never widens. A refusal
cites the `EGRESS_BLOCK` reason and a bounded witness naming only the offending host and
its class, never the policy.

```text
+------------------+     +------------------------+     +---------------------------+
| agent tool call  | --> | fak guard egress rung  | --> | metadata / link-local     |
| (on the VM)      |     | (internal/egressfloor) |     | class: EGRESS_BLOCK       |
+------------------+     +------------------------+     +---------------------------+
                                    |
                                    +--> everything else: not blocked by this rung
```

**Extend it for your VM.** The hardwired metadata set is the floor; a policy manifest can
*tighten* it with your own sensitive destinations (an internal secrets service, a corp
metadata mirror) — it can never carve a hole in the hardwired block:

```json
{ "version": "fak-policy/v1",
  "egress": { "deny_hosts": ["secrets.corp.internal", "10.0.0.53"] } }
```

```bash
fak guard --policy my-floor.json -- claude   # blocks the metadata class AND your hosts
```

Wrap a live agent the same way and the floor rides into the VM with it:

```bash
fak guard -- claude          # every tool call crosses the floor; metadata SSRF is refused
```

## Honest boundary

This rung blocks the cloud-metadata / link-local **destination class** — the universal,
never-legitimate target. It is **not** a full deny-by-default egress allow-list (that is
the documented next increment: a policy-configurable allowed-destination set layered on
the same seam). It inspects only args that NAME a destination (`url`, `endpoint`,
`webhook`, …) and shell command lines that fetch one (`curl` / `wget` /
`Invoke-WebRequest`) — it deliberately does **not** scan file-content args, so writing a
doc, test, or security note that *mentions* `169.254.169.254` (this very demo) is never
refused; only *reaching* the endpoint is. A destination resolved at runtime through a
custom DNS name that points at the metadata IP is a known residual a name block cannot
see. The strategy this is the first increment of is in
[`docs/notes/RESEARCH-cloud-vm-remote-agent-landscape-2026-06-23.md`](../../docs/notes/RESEARCH-cloud-vm-remote-agent-landscape-2026-06-23.md)
(recommendation #1).

## Where this fits

- The flag witness, standalone: `fak egress check --url <URL> | --command <CMD> | --host <HOST>`
- The wrapping form: [`../../cmd/fak/guard.go`](../../cmd/fak/guard.go) (`fak guard -- <agent>`)
- The auth half of running off-loopback: [`../auth-hardening/`](../auth-hardening/)
- The classifier + the rung: [`../../internal/egressfloor/`](../../internal/egressfloor/) · [`../../internal/adjudicator/decide.go`](../../internal/adjudicator/decide.go)

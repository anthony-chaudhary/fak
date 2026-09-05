---
title: "Air-gapped deployment kit + supply-chain posture"
description: "The hardened, no-egress fak bring-up for a regulated boundary, the captured zero-network governed-session witness, the generated SBOM, and the reviewer checklist."
---

# Air-gapped deployment kit + supply-chain posture

**Audience:** an operator or a FedRAMP/IL-style reviewer standing fak up inside a
network boundary that nothing leaves. Tracking issue:
[#3279](https://github.com/anthony-chaudhary/fak/issues/3279) (epic
[#3256](https://github.com/anthony-chaudhary/fak/issues/3256), workstream C).

The pitch this page has to make honest: fak is **local-first *and* governed**. The
January 2026 scan that found 175,108 publicly-reachable, auth-less local-model servers
is the failure mode this kit exists to prevent — running locally is the easy half,
staying governed while doing it is the half that needs a checklist.

Related routes, which this page does **not** restate:

- [Choose a deployment envelope](deployment.md) — which topology you are in.
- [Edge quickstart](fak/edge-quickstart.md) — the 5-minute offline audit-trail walk.
- [Supply-chain posture: reproducible builds](supply-chain-reproducibility.md) (#3711) —
  the byte-identity witness. This page links it rather than duplicating it.

## What ships, and what does not

| Element | Status |
|---|---|
| One static `CGO_ENABLED=0` binary, no request-path egress | **Shipped** |
| Required-bearer-token door (`--require-key-env`) | **Shipped** |
| Loopback-by-default listener (`--addr 127.0.0.1:8080`) | **Shipped** |
| Offline capability floor (`fak preflight`) + tamper-evident journal (`fak audit verify`) | **Shipped** |
| Captured zero-network governed session, **mock-planner seam** | **Shipped** — witness below |
| Generated SBOM ([`sbom/fak.spdx.json`](sbom/fak.spdx.json)) | **Shipped** |
| SBOM/`go.mod` drift gate (`go test ./internal/architest -run TestSBOM`) | **Shipped** — see [Regenerate and verify](#regenerate-and-verify-this-sbom) |
| Shift-left zero-egress bootstrap gate (`make test-airgap`) | **Shipped** (#11387) — witness below |
| Captured zero-network governed session, **`--gguf` model-backed seam** | **Not yet witnessed** — see [Not yet witnessed](#not-yet-witnessed) |
| A *guard* that refuses an auth-less bind on a routable interface | **Shipped** (#5373) — startup refusal `UNAUTHENTICATED_OFF_HOST_BIND`. See [Bind safety](#bind-safety-read-this-one) |

## Stage the artifacts (what crosses the boundary, once)

Everything below crosses the boundary on removable media or an internal mirror, and
nothing crosses again at runtime:

1. **The binary.** Build it outside the boundary from a tagged commit through the one
   canonical recipe, then carry the artifact in:

   ```sh
   git checkout v0.41.0
   GOOS=linux GOARCH=amd64 CGO_ENABLED=0 \
     OUT=/tmp/fak VERSION="$(cat VERSION)" sh scripts/build.sh
   sha256sum /tmp/fak     # record; compare against the published SHA256SUMS entry
   ```

   Or carry the source clone and build **inside** the boundary — the module vendors
   nothing but needs its two `golang.org/x` modules in the local module cache, so
   pre-populate `GOMODCACHE` or pass `-mod=mod` against an internal proxy. See
   [Supply-chain posture](#supply-chain-posture) for exactly which two.
2. **The policy manifest** — your capability floor, e.g. a copy of
   [`examples/customer-support-readonly-policy.json`](../examples/customer-support-readonly-policy.json).
3. **The model weights** (optional, only for the model-backed tier) — a GGUF file for
   `--gguf`. The kernel proof below needs no model at all.
4. **This page and [`sbom/fak.spdx.json`](sbom/fak.spdx.json)** — the reviewer's evidence.

## Hardened bring-up

```sh
# 1. The token door. Required on EVERY request; the gateway compares digests in
#    constant time and 401s without it.
export FAK_API_KEY='<a token your secret store issued>'

# 2. Serve. Explicit loopback bind + required key + the local model path.
#    Drop --gguf to run the kernel/mock tier with no weights at all.
./fak serve \
  --addr 127.0.0.1:8080 \
  --require-key-env FAK_API_KEY \
  --gguf /srv/models/your-model.gguf

# 3. Prove the door is shut, from the same host.
curl -f http://127.0.0.1:8080/healthz                       # live
curl -s -o /dev/null -w '%{http_code}\n' \
  http://127.0.0.1:8080/v1/chat/completions -d '{}'         # expect 401 (no bearer)
```

Flags used, all verified live in [`cmd/fak/serve.go`](../cmd/fak/serve.go):

| Flag | Default | What it does |
|---|---|---|
| `--addr` | `127.0.0.1:8080` | HTTP listen address. The default is **loopback**, so an unconfigured `fak serve` is not reachable off-host. |
| `--require-key-env` | `""` (**no auth**) | Names an env var holding a bearer token to require on every request. Empty means no auth — set it. |
| `--gguf` | `""` | Load local GGUF weights into the in-kernel engine at boot. With `--gguf` and no `--base-url`, `/v1/chat/completions` and `/v1/messages` are served in-kernel, so there is no upstream to call. |

### Bind safety (read this one)

The default `--addr` is loopback, and since #5373 the 175,108-server shape is a **kernel
refusal, not a convention**: `fak serve` will not come up on an interface reachable from
off this host while no inbound token door is named. Captured on this host from the staged
binary — it exits before any socket is bound:

```
$ ./fak serve --addr 0.0.0.0:8080
fak serve: UNAUTHENTICATED_OFF_HOST_BIND — refusing to bind 0.0.0.0:8080, which is
reachable from off this host, with no inbound token door: every request would be served
unauthenticated. Fix it one of three ways: bind loopback (--addr 127.0.0.1:8080, the
default), require a bearer (--require-key-env FAK_API_KEY, with that env var set), or bind
per-tenant keys (--key-principal acme=ACME_KEY). If this host really is meant to serve an
unauthenticated interface, pass --unsafe-allow-unauthenticated-bind to proceed anyway.
  next:   fak recover UNAUTHENTICATED_OFF_HOST_BIND
$ echo $?
2
```

What a reviewer needs to know about the rule's edges, all of it in
[`cmd/fak/serve_bind_safety.go`](../cmd/fak/serve_bind_safety.go):

- It fires only on the **conjunction** — reachable off-host **and** no token door. An
  off-host bind that names `--require-key-env` or `--key-principal` is admitted, and so is
  every loopback bind (`127.0.0.0/8`, `::1`, `localhost`) and `--stdio`, which binds
  nothing.
- It classifies by **IP value, not string prefix**, so `127.0.0.1.evil.example` is not
  mistaken for loopback. Wildcards (`0.0.0.0`, `::`, a bare `:8080`) are off-host.
- It **fails open on ambiguity**: a host that is not a parseable IP (a DNS name) cannot be
  proven off-host at boot, so it is admitted rather than wedging a boot. That is the one
  gap this refusal does not cover — see the checklist below.
- `UNAUTHENTICATED_OFF_HOST_BIND` is a stable token declared in `dos.toml`, so log scrapes
  can match the refusal by token rather than by prose.
- The escape hatch is `--unsafe-allow-unauthenticated-bind`, for the isolated lab segment
  or the host firewall genuinely doing the work. It prints a loud stderr warning **every
  boot** naming the overridden refusal. Deliberate ATO-visible risk, not a default.

In a regulated boundary the host controls are still yours to own, because the refusal
governs *this* process and not the network around it:

- Deny inbound on the listener port at the host firewall / network policy.
- If `--addr` names a **DNS host** rather than an IP, the refusal cannot classify it —
  assert the token door yourself in your bring-up check.
- If anyone passes the override, record it as a residual risk with its compensating
  control.

## The zero-network governed-session witness

This is acceptance bullet 1's captured artifact, at the **mock-planner seam**: a
governed session that runs end to end with no network, no model, no key, and no GPU.
Reproduce it inside the boundary with the staged binary and policy.

```
### 1. binary build info (2 deps, CGO_ENABLED=0)
	mod	github.com/anthony-chaudhary/fak	v0.41.1-0.20260724051306-1a9fcd5adef1+dirty
	dep	golang.org/x/sys	v0.46.0	h1:noSf2Fq6F8DBgS+LysIkx7rIExoNHJsxOAtPp4rthXw=
	dep	golang.org/x/term	v0.44.0	h1:0rLvDRCtNj0gZkyIXhCyOb2OAzEhLVqc4B+hrsBhrmc=
	build	CGO_ENABLED=0

### 2. capability floor: DENY
verdict=DENY reason=POLICY_BLOCK by=monitor
### 3. capability floor: ALLOW
verdict=ALLOW reason=NONE by=monitor

### 4. governed session, no network
seam        : OFFLINE (deterministic mock planner)
injection in context                YES           no
destructive op executed             YES           no
task completed (booked)             YES          YES
  destructive op prevented  : YES
```

The commands that produced it, run from a scratch directory with the staged binary:

```sh
go version -m ./fak | grep -E '^\s+(mod|dep|build\s+CGO)'
./fak preflight --policy ./customer-support-readonly-policy.json --tool refund_payment --args '{}'   # -> DENY  (POLICY_BLOCK)
./fak preflight --policy ./customer-support-readonly-policy.json --tool search_kb     --args '{}'   # -> ALLOW (NONE)
./fak agent --offline
```

What a reviewer should take from it: the refusal is **structural**. `refund_payment` is
denied by the capability floor with no model in the loop, so there is no prompt to
talk past; and in the governed session the injected instruction reaches the context in
the baseline arm (`YES`) but not the fak arm (`no`), while the task still completes.

Captured 2026-07-24 at commit `1a9fcd5ad`, Go `go1.26.5`. The `+dirty` suffix is this
shared-trunk working tree, not the release artifact — a release build stamps the tag.

Pair it with the tamper-evident trail for the evidence half:

```sh
./fak audit verify <journal.jsonl>     # exit 1 if the hash chain was edited
```

## Shift-left network-isolated harness bootstrap gate with zero-egress enforcement

To guarantee that air-gapped harness bring-up cannot leak traffic, contact remote services, or depend on out-of-boundary resources, `make test-airgap` (`TestAirGapBootstrap_ZeroEgress` in `internal/airgaptest`) enforces a shift-left CI gate with zero-egress enforcement.

Run the gate locally or in CI:

```sh
make test-airgap
# Runs: go test -v ./internal/airgaptest -run 'TestAirGapBootstrap_ZeroEgress'
```

### Enforced invariants and failure tokens

1. **Zero-egress enforcement (`AIRGAP_EGRESS_VIOLATION`):**
   - The test environment isolates proxy variables (`HTTP_PROXY=http://127.0.0.1:1`, `HTTPS_PROXY=http://127.0.0.1:1`, `ALL_PROXY=http://127.0.0.1:1`, `GOPROXY=off`).
   - An active socket tracer/egress trap intercepts all dial attempts. Any connection attempt to a non-loopback address (not `127.0.0.1`, `::1`, or `localhost`) fails closed immediately with error token `AIRGAP_EGRESS_VIOLATION`.
   - The test asserts that exactly zero outbound connections outside loopback occurred across the entire bootstrap and execution lifecycle.

2. **Offline remote dependency refusal (`AIRGAP_UNRESOLVED_REMOTE_DEPENDENCY`):**
   - Bundles and lockfiles are verified strictly offline. If a lockfile or bundle references a remote MCP server URL (such as `http://external.service/mcp` or `https://...`) or unbundled remote assets, `fakpack.Verify` and bootstrap preflight fail closed with error token `AIRGAP_UNRESOLVED_REMOTE_DEPENDENCY`.

3. **Hermetic end-to-end execution:**
   - A synthetic self-contained `.fakpack` bundle is built with valid v2 `harness.lock.json`, local MCP server binary (`os.Args[0]` `TestHelperProcess` pattern), local disk-backed memory journal, in-kernel mock model, and security policy.
   - The bundle is verified offline with `fakpack.Verify`.
   - The all-in-one supervisor boots from the bundle, waits for `/healthz` to report `200 OK`, and dispatches agent turns via `/v1/fak/agent/sessions`.
   - Full tool dispatch to the MCP child process over stdio, session completion (`session.end`), and durable disk-backed memory journal records are verified.

## Supply-chain posture

**The honest, build-verifiable statement:** fak is one static Go binary whose entire
external dependency set is **two `golang.org/x` extended-standard-library modules**,
pinned by a **4-line `go.sum`**.

> **The older "zero external dependencies, no `go.sum`" phrasing is stale — do not use it.**
> It was true once; it is not true at this commit, and shipping it would be a false claim.
> The correct number is two modules. The `go.mod` header comment has since been corrected,
> and `internal/architest/zerodep_claim_test.go` reds if the older wording reaches a new
> reader-facing page — three (`AGENTS.md`, `INDEX.md`, `docs/product-scorecard/README.md`)
> still carry it as pinned, review-visible debt.

| Module | Version | Direct? | License |
|---|---|---|---|
| `golang.org/x/term` | v0.44.0 | direct | BSD-3-Clause |
| `golang.org/x/sys` | v0.46.0 | indirect (via `x/term`) | BSD-3-Clause |

Why this is a defensible posture rather than a slogan:

- **Both are Go-project-maintained** extended-stdlib modules, not third-party
  transitive sprawl. The comparison class — an LLM proxy whose PyPI dependency tree was
  the March 2026 supply-chain attack surface — is a different order of magnitude.
- **No C toolchain.** The shipped binary is `CGO_ENABLED=0` (verifiable above via
  `go version -m`), so no host C library leaks into the artifact.
- **No request-path egress.** With `--gguf` and no `--base-url` the model runs
  in-process; there is no upstream to call.
- **Reproducible.** Byte-identity is witnessed by a double-build CI job — see
  [`supply-chain-reproducibility.md`](supply-chain-reproducibility.md). The one
  explicitly excluded path is the `Dockerfile.cuda` image (unpinned C toolchain).

The generated SBOM is [`sbom/fak.spdx.json`](sbom/fak.spdx.json) (SPDX 2.3).

### Regenerate and verify this SBOM

Every package entry resolves from the build, with no network and no third-party tool:

```sh
go list -m all                    # -> exactly: the module + x/sys v0.46.0 + x/term v0.44.0
cat go.sum                        # -> exactly 4 lines; the h1: digests in the SBOM's sourceInfo
go version -m ./fak | grep dep    # -> the module set actually LINKED into the shipped binary
```

`go version -m` is the authoritative one: it reports what is in the artifact, not what
the source tree merely declares. If those three disagree with
[`sbom/fak.spdx.json`](sbom/fak.spdx.json), the SBOM is stale — regenerate it against
`go list -m all` and re-cut. A release that changes `go.mod` must re-cut the SBOM.

**You no longer have to remember to run them.** `TestSBOMMatchesGoMod` in
[`internal/architest/sbom_drift_test.go`](../internal/architest/sbom_drift_test.go) reds the
trunk on every `go test ./...` when this SBOM and `go.mod` disagree, and names the module and
the direction:

- a module `go.mod` requires that the SBOM omits is the **blind spot** — bytes ship that a
  reviewer reading this document never learns about;
- a version the SBOM states that `go.mod` contradicts is worse than silence, because a
  reviewer checks advisories against the version named here;
- a module the SBOM lists that `go.mod` no longer requires is the **weaker** direction (no
  unlisted bytes ship) but still a false claim in a regulated artifact, and it makes an
  operator stage a module zip the build never asks for.

It also catches a half-refreshed entry (`versionInfo` bumped, purl or proxy URL left behind),
and fails closed on a `replace` directive, which this SPDX shape has no field to express. The
gate covers the `require` set, direct and indirect; `exclude` and the toolchain directives are
deliberately out of scope because they put no bytes in the artifact.

## Regulated-deployment checklist

Work top to bottom; each line is checkable, not aspirational.

**Boundary**

- [ ] Shift-left zero-egress harness bootstrap gate verified via `make test-airgap` (`TestAirGapBootstrap_ZeroEgress`).
- [ ] The binary, policy manifest, and model weights were staged once; no runtime path
      leaves the boundary.
- [ ] Egress is denied at the host firewall / network policy, and that denial is
      captured by an operator-owned check. `/healthz` does **not** prove isolation.
- [ ] `--base-url` is unset (or points inside the boundary) so no upstream is dialed.

**Access**

- [ ] `--require-key-env` is set and the token comes from a secret store, not a file in
      the image.
- [ ] `--addr` is loopback, **or** it is routable *and* the token door is on *and* a
      firewall rule backs it (see [Bind safety](#bind-safety-read-this-one)). fak refuses
      the auth-less routable case itself; the firewall rule is yours.
- [ ] `--addr` names an **IP, not a DNS host** — the bind refusal cannot classify a name,
      so a name is the one shape it admits unchecked.
- [ ] `--unsafe-allow-unauthenticated-bind` is **not** passed (`grep` your unit file); if
      it is, record it as residual risk with its compensating control.
- [ ] The 401-without-bearer check above was run and passed post-deploy.

**Governance**

- [ ] A capability floor (policy manifest) is loaded, and a known-destructive tool was
      confirmed `DENY / POLICY_BLOCK` on this host.
- [ ] The audit journal is on durable storage and `fak audit verify` passes.
- [ ] Retention meets your regime (the journal is append-only and SHA-256 hash-chained;
      the edge quickstart maps the structure to EU AI Act Article 12). Retention policy
      itself is tracked as #C3 of the epic.

**Supply chain**

- [ ] [`sbom/fak.spdx.json`](sbom/fak.spdx.json) was regenerated and matches
      `go list -m all` at the deployed commit.
- [ ] The deployed binary's `sha256sum` matches the published `SHA256SUMS` entry, or it
      was rebuilt in-boundary and compared per
      [`supply-chain-reproducibility.md`](supply-chain-reproducibility.md).
- [ ] `go version -m ./fak` records the expected `vcs.revision`.

**Residual risk to record in your ATO package**

- [ ] The `--gguf` model-backed air-gapped run is not yet covered by a captured
      upstream witness — you own that evidence for your own bring-up.
- [ ] If `--addr` is a DNS name, or the bind override was passed, record the compensating
      control — those are the two shapes the bind refusal does not cover.

## Not yet witnessed

Stated plainly so nothing above has to be walked back.

**The `--gguf` model-backed air-gapped session has no captured witness here.** The
witness above is the mock-planner seam, which needs no weights. The model-backed run
needs a host with resident GGUF weights; it was not reachable from the Windows
development box (no local model, and native `go test` is blocked by an OS
Application-Control policy — see
[`AGENTS.md`](../AGENTS.md)). It is not blocked on missing capability, only on
execution: run the [hardened bring-up](#hardened-bring-up) with `--gguf` on a fleet
compute node ([`fleet-compute-nodes.md`](fleet-compute-nodes.md)), capture the
`/healthz` + 401 + governed-turn transcript, and append it beside the witness above.

**Generation bookkeeping** (`gen/next`, per [`generation.md`](generation.md)):

- **Promotion evidence** — what moves #3279 toward `now`. Two were named; **one has now
  landed.** The kernel-side bind-safety refusal shipped (#5373, `serve_bind_safety.go`,
  witnessed above), which retired this page's largest fence: the Direction's
  "never expose auth-less on a routable interface" is a startup refusal rather than the
  operator convention this page used to document. The remaining one is the captured
  `--gguf` model-backed air-gapped transcript (#5372). When that lands, the
  [Not yet witnessed](#not-yet-witnessed) section empties and this page promotes.
- **Demotion / retirement evidence**: if epic #3256 re-scopes away from air-gap, or a
  supported distro/image path subsumes the single-binary kit, this kit demotes in favor
  of that path and the checklist moves with it.
- **Invalidating assumption (live, from #5373)**: that "reachable from off this host" is
  decidable from the `--addr` string. It is not, for a DNS name — the refusal deliberately
  admits what it cannot prove, so that a malformed-but-harmless address never wedges a
  boot. An operator who binds `--addr fak.internal:8080` with no token door gets **no
  refusal**, and this page's table row still reads **Shipped**. That is the honest edge of
  the guard, and it is why the checklist keeps an IP-not-a-name line rather than treating
  the refusal as total coverage. Falsify it by binding a resolvable off-host name with no
  token door: if that must be refused too, the rule needs a boot-time resolve it does not
  do today.
- **Invalidating assumption (retired, #5374)**: that the dependency set stays at two
  `golang.org/x` modules. It is not a law — it went from zero to two without this page
  noticing, which is precisely how the stale "zero deps" claim survived. The freshness of
  the SBOM no longer rests on the last person who ran the three commands in
  [Regenerate and verify](#regenerate-and-verify-this-sbom): `TestSBOMMatchesGoMod` in
  `internal/architest` reds the trunk on SBOM/`go.mod` divergence and names the drifted
  module. What the gate still does NOT check is `go.sum`'s `h1:` digests against the ones
  quoted in each entry's `sourceInfo`, and it compares the declared `require` set rather
  than `go version -m` on a built binary — so a linked-set surprise that `go.mod` does not
  show would still get past it.

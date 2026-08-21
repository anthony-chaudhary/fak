---
title: "Managed hosts: corporate TLS interception and non-bearer upstreams"
description: "Why a setup that works in your shell can silently degrade inside fak on a corporate box, and the one trust knob (FAK_CA_BUNDLE) plus the two refusals (UPSTREAM_TRUST_UNVERIFIED, UPSTREAM_UNSUPPORTED) that make the posture nameable."
---

# Managed hosts

A **managed host** is a machine whose network and identity are administered by
someone other than you: a corporate laptop, a locked-down CI runner, a VDI
desktop. Two things about such a host break assumptions fak used to make
silently, and both present as "fak is broken" rather than as themselves.

1. **TLS is intercepted.** A proxy terminates every outbound connection and
   re-signs it with a private root. Nothing is blocked. The chain simply does not
   validate against the trust store the tool is using.
2. **The model upstream is not a bearer-token API.** The host is pointed at
   Amazon Bedrock or Google Vertex, where each request is signed with SigV4 or an
   ADC token instead of carrying an `Authorization: Bearer` header.

Neither is exotic. Both were invisible to fak, which is the actual defect: the
failure surfaced somewhere else, as something else.

## Symptom 1: everything reads as "the network is down"

Every runtime reports interception differently, and none of them says
"interception":

| Runtime | What you see |
|---|---|
| Go (fak itself) | `x509: certificate signed by unknown authority` |
| Node (Claude Code) | `SELF_SIGNED_CERT_IN_CHAIN` |
| Windows/schannel | `CRYPT_E_NO_REVOCATION_CHECK` |
| curl / OpenSSL | `self-signed certificate in certificate chain` |

All four mean the same thing, and all four read as a firewall. The tell is that
`curl https://api.anthropic.com` fails the same way — a real firewall usually
hangs or refuses the connection instead of completing a handshake you cannot
verify.

### One trust source

Point `FAK_CA_BUNDLE` at a PEM file holding your corporate root:

```bash
export FAK_CA_BUNDLE=/etc/corp/ca-bundle.pem      # a file, not an OS-store install
fak doctor trust
```

That is the whole configuration. What fak does with it:

- **Widens, never replaces.** `RootCAs` is the platform trust store **plus** your
  bundle. This is not a nicety. A site routinely runs more than one interceptor —
  the proxy in front of the model endpoint is frequently not the one in front of
  the cloud control plane — so building a pool from your bundle alone trades one
  broken endpoint for every other endpoint that was already working. If the
  platform store cannot be read at all, fak refuses rather than narrowing trust to
  the bundle alone.
- **Derives the sibling variables.** No two runtimes read the same one, so
  declaring the bundle once sets `NODE_EXTRA_CA_CERTS`, `AWS_CA_BUNDLE`,
  `CURL_CA_BUNDLE`, `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, and `GIT_SSL_CAINFO`
  for children that do not already have their own. A value you set by hand is
  never overwritten — derivation fills gaps, it does not outrank you.
- **Refuses when it cannot honor what you declared.** A bundle that does not read,
  or that carries no `CERTIFICATE` block, exits with
  [`UPSTREAM_TRUST_UNVERIFIED`](config-bails.md) instead of quietly falling back to
  the platform default and failing on the first request.

There is deliberately **no `--insecure`, no `--skip-verify`, and no
`InsecureSkipVerify` anywhere in the trust path.** Interception is a trust
problem; a governance tool that ships a "just don't check" flag has published the
answer to every future incident review. `fak doctor trust` names the intercepting
CA without one, by reading the chain out of the verification failure itself.

### Reading `fak doctor trust`

```console
$ fak doctor trust
== fak doctor: upstream trust ==
trust source: FAK_CA_BUNDLE=/etc/corp/ca-bundle.pem
interception: WITNESSED on this host (see the upstream-tls rows)
[OK  ] trust-source        validating against the platform trust store PLUS /etc/corp/ca-bundle.pem (1 certificate(s): Corp Root CA)
       recommend: children inherit the same file via NODE_EXTRA_CA_CERTS, AWS_CA_BUNDLE, …
[OK  ] bundle-expiry       soonest certificate expiry 2031-04-02T00:00:00Z
[OK  ] upstream-tls        api.anthropic.com:443: TLS-INTERCEPTED — the platform trust store rejects this
                           chain (x509: certificate signed by unknown authority) and it validates once
                           /etc/corp/ca-bundle.pem is appended; the interceptor's CA is "Corp Root CA"
[OK  ] child-runtime-trust every child runtime inherits fak's trust source (…)
doctor: healthy (0 findings)
```

Severity answers "must you act", not "is this host unusual". An intercepted host
whose interceptor fak already trusts is a *working* host, so it stays green — a
check that goes red on every correctly-configured corporate box is a check nobody
runs twice. What warns:

| Row | Why it warns |
|---|---|
| `trust-source` | A declared bundle did not load, **or** interception is in play and nothing is declared. |
| `bundle-expiry` | A certificate in the bundle is expired or expires within 30 days. Every intercepted connection on the host fails the moment it passes. |
| `upstream-tls` | An endpoint validates neither on the platform store nor with your bundle appended — usually a *second* interceptor whose root the bundle does not carry. Concatenate its PEM; adding a root never removes one. |
| `sibling-trust-vars` | Another runtime here already points at a corporate bundle and `FAK_CA_BUNDLE` does not. This needs no network and is the single most common finding. |
| `child-runtime-trust` | A wrapped child would receive a **narrower** trust store than fak itself, so a runtime the agent shells out to still fails. |
| `cloud-route` | See below. No certificate fixes this one. |

Probing is on by default, bounded, and read-only. An endpoint that cannot be
reached at all is reported as "no trust verdict available" rather than as a trust
failure, so an offline or air-gapped host raises nothing. `--probe=false` keeps the
run strictly local.

On Windows and macOS there is a third state worth recognizing, because it is the
one that gets misread. If IT installed the root machine-wide, the OS verifier
already accepts the intercepted chain, so with **nothing declared** the row reads:

```console
[OK  ] upstream-tls        api.anthropic.com:443: the chain validated against the platform
                           trust store (anchor "corp-proxy-root")
```

No interception is *witnessed*, because there was no verification failure to read a
chain out of — the anchor name is the tell that a private CA is terminating the
connection. fak's own calls are fine here and `FAK_CA_BUNDLE` is not required for
them. Declare it anyway if the agent shells out to Node, Python, or curl: those
read their own variables and do not consult the OS store, which is why a host can
pass this check and still fail inside the wrapped session. The
`sibling-trust-vars` row is the same observation arriving from the other direction.

## Symptom 2: a 24-hour wait for a login that cannot happen

`fak guard` puts its gateway in front of the agent by repointing
`ANTHROPIC_BASE_URL` at itself. A child running with `CLAUDE_CODE_USE_BEDROCK=1`
(or `CLAUDE_CODE_USE_VERTEX=1`) **never reads that variable**: it signs each
request itself and talks to the cloud endpoint directly. The repoint is inert, and
there is no model traffic for the gateway to adjudicate.

What that used to look like is worth writing down, because it is the shape of the
whole bug class:

```
fak guard: parked — STALE_CRED: waiting for a re-login to land at
  …\.credentials.json (poll every 2m0s, ceiling 24h0m0s)…
```

The host had a perfectly good working Bedrock credential. fak checked for a
subscription credential file it does not use on that posture, did not find one,
and parked for 24 hours waiting for a re-login that could never change which
credential the child sends. The credential gate was right about human-paced
recovery and had the wrong precondition.

Now the posture is named before any of that, as
[`UPSTREAM_UNSUPPORTED`](config-bails.md), with the route that does work:

```console
$ fak guard -- claude
fak guard: the wrapped agent is routed to AWS Bedrock (SigV4-signed requests) by
  CLAUDE_CODE_USE_BEDROCK, so it authenticates each request itself and ignores
  ANTHROPIC_BASE_URL — fak's gateway repoint cannot take effect and the session would
  be adjudicated by nothing. Observed: CLAUDE_CODE_USE_BEDROCK=1
  reason: UPSTREAM_UNSUPPORTED
  config: env CLAUDE_CODE_USE_BEDROCK = on  (want: unset, to route this session through fak's gateway with a bearer credential)
          env ANTHROPIC_BASE_URL = (would be injected, and ignored by the child)
          env FAK_GUARD_ALLOW_UNSUPPORTED_UPSTREAM = (unset)  (want: 1 to launch anyway with the model traffic unadjudicated)
  check:  fak doctor trust
  next:   fak recover UPSTREAM_UNSUPPORTED
  works:  fak serve --stdio --policy FILE   (fak as an MCP server the agent calls —
          provider-agnostic, so the capability floor is enforced whatever the model wire is)
```

`fak serve --stdio` is not a consolation prize: the capability floor, tool
brokering, transcript, and refusal vocabulary all apply, because the agent calls
fak rather than fak intercepting the agent. It is provider-agnostic by
construction.

If you want the guard anyway for its other properties — the hook floor, tool
brokering, transcript, sandbox — with the model traffic **unadjudicated**:

```bash
export FAK_GUARD_ALLOW_UNSUPPORTED_UPSTREAM=1
```

Every launch then says so on its startup report, in those words. A waiver that
went quiet would be worse than the refusal it replaced.

## Setup patterns

Four ways to land trust on a managed host, in the order most sites should prefer
them. Working examples: [`examples/corporate-tls/`](../examples/corporate-tls/README.md).

| Pattern | When | How |
|---|---|---|
| **OS store** | Your IT team already installs the root machine-wide, and every tool that reads the OS store works. | Nothing to configure — fak validates against the platform store by default. Set `FAK_CA_BUNDLE` anyway if children need the sibling variables (Node does not read the OS store on Linux). |
| **File bundle** | The common case. You have the root as a `.pem`/`.crt` and no admin rights. | `export FAK_CA_BUNDLE=/path/to/root.pem`. No OS-store install, no admin, no `-k`. |
| **Container** | fak runs in a container behind the host's proxy. | Mount the bundle read-only and set `FAK_CA_BUNDLE` to the mounted path: `-v /etc/corp/ca.pem:/etc/corp/ca.pem:ro -e FAK_CA_BUNDLE=/etc/corp/ca.pem`. Do not bake the root into the image. |
| **Cloud gateway** | The upstream is Bedrock/Vertex. | Trust and routing are separate problems: fix the certificate with one of the patterns above, then use `fak serve --stdio --policy FILE`, because no certificate makes an inert base-URL repoint adjudicate anything. |

### Getting the root as a file

If you do not have the PEM, read it off the connection your proxy is already
intercepting:

```bash
openssl s_client -showcerts -connect api.anthropic.com:443 </dev/null 2>/dev/null \
  | openssl x509 -noout -subject -issuer -enddate
```

The `issuer` on that leaf is the interceptor. `fak doctor trust` names the same CA
without OpenSSL, from the chain inside the verification failure.

More than one interceptor is normal. Concatenate the PEMs into one file — fak
widens the platform store, so adding a root never removes one:

```bash
cat netskope-root.pem corp-issuing-ca.pem > /etc/corp/ca-bundle.pem
```

## See also

- [Config bails](config-bails.md) — the `UPSTREAM_TRUST_UNVERIFIED` and
  `UPSTREAM_UNSUPPORTED` tokens, and `fak recover` for each.
- [Troubleshooting route](troubleshooting.md) — start there when you have a
  symptom rather than a token.
- [Air-gapped deployment kit](air-gapped-deployment-kit.md) — the adjacent
  posture: no egress at all, rather than intercepted egress.

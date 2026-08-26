# Corporate TLS interception and cloud upstreams (the managed-host patterns)

On a corporate laptop, a setup that works in your shell can silently stop working
inside fak. Two independent reasons, one symptom ("fak is broken"):

1. **TLS is intercepted.** A proxy re-signs every outbound connection with a private
   root, so the chain does not validate against the trust store fak is using. Nothing
   is blocked — but every runtime reports it differently and none of them says
   "interception" (`x509: certificate signed by unknown authority`,
   `SELF_SIGNED_CERT_IN_CHAIN`, `CRYPT_E_NO_REVOCATION_CHECK`).
2. **The model upstream is not a bearer API.** With `CLAUDE_CODE_USE_BEDROCK=1` (or
   Vertex) the child signs each request itself and never reads `ANTHROPIC_BASE_URL`,
   so fak's gateway repoint is inert and would adjudicate nothing.

This directory is the runnable proof that fak now **names both at launch**, with a
config-bail token, a read-only check, and a recovery plan — and that it raises
nothing at all on a host that has neither. Shipped in
[#8172](https://github.com/anthony-chaudhary/fak/issues/8172).

## Run it

```bash
examples/corporate-tls/run.sh
```

**No proxy, no certificate store, no admin rights, no `-k`, no network, no model, no
GPU, no key.** Every `fak doctor trust` run passes `--probe=false`, and every `fak
guard` run is stopped at a gate (or by an `--api-key-env` this example deliberately
leaves unset) *before* a gateway binds, a child spawns, or a credential is read. The
one optional dependency is `openssl`, used only to mint a throwaway root for witness
2; without it that witness prints `SKIP` and the rest still runs. A captured run is
in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

With an existing `fak` binary the seven offline witnesses complete in a few
seconds; the optional OpenSSL certificate mint can add a few more. The expected
exit codes, tokens, assertions, and final pass/fail result are deterministic for a
fixed `fak` binary and tool set. Temporary paths, certificate expiry timestamps,
and the names of cloud variables already present on the host can vary, and the
captured run labels those normalizations explicitly.

## What you see

Each witness prints the real `doctor`, `guard`, or `recover` output followed by
assertion lines. The script returns zero only after all assertions pass. Run
`bash examples/corporate-tls/selfcheck.sh` to exercise the separate strict-mode
regression: it forces the first normally-successful command to fail and proves the
main script stops before witness 2 or the success summary.

The seven witnesses:

| # | Host posture | Expect |
|---|---|---|
| 1 | nothing declared, no interception | exit **0**, zero findings — the check is silent on a clean box |
| 2 | `FAK_CA_BUNDLE` points at a real PEM | exit **0**; the trust store, the CA subjects, the expiry, and the child runtimes that inherit it all named |
| 3 | a *sibling* runtime (`AWS_CA_BUNDLE`) has a bundle, fak does not | exit **1**, `sibling-trust-vars` — the common case, caught with no network |
| 4 | `FAK_CA_BUNDLE` points at a file that is not a PEM | exit **2**, `reason: UPSTREAM_TRUST_UNVERIFIED` + `next: fak recover …` |
| 5 | `CLAUDE_CODE_USE_BEDROCK=1` | exit **2**, `reason: UPSTREAM_UNSUPPORTED` + the `works:` line, and **never** `STALE_CRED` |
| 6 | …plus `FAK_GUARD_ALLOW_UNSUPPORTED_UPSTREAM=1` | the waiver proceeds and says so, in the words "fak will see NONE of this session's model traffic" |
| 7 | `fak recover` for both tokens | each maps to concrete commands and notes |

Witness 5 is the one that mattered on the machine this was written on. Before the
gate, a Bedrock-routed `fak guard --probe claude` printed
`parked — STALE_CRED: waiting for a re-login to land at …\.credentials.json (poll
every 2m0s, ceiling 24h0m0s)` and hung for up to 24 hours — on a host holding a
perfectly good working credential, waiting for a re-login that could not have changed
which credential the child sends.

## Setup patterns

Four ways to land trust on a managed host. Pick one; they are alternatives, not
layers.

### 1. OS store (nothing to configure)

Your IT team installs the root machine-wide and fak validates against the platform
trust store by default — on Windows and macOS that is the OS verifier, so an
intercepted chain already validates.

```bash
fak doctor trust                 # confirm: the chain validated, anchor named
```

Still set `FAK_CA_BUNDLE` if the agent shells out to **Node** (`SELF_SIGNED_CERT_IN_CHAIN`
even with the root in the store) or to Python/curl on Linux — those read their own
variables, and declaring the bundle once derives all of them.

### 2. File bundle (the common case: no admin rights)

```bash
export FAK_CA_BUNDLE=/etc/corp/ca-bundle.pem      # or C:\corp\ca-bundle.pem
fak doctor trust
fak guard -- claude
```

fak validates against **the platform store PLUS that file** — never the file alone,
because a site routinely runs more than one interceptor and a bundle-only pool trades
one broken endpoint for every endpoint that was working. Concatenate roots freely:

```bash
cat netskope-root.pem corp-issuing-ca.pem > /etc/corp/ca-bundle.pem
```

Don't have the PEM? Read it off the connection your proxy is already intercepting —
the `issuer` on that leaf is the interceptor:

```bash
openssl s_client -showcerts -connect api.anthropic.com:443 </dev/null 2>/dev/null \
  | openssl x509 -noout -subject -issuer -enddate
```

`fak doctor trust` names the same CA with no OpenSSL, by reading the chain out of the
verification failure itself. There is no `--insecure` and no skip-verify path.

### 3. Container (mount, never bake)

```bash
docker run --rm -it \
  -v /etc/corp/ca-bundle.pem:/etc/corp/ca-bundle.pem:ro \
  -e FAK_CA_BUNDLE=/etc/corp/ca-bundle.pem \
  -e HTTPS_PROXY -e HTTP_PROXY -e NO_PROXY \
  your-image fak doctor trust
```

Mount the root read-only rather than `COPY`ing it into the image: the root rotates on
the host's schedule, not your build's, and an image that carries a site's root is an
artifact you now have to keep out of a registry.

### 4. Cloud gateway (Bedrock / Vertex)

Trust and routing are **different problems with the same symptom**. Fix the
certificate with one of the patterns above, then use the route that actually
adjudicates on this posture:

```bash
fak serve --stdio --policy examples/policy.example.json
```

fak as an MCP server the agent calls — provider-agnostic, because the agent is the
client. The capability floor, tool brokering, transcript, and refusal vocabulary all
apply whatever the model wire is. `fak guard` on a signed-request route refuses with
`UPSTREAM_UNSUPPORTED` instead, because no certificate makes an inert base-URL
repoint adjudicate anything.

Want guard's *other* properties anyway — hook floor, tool brokering, transcript,
sandbox — on a session whose model traffic fak will not see?

```bash
export FAK_GUARD_ALLOW_UNSUPPORTED_UPSTREAM=1
```

Every launch then says so on its startup report. A waiver that went quiet would be
worse than the refusal it replaced.

## What this does not do

- **It does not weaken verification, ever.** There is no `--insecure`, no
  `--skip-verify`, and no `InsecureSkipVerify` in the trust path. Witness 4 asserts
  the refusal never even offers one.
- **It does not install anything into a trust store.** `FAK_CA_BUNDLE` is read; no
  OS store is written, and nothing here needs admin rights.
- **It does not make `fak guard` adjudicate a Bedrock/Vertex session.** That is a
  property of the wire, not a setting. Pattern 4 is the supported route.
- **It does not replace a proxy allow-list.** If your proxy blocks
  `api.anthropic.com`, no bundle helps — `fak doctor trust` reports an unreachable
  endpoint as "no trust verdict available" rather than as a trust failure, precisely
  so the two stay distinguishable.

## Where this fits

- **The full page:** [`../../docs/managed-hosts.md`](../../docs/managed-hosts.md)
- **The two tokens:** [`../../docs/config-bails.md`](../../docs/config-bails.md) —
  `UPSTREAM_TRUST_UNVERIFIED`, `UPSTREAM_UNSUPPORTED`
- **The check:** `fak doctor trust [--probe=false] [--host HOST:PORT] [--json]`
- **The code:** [`../../internal/httptrust`](../../internal/httptrust) (resolve +
  assess + probe), [`../../internal/cloudroute`](../../internal/cloudroute) (route
  detection), `cmd/fak/guard_upstream_trust.go` (the two launch gates)
- **The issue:** [#8172](https://github.com/anthony-chaudhary/fak/issues/8172)

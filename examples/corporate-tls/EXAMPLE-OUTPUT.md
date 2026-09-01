# Captured run: `examples/corporate-tls/run.sh`

A real run, captured on a **corporate Windows 11 workstation** (Git Bash, two active
TLS interceptors, Claude Code routed to Bedrock) — the host class the seam exists
for. Temporary paths are normalized to `C:\Users\you\AppData\Local\Temp\tmp.EXAMPLE`
so the capture diffs cleanly across runs; nothing else is edited.

Witness 5 shows `Observed: AWS_PROFILE, AWS_REGION, CLAUDE_CODE_USE_BEDROCK` because
this host really does carry an AWS profile. On a host without one that list is just
`CLAUDE_CODE_USE_BEDROCK` — the refusal is identical either way, and `Observed`
prints variable NAMES only, never values.

## The full witness run

```console
$ FAK_BIN=./fak examples/corporate-tls/run.sh
== 1/7  a clean host: the check is silent ==
== fak doctor: upstream trust ==
trust source: none declared (FAK_CA_BUNDLE unset) — platform trust store only
[OK  ] trust-source        no corporate trust source declared; validating against the platform trust store — the correct posture on an unintercepted host
doctor: healthy (0 findings)
  PASS  exit 0 — no corporate trust declared, nothing to report
  PASS  names the platform-store posture as correct, not as a gap
  PASS  zero findings
  PASS  not one warning on an unintercepted host

== 2/7  a declared bundle: the trust store fak validates with, named ==
== fak doctor: upstream trust ==
trust source: FAK_CA_BUNDLE=C:\Users\you\AppData\Local\Temp\tmp.EXAMPLE\corp-root.pem
[OK  ] trust-source        validating against the platform trust store PLUS C:\Users\you\AppData\Local\Temp\tmp.EXAMPLE\corp-root.pem (1 certificate(s): Example Corp Root CA)
       recommend: children inherit the same file via NODE_EXTRA_CA_CERTS, AWS_CA_BUNDLE, CURL_CA_BUNDLE, SSL_CERT_FILE, REQUESTS_CA_BUNDLE, GIT_SSL_CAINFO
[OK  ] bundle-expiry       soonest certificate expiry 2036-08-16T20:33:08Z
[OK  ] child-runtime-trust every child runtime inherits fak's trust source (NODE_EXTRA_CA_CERTS, AWS_CA_BUNDLE, CURL_CA_BUNDLE, SSL_CERT_FILE, REQUESTS_CA_BUNDLE, GIT_SSL_CAINFO)
doctor: healthy (0 findings)
  PASS  exit 0 — a bundle that loads is a healthy host, not a finding
  PASS  widen, never replace — the platform store is still in the pool
  PASS  the CA subject the bundle actually carries
  PASS  one declaration, derived for every child runtime
  PASS  and it says which child runtimes would NOT inherit it

== 3/7  the no-network signal: a sibling runtime knows something fak does not ==
== fak doctor: upstream trust ==
trust source: none declared (FAK_CA_BUNDLE unset) — platform trust store only
[OK  ] trust-source        no corporate trust source declared; validating against the platform trust store — the correct posture on an unintercepted host
[WARN] sibling-trust-vars  AWS_CA_BUNDLE already point at a corporate CA bundle on this host, but FAK_CA_BUNDLE is unset — fak is the only runtime here still on the platform default
       recommend: export FAK_CA_BUNDLE=C:\Users\you\AppData\Local\Temp\tmp.EXAMPLE\corp-root.pem (the same file AWS_CA_BUNDLE already uses)
doctor: 1 finding(s)
  PASS  exit 1 — a finding exits 1, so CI can route on it
  PASS  the sibling row
  PASS  names the runtime that already knew
  PASS  and hands over the exact same file to declare

== 4/7  a declared bundle that does not load: UPSTREAM_TRUST_UNVERIFIED ==
fak guard: a corporate CA bundle was declared and fak cannot validate with it — refusing to launch a session whose every upstream call would fail on a chain fak cannot verify
  reason: UPSTREAM_TRUST_UNVERIFIED
  config: env FAK_CA_BUNDLE = C:\Users\you\AppData\Local\Temp\tmp.EXAMPLE\not-a-pem.txt  (want: a readable PEM file containing at least one CERTIFICATE block)
          file C:\Users\you\AppData\Local\Temp\tmp.EXAMPLE\not-a-pem.txt = bundle contributed no usable certificates: C:\Users\you\AppData\Local\Temp\tmp.EXAMPLE\not-a-pem.txt
  check:  fak doctor trust
  next:   fak recover UPSTREAM_TRUST_UNVERIFIED --set path=C:\Users\you\AppData\Local\Temp\tmp.EXAMPLE\not-a-pem.txt
  PASS  exit 2 — refused before the gateway binds or the child spawns
  PASS  the closed-vocabulary token
  PASS  bound to its own recovery plan, with the path filled in
  PASS  and to the read-only command that shows what fak saw
  PASS  no skip-verify escape is ever offered

== 5/7  a request-signed cloud upstream: UPSTREAM_UNSUPPORTED ==
fak guard: the wrapped agent is routed to AWS Bedrock (SigV4-signed requests) by CLAUDE_CODE_USE_BEDROCK, so it authenticates each request itself and ignores ANTHROPIC_BASE_URL — fak's gateway repoint cannot take effect and the session would be adjudicated by nothing. Observed: AWS_PROFILE, AWS_REGION, CLAUDE_CODE_USE_BEDROCK
  reason: UPSTREAM_UNSUPPORTED
  config: env CLAUDE_CODE_USE_BEDROCK = on  (want: unset, to route this session through fak's gateway with a bearer credential)
          env ANTHROPIC_BASE_URL = (would be injected, and ignored by the child)
          env FAK_GUARD_ALLOW_UNSUPPORTED_UPSTREAM = (unset)  (want: 1 to launch anyway with the model traffic unadjudicated)
  check:  fak doctor trust
  next:   fak recover UPSTREAM_UNSUPPORTED
  works:  fak serve --stdio --policy FILE   (fak as an MCP server the agent calls — provider-agnostic, so the capability floor is enforced whatever the model wire is)
  PASS  exit 2 — refused at the posture, before any credential is read
  PASS  the closed-vocabulary token
  PASS  names the mechanism that went inert
  PASS  and the route that DOES work on this posture
  PASS  never a credential refusal for a credential that is fine
  PASS  and never advice that cannot apply here

== 6/7  the waiver is loud, not silent ==
fak guard: UPSTREAM_UNSUPPORTED — proceeding under FAK_GUARD_ALLOW_UNSUPPORTED_UPSTREAM=1: the hook floor, tool brokering, transcript, and sandbox still apply, but fak will see NONE of this session's model traffic.
fak guard: --api-key-env FAK_CORPORATE_TLS_EXAMPLE_KEY is set but that env var is empty — export it for API billing, drop the flag to use your Claude Pro/Max subscription, or pass --anthropic-oauth to force the subscription.
  PASS  exit 2 — the waiver reaches the deliberately empty API-key guardrail
  PASS  the waiver announces itself on every launch
  PASS  in the words that matter
  PASS  the refusal is downgraded, so the bail block is gone
  PASS  the launch then stops on this example's own guardrail — nothing was spawned

== 7/7  both tokens have a recovery plan ==
recover UPSTREAM_TRUST_UNVERIFIED (dry-run)
reason: a corporate CA bundle was declared and fak is not validating with it, so every outbound call would fail on a chain fak cannot verify
commands:
  fak doctor trust
    # report the declared trust source, the CA subjects it carries, and which child runtimes would not inherit it (read-only)
  openssl x509 -noout -subject -issuer -enddate -in <path>
    # confirm the file really is a PEM certificate and has not expired
note: the bail's `file` line says which half failed: the bundle could not be READ (a wrong path, or a path relative to a working directory that is not yours under an MCP launcher), it read but held no CERTIFICATE block (a DER/.crt export, or the ticket text rather than the attachment), or the platform trust store fak must widen was unavailable
note: fak always ADDS your root to the system pool, never replaces it — that is why an unavailable system pool is a refusal and not a fallback: validating against your bundle alone would break every endpoint the bundle does not cover, which an operator would read as fak breaking their network
note: export the root as PEM (`-----BEGIN CERTIFICATE-----`), not DER: on Windows, certmgr's "Base-64 encoded X.509" is the right export option and "DER encoded binary" is not
note: a bundle may hold several roots; a site with more than one interceptor usually needs all of them concatenated, because the CA in front of the model endpoint is often not the CA in front of the cloud control plane
note: there is deliberately no --insecure or skip-verify escape: interception is a trust problem, and a governance tool that normalizes unverified TLS has given up the property it exists to assert
note: once it loads, fak derives NODE_EXTRA_CA_CERTS / AWS_CA_BUNDLE / CURL_CA_BUNDLE / SSL_CERT_FILE / REQUESTS_CA_BUNDLE / GIT_SSL_CAINFO for children from the same file, so the wrapped agent and its hooks stop needing their own answer
  PASS  UPSTREAM_TRUST_UNVERIFIED routes to the read-only check
  PASS  and explains widen-never-replace where the operator is standing
recover UPSTREAM_UNSUPPORTED (dry-run)
reason: the wrapped agent is routed to a request-signed cloud gateway (Bedrock SigV4 / Vertex ADC), so fak's base-URL repoint is inert and the gateway would adjudicate nothing
commands:
  fak serve --stdio --policy <policy>
    # the path that DOES work on this posture: fak as an MCP server the agent calls, provider-agnostic because the agent is the client
note: this is not a credential failure. Your cloud credential is fine — the problem is that a signed-request child never reads ANTHROPIC_BASE_URL, so pointing it at fak's gateway changes nothing and fak would see none of the traffic it is supposed to adjudicate
note: the supported route on this posture is the MCP server: register `fak serve --stdio --policy FILE` with the agent and the capability floor is enforced on every tool call, whatever the model wire is
note: docs/supported/clouds.md marks Bedrock Partial for exactly this reason — the native path needs SigV4 or a Bedrock bearer key, not an endpoint swap
note: to run the guard anyway for its OTHER properties (hook floor, tool brokering, transcript, sandbox) with the model traffic unadjudicated, set FAK_GUARD_ALLOW_UNSUPPORTED_UPSTREAM=1; it proceeds and says so loudly on every launch, which is the intended escape and not a workaround to hide
note: to route this session through fak's gateway instead, unset the cloud selector (CLAUDE_CODE_USE_BEDROCK / CLAUDE_CODE_USE_VERTEX) and give the agent a bearer credential fak can front
  PASS  UPSTREAM_UNSUPPORTED routes to the MCP path
  PASS  and says plainly what it is not

================================================================
corporate-tls: 33 passed, 0 failed
corporate-tls: PASS

  Nothing here needed a proxy, a certificate store, admin rights, or -k. A clean
  host raised nothing; a declared bundle was named along with the CA subjects it
  carries and the child runtimes that inherit it; a sibling runtime's own bundle
  was enough to catch the gap with no network at all; and the two postures fak
  cannot silently absorb — a bundle that will not load, and a request-signed cloud
  route — were refused at launch with a token, a check, and a recovery plan.
```

## What was elided from the capture

Elided from witnesses 4–6 above: each `fak guard` invocation also prints its
`fak guard: response-profile {…}` JSON block before the gate runs. It is unrelated
to this example and is dropped here only to keep the capture readable — the live run
shows it.

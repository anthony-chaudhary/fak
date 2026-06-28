# Audit: `fak guard` "stale binary?" and the SECRET_EXFIL quarantine banner

Date: 2026-06-28. Scope: two reported symptoms of `fak guard -- claude`:

1. "It still seems like old versions of `fak guard` are running."
2. The repeated banner
   `[fak] quarantined 1 inbound tool result(s) before the model read them: tool_result (SECRET_EXFIL). The content was paged out ...`
   should be "more permissive, useful, or clear by default."

## Finding 1 — the banner is current HEAD behavior, not a stale binary

The SECRET_EXFIL banner is produced by code that is live in HEAD, so seeing it does
not imply an old build:

- `internal/normgate/normgate.go` registers at rank 5 and is **default-on**
  (`FAK_NORMGATE != "off"`). When `canon.Scan` finds a secret in an **inbound** tool
  result it quarantines the bytes with reason `abi.ReasonSecretExfil` ("SECRET_EXFIL").
- `internal/gateway/messages.go` `resultAdmissionNote` rendered that raw reason code
  verbatim into the `[fak] quarantined ...` text, on both the buffered and streaming
  Anthropic paths.
- That banner string is unchanged since the v0.30.0 public release (commit 1029e37c,
  2026-06-21). So the behavior the user sees is in the current tree.

### Why "is my guard current?" was genuinely hard to answer

`fak version` printed only `appversion.Current()`, which resolves the version by
walking **up from the current directory** to the nearest `VERSION` file (then
`$FAK_APP_VERSION`, then the `-ldflags` `BuildVersion`). Consequence: a **stale `fak`
binary run from inside an up-to-date checkout reports the tree's version** (e.g.
`0.34.0`) — it cannot reveal that the binary itself is old. The version line looks
current even when the running binary is not, which is exactly the "old guard seems to
be running, but version says current" confusion.

**Shipped:** `fak version` now also prints the binary's embedded build provenance from
`runtime/debug.ReadBuildInfo()` — the VCS revision (short), commit time, and a
`+uncommitted` flag — e.g.

```
0.34.0
build: 69954ee0ec8f +uncommitted  (committed 2026-06-28T15:21:25Z)
go: go1.26.3  windows/amd64
```

The stamp travels with the binary, so it is a reliable "is the `fak`/guard I'm running
current?" check that a stale binary cannot fake. Line 1 is still `appversion.Current()`
verbatim, so anything parsing line 1 is unaffected. (There is no version-staleness check
inside `fak guard` itself; it always compiles the floor in and uses
`appversion.Current()` for the gateway `Version`.)

## Finding 2 — the detector over-fires on benign content (witnessed)

`internal/canon/canon.go` is the shared de-obfuscating detector. Two rules drive the
false positives:

- Secret pattern #10, the keyword-proximity rule
  `(?i)(token|secret|api[_-]?key|bearer|password|passwd)["'\s:=]{1,4}[A-Za-z0-9+/_-]{20,}`,
  matches a key-like keyword followed by any 20+ char run — including placeholder and
  example values in docs, `.env.example`, and source.
- The injection marker list includes the bare word `exfiltrate`, which fires on any
  document or source that merely discusses exfiltration.

This was reproduced live while auditing: reading the kernel's own security source
(`canon.go`, `normgate.go`, `secretgate.go`) raised
`tool_result (TRUST_VIOLATION); tool_result (SECRET_EXFIL)` — the detector trips on its
own pattern/marker text. The dominant real-world trigger is the agent reading its own
tree.

### Shipped: a clear, useful, non-alarming banner (default)

`resultAdmissionNote` was rewritten and `quarantineReasonPhrase` added so the banner now:

- explains, in plain language, what was held and why;
- frames it as a routine safety precaution, **not an agent error** ("continue normally");
- names the false-positive risk on placeholder/example credentials and own-source reads;
- says the bytes were paged out (not lost) and are retrievable via the page-in gate;
- maps each closed-vocabulary code (`SECRET_EXFIL`/`RESULT_SECRET_DISCOVERED`,
  `TRUST_VIOLATION`, `OVERSIZE`) to a human line, falling back to the raw name for an
  unrecognized code.

The structured per-result verdicts still ride the machine-readable `fak` response
extension; only the human/agent-facing prose changed.

## Not changed here — the "more permissive detection" decision is the maintainer's

Detection itself (`canon.Scan`) and the seal-vs-retrievable policy were left unchanged,
because loosening them touches a **tested security contract**:

- `internal/normgate/normgate_test.go` `TestTrustedLocalSecretStillQuarantines`
  explicitly asserts that a secret from a **trusted-local** read still seals. normgate's
  own doc comment states the stance: "Secrets quarantine regardless of source (a leaked
  credential is worth holding even from a local read)."

Options if a maintainer wants the default to be genuinely more permissive, roughly in
increasing blast radius:

1. **Knobs already exist (no code):** `FAK_NORMGATE=off` disables the whole
   normalize-and-rescan floor (also drops injection protection); `FAK_SECRETGATE` opts
   into the on-discovery secret rung (rank 4) which carries the correct inbound reason
   `RESULT_SECRET_DISCOVERED` and a posture (admit_and_log / quarantine / fail_closed).
2. **Reason correctness (small):** on the inbound result path, report
   `RESULT_SECRET_DISCOVERED` (the #152 on-discovery code) instead of `SECRET_EXFIL`
   (the egress verdict). Touches `normgate.go:138`, `ctxmmu.go:480`, `recall.go`, and the
   tests that pin `SECRET_EXFIL`.
3. **Placeholder suppression (medium):** in `canon.Scan`, after a secret match, suppress
   when the matched span is an obvious placeholder/example (`your-...-here`, `xxxx`,
   `<...>`, `REDACTED`, `changeme`, `example...`). Purely subtractive of false positives
   (a real credential is never a placeholder); preserves the combined-regex equivalence
   proof since it is a separate stage.
4. **Provenance-aware secrets (policy reversal):** treat a secret from a trusted-local
   read as a retrievable Transform rather than a sealed Quarantine, symmetric with how
   injection is already handled. This **reverses the tested secrets-are-absolute-by-source
   stance** above, so it needs an explicit decision and a test update.
5. **Drop the lone-word `exfiltrate` injection marker (small, weakens injection):** the
   other markers are multi-word injection phrases; a bare technical word is a poor signal
   and fires on security prose.

Recommendation: (2) + (3) give the biggest clarity/false-positive win without reversing a
stated security property; (4) only with maintainer sign-off.

---
title: "Audit: the SECRETEXFIL quarantine blinds a legitimate dev — five whys"
description: "How a default secret-shape quarantine blinded a legitimate dev by holding a whole tool result out of context, plus the warn-first redact fix."
---

# Audit: the SECRET_EXFIL quarantine blinds a legitimate dev — five whys + warn-first fix

Date: 2026-07-07. Scope: a real session (`e2daa325`, in `work/Benchmark`) that
livelocked while dogfooding the Confluence-report feature over the private bridge Slack
control-bridge to the `da33` lab box. The reported symptom was "DOS cycled a livelock
badly on genuine secret usage." This note records what actually happened, the systemic
root cause, and the durable warn-first fix that shipped with it.

## What actually happened (not what the symptom said)

DOS did **not** refuse the developer's secret usage. Across 230 tool calls only two
real guard refusals fired — one `POLICY_BLOCK` on an `rm -rf`, one incidental
`OFF_TRUNK`. The developer built the bridge, reached the internal bridge live, saved the Confluence
PAT on-box, and got most of the way to a working dogfood.

The livelock was a two-gate interaction at the very end (`repeat=9..11`, `fuse=armed`):

1. Every private bridge readback echoed the remote command's combined stdout+stderr verbatim
   into the Slack transcript, and that text carried a credential shape (a PAT, a token).
2. `internal/normgate` (default-on) matched the secret shape and **held the ENTIRE
   readback out of context** with reason `SECRET_EXFIL` — the whole legitimate output
   gone because one span looked like a credential.
3. Because a secret-class quarantine **never pages back** (`normgate.PageIn` re-screens
   and refuses any secret-bearing bytes, unconditionally), the model never saw the
   readback. It could not tell "still running" from "wedged", so it re-issued the same
   readback — identical quarantined outcome — until the result-side livelock detector
   armed. The result-side livelock is advisory-only (by design), so the loop continued
   until the user interrupted 11 times.

## Five whys — the issue behind the issue

1. **Why livelock?** The model re-read a bridge result it would never receive.
2. **Why never receive it?** normgate held the whole result out because a span matched a
   secret pattern, and the page-in gate refuses to return secret-bearing bytes.
3. **Why hold a whole legitimate dev result for one span?** The default secret posture is
   the absolute seal — the same maximally-strict behavior for an interactive dev session
   as for an unattended prod batch.
4. **Why is the strictest posture the default for a dev session?** The guard was designed
   security-first: "a credential in a tool result is an exfil event, hold it absolutely."
   The legitimate-dev case (a dev handling their OWN credential on their OWN box) was
   treated as acceptable collateral.
5. **Root.** The guard's default stance **conflated "a secret shape was detected" with
   "this is an attack."** Every downstream harm — the absolute hold, the never-page-back,
   the advisory result-livelock with no exit, the bridge echoing plaintext — is a
   consequence of that one conflation. The guard defaulted to treating the developer as
   the adversary.

Note this was flagged as an open maintainer decision in the earlier audit
[AUDIT-fak-guard-secret-exfil-2026-06-28](AUDIT-fak-guard-secret-exfil-2026-06-28.md)
("the tested secrets-are-absolute-by-source stance as an explicit maintainer decision").
This note is the follow-through: the stance is now warn-first by default, strict by
policy.

## The fix — warn-first default, strict is opt-in policy

The design principle: **defaults permissive for dev and legitimate use; super-security
enforcement is a policy an operator opts into.** Three coordinated changes.

### 1. normgate: redact-in-place by default (root fix)

`internal/canon/redact.go` (`RedactSecrets`, `RawSecretComplete`) masks a credential
span IN PLACE — `[redacted:secret:NB]` — keeping every other byte, using the SAME
`combinedSecret` matcher and placeholder filter as detection (so the two can never
drift). A redacted body re-screens clean (verified), so it needs no held handle and no
page-in gate.

`internal/normgate/normgate.go` `admitSecret` routes a detected secret:

- **fail_closed posture (OPT-IN)** → seal the whole result (`SECRET_EXFIL`), the
  pre-warn-first behavior. This is the "super-security" policy for untrusted / unattended
  contexts, read from `adjudicator.Default.SecretPolicy()` the same way secretgate reads
  it.
- **obfuscated secret** (caught only on a de-obfuscated view, no raw span) → seal too:
  the in-place redactor cannot reach it, and an obfuscated secret is the adversarial case
  the permissive path must not cover.
- **otherwise (DEFAULT)** → redact the span(s) in place, keep the rest of the output in
  context (`SECRET_REDACTED` Transform). No paged-out stub → nothing to re-read → the
  livelock is structurally impossible, and the developer sees their own output with only
  the credential masked.

This flips the package's prior "secrets quarantine regardless of source" policy: strict
is now a policy choice, permissive is the default.

### 2. gateway: a warn note, not a held-out banner

`internal/gateway/messages.go` `resultAdmissionNote` now emits a one-line WARN for a
`SECRET_REDACTED` transform ("masked N credential-shaped spans ... the rest of the output
is intact and in context ... to hold the whole result instead, set the fail_closed secret
posture"). It never reads as "held out of context" and never baits a re-read. The
result-side livelock stays advisory (deliberately NOT a hard-stop — a session-killing
stop would blind/kill a legitimate dev faster, the opposite of the goal). A terminal stop
remains available only under the opt-in strict posture path.

### 3. private bridge: secret-safe readback at the source (fak-private)

`fak-private/tools/private bridge`: the `run/exec/cmd` verbs now base64-frame the remote
command's combined output before it crosses the Slack transcript (`Bridge.EncodeOutput`,
default on for those verbs; `-raw` opts out), mirroring the already-safe `ReadFile` path.
A secret echoed by a command never appears in cleartext on the wire — closing both the
fak-side quarantine trigger AND the Slack-transcript exfil hole. The sentinels/RC stay
plaintext so `extractBlock`/`extractRC` still anchor; only the payload is encoded and
decoded client-side via the shared `decodeB64Block`.

## Tests

- `internal/canon/redact_test.go` — mask keeps the rest + re-screens clean; structural
  placeholders untouched; multiple secrets masked; `RawSecretComplete` routing.
- `internal/normgate/normgate_test.go` — plaintext secret REDACTED by default; SEALED
  under fail_closed; obfuscated secret seals even by default; the held-ledger bound tests
  opt into the seal posture.
- `internal/gateway/messages_result_note_test.go` — redaction warn is a one-line WARN
  (not a held-out banner); a mixed turn surfaces both banners.
- `fak-private/tools/private bridge/internal/encode_readback_test.go` — an EncodeOutput
  readback hides a secret on the wire and decodes clean; the plaintext path is unchanged.

## What is deliberately NOT done

- The result-side livelock is **not** converted to a hard stop. Under warn-first the
  correct fix removes the blinding (so the loop never forms), not a faster session kill.
- The zero-value `SecretQuarantine` posture token keeps its meaning (seal) for
  compatibility; the permissive default lives in normgate's policy decision, and
  `fail_closed` is the explicit opt-in seal.

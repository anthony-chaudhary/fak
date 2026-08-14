---
title: "Centralized fak Policy Configuration for Teams"
description: "Configure one signed fak capability-floor policy for a team, enroll developer machines, verify effective policy, and preserve local tightening without allowing weaker overrides."
---

# Centralized policy for a team

`fak manage` gives one developer a capability floor on one machine. This page is about
the next problem: you have twenty developers, each with their own `.fak/guard/allow.json`,
and you want one answer to "what is every agent on this team allowed to do?"

The org policy plane is that answer. A signed manifest published by your org sets the
floor for every enrolled machine. Local operators can still tighten it. They cannot
loosen it past what you granted.

**Status:** the plane is built and inspectable end to end (`fak enroll`, `fak org status`),
and the composed floor is not yet installed into a running `fak manage` session. See
[What is wired today](#what-is-wired-today) before you plan a rollout.

## Who this is for

Use it when more than one person runs agents against the same codebase or the same
production surface, and you have caught yourself asking any of these:

- Did everyone actually get the new deny rule, or just the people who read the Slack message?
- Which machine allowed the agent to run `terraform apply`, and who approved that?
- If our policy endpoint goes down, does the fleet fall open?

If you are one developer on one laptop, you do not need this. `fak manage` and
[policy-guide.md](policy-guide.md) cover you.

## The rule that makes it safe

Four channels can speak about any capability knob. They are ordered:

```
compiled-in FROZEN floor  >  central org manifest  >  operator overlay  >  agent-self
```

Read "greater than" as **caps**, not as **overrides**. That distinction is the whole
design, and it cuts both ways:

| Knob class | What central can do | What the local operator can do |
|---|---|---|
| `FROZEN` | Nothing. The compiled floor owns it. | Nothing. |
| `RATCHET` | Tighten it fleet-wide. | Tighten it further. |
| `GATED_WIDEN` | Widen it, up to the compiled cap. | Tighten below the grant. |

Two consequences worth stating out loud, because teams get them backwards:

**A local operator can always be stricter than you.** If a developer denies a tool your
manifest allows, they keep that deny. Your manifest is a floor for the fleet, not a
ceiling on anyone's caution.

**Nobody can widen past the channel above them.** A machine cannot re-admit a tool your
manifest withholds, and your manifest cannot reach the hardwired refusals (the egress
SSRF floor, the reversibility gate, the structural danger rules) that ship in the binary.

The full per-class truth table lives in
[`internal/policy/org_precedence_test.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/policy/org_precedence_test.go),
which is an executable spec rather than a description of one.

## Setting it up

### 1. Sign a manifest

The envelope is a `fak-policy/v1` manifest wrapped in a signed header. Your signing key
never leaves your CI or your HSM; only the public half is distributed.

```json
{
  "issuer": "acme-corp",
  "alg": "ed25519",
  "version": 9,
  "not_before": 1786000000,
  "expires": 1786086400,
  "sig": "<base64 Ed25519 over the canonical bytes>",
  "body": { "version": "fak-policy/v1", "allow": ["search_web", "deploy_stage"] }
}
```

`version` is a monotonic anti-rollback counter. Each machine remembers the highest it
has accepted and refuses anything lower, so a captured older, more permissive manifest
cannot be replayed to widen a box back open.

Serve the signed bytes from any HTTPS endpoint your fleet can reach.

### 2. Enroll each machine

Enrollment is opt-in and pins the trust anchor. A machine that never runs this behaves
exactly as it does today.

```bash
fak enroll --org https://policy.acme.example/fak \
           --root-key ./acme-org.pub \
           --device "$(hostname)"
```

The root key is operator-supplied on purpose. Fetching a key from the same endpoint that
serves the policy would authenticate the endpoint to itself, which is the attack pinning
exists to stop. Distribute the public key through whatever channel you already trust for
your own CA bundle.

Re-pinning onto a different org is refused rather than done silently. `fak enroll --revoke`
is the sanctioned way back to un-enrolled.

### 3. Verify it landed

```bash
fak org status
```

This is the screen to check after a rollout, and it is offline by default. Pass `--pull`
when you want it to fetch a fresh envelope.

```
org policy plane: FRESH — a verified org manifest is in force
  org:        https://policy.acme.example/fak
  issuer:     acme-corp
  device:     node-a
  version:    9
  freshness:  fresh (verified just now) (refuse-to-widen after 12h0m0s)
  enrollment: /home/dev/.config/fak/org-enrollment.json

capability floor: built-in guard floor (fak manage --dump-policy to see it)
  operator overlay: /repo/.fak/guard/allow.json
  central widened: added_allow=deploy_stage
  central tightened: added_deny=curl
  operator CLAMPED to the central grant: added_allow=deploy_prod

per-knob provenance (21 knobs)
  Allow                      GATED_WIDEN  central            WIDENED
  Deny                       RATCHET      central
  EgressBlockHosts           RATCHET      operator-overlay
  Posture                    GATED_WIDEN  compiled-in
  ...
```

Four things to read off it:

- **Posture.** `FRESH` and `LAST_GOOD` both mean your manifest is in force. `FLOOR` means
  the machine is refusing to widen and running on the compiled floor alone. `INERT` means
  the machine is not enrolled.
- **The clamp line.** `operator CLAMPED to the central grant` names a local overlay that
  asked for more than your manifest allows. The knob is rolled back to your value and the
  ask is reported, so a developer who thinks they enabled something learns that they did not.
- **Provenance.** Each knob names the channel that owns its running value. That is the
  field a developer needs to know whether to edit their own overlay or file a ticket with you.
- **Refusals.** `central REFUSED` means your manifest reached at a knob no central manifest
  may move. It changed nothing, and it is worth investigating.

Add `--json` for a machine-readable form with the same fields. Every key is always
present, so a consumer never has to read an absent key as "no central authority".

## What happens when your endpoint goes down

Nothing opens up. The four postures are exhaustive and none of them fails open:

| Posture | Meaning | Does your manifest apply? |
|---|---|---|
| `inert` | Not enrolled | No — the plane does nothing |
| `fresh` | Fetched and verified now | Yes |
| `last_good` | Source failed, cache still inside both freshness bounds | Yes |
| `floor` | Refusing to widen | No — compiled floor only |

A verified manifest keeps applying from cache for up to 12 hours after the endpoint stops
answering. Past that, the machine drops to the compiled floor. That bound is a compiled-in
constant: no manifest, flag, or environment variable can extend it, because a knob that
let the fetched document lengthen its own tolerated staleness would make "stale"
unreachable by construction.

Freshness is measured against the local clock at the moment the machine verified the
envelope, not against a timestamp inside the envelope. An attacker replaying a captured
document cannot restate when you received it.

## Auditing what your org handed out

Every capability your manifest widens is written to the session journal as a
`CAPABILITY_GRANT` row carrying the knob, the old and new values, the amendment class,
your issuer, and the envelope version:

```
knob=Allow  direction=widen  class=GATED_WIDEN  new=deploy_stage
channel=central  actor=acme-corp  source=acme-corp@v9
```

Issuer and version are both recorded because the issuer alone cannot identify which of
your manifests handed out a capability. Tightenings are not journaled as grants; a grant
records a loosening, and a ratchet needs no provenance trail to justify it.

## What is wired today

Being precise about the seam, because a rollout plan depends on it:

- `fak enroll` pins, shows, and revokes the anchor. **Shipped.**
- Pull, verify, cache, age out, refuse to widen. **Shipped** (`internal/policy/orgpull.go`).
- The precedence fold and the composed floor. **Shipped** (`internal/policy/orgcompose.go`).
- `fak org status`. **Shipped.**
- Installing the composed floor into a running `fak manage` session. **Not yet.**
  `loadGuardCapabilityFloor` still assembles compiled-in → operator with no central stage.

So today `fak org status` reports the floor the lattice *would* produce. Use it to
validate a manifest, an enrollment, and a rollout plan. Hold off on treating a central
deny as enforced on the guarded session until the launch path lands.

## See also

- [policy-guide.md](policy-guide.md) — authoring the manifest body that goes inside the envelope
- [security.md](security.md) — the threat model the trust anchor sits in
- [hosted-control-plane.md](hosted-control-plane.md) — where the managed version of this is heading

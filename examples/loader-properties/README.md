# fak kernel — policy-loader safety properties (fail-loud · replace-not-merge · round-trip)

**A misconfigured policy must fail at *load* time, not at the first bad call in
production.** That is the difference between "the file was ignored and the agent ran with
default permissions" (bad) and "the file failed validation and the kernel refused to
start" (good). [`POLICY.md` "Safety properties of the
loader"](../../POLICY.md#safety-properties-of-the-loader) documents three structural
guarantees of `fak`'s policy loader; this demo is the runnable witness of all three, plus
the empty-manifest case.

```
  fak policy --check <manifest>          # the loader, in isolation: parse a manifest
        │                                # exactly as `fak serve --policy` would
        ▼
   parses clean?  no ──▶ FATAL: named error on stderr, exit non-zero   (fail-loud)
        │ yes
        ▼
   print the COMPLETE admitted floor on stdout, exit 0   (replace-not-merge: what you
        │                                                  see is the whole floor)
        ▼
   --dump | --check round-trips to the identical floor   (round-trip stable)
```

The loader is a **pure function of the manifest bytes** — no model, no server, no network,
no GPU. Every witness below is one `fak policy` invocation with a deterministic exit code.
Expected runtime: the full loader-property run completes in seconds.

## Run it

```bash
./examples/loader-properties/run.sh                  # build/locate fak, run all five witnesses
FAK_BIN=/path/to/fak ./examples/loader-properties/run.sh   # use a prebuilt binary instead of building
```

`run.sh` gates its exit code on all five witnesses (CI-usable): it FAILS if any bad
manifest is accepted, if the round-trip isn't exact, or if the empty floor isn't warned.
A full captured run is in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

Windows users: run the `.sh` launcher from WSL or Git Bash, or invoke the `fak policy
--check` commands below directly (they work verbatim in PowerShell with a prebuilt
`fak.exe`).

## The three properties + the empty floor

### 1. FAIL-LOUD on config errors

A malformed manifest, an unknown reason, an unknown posture, or an unknown JSON field is a
**fatal load error** — `fak` does not silently fall back to a more permissive default. Each
case exits non-zero with a *named* error:

| witness | manifest | command | result |
|---|---|---|---|
| unknown JSON field | [`bad-unknown-field.json`](bad-unknown-field.json) (`"allows"` typo for `"allow"`) | `fak policy --check examples/loader-properties/bad-unknown-field.json` | exit 1 · `invalid manifest: json: unknown field "allows"` |
| unknown deny reason | [`bad-deny-reason.json`](bad-deny-reason.json) | `fak policy --check examples/loader-properties/bad-deny-reason.json` | exit 1 · `unknown deny reason(s): …; valid reasons: …` |
| unknown posture | [`bad-posture.json`](bad-posture.json) | `fak policy --check examples/loader-properties/bad-posture.json` | exit 1 · `unknown posture "fail_open_please" (want fail_closed\|admit_and_log)` |
| malformed JSON | [`bad-malformed.json`](bad-malformed.json) (truncated) | `fak policy --check examples/loader-properties/bad-malformed.json` | exit 1 · `invalid manifest: unexpected EOF` |

The first one is the load-bearing case: the JSON decoder runs with
`DisallowUnknownFields` (`internal/policy/policy.go` `ParseManifest`), so a one-character
typo — `"allows"` for `"allow"` — is a hard error, **not** a key silently dropped that
would leave `allow` empty and the floor unexpectedly permissive-by-default. That silent
drop is exactly the failure mode this property exists to make impossible.

### 2. REPLACE, NOT MERGE — and the empty `{}` floor

A loaded manifest **is** the whole floor; it replaces the default, it is not merged into
it. So `--dump` is how you get the *complete* default to edit from — you never lose a
baked-in protection by omission, because there is no "inherited" layer underneath your
file.

The extreme of replace-not-merge is the empty manifest `{}`. Because nothing is merged in,
it allows **nothing** — the maximally paranoid floor where every call resolves to
`DEFAULT_DENY`. It is *valid* (the safe default should never be a parse error), but
`--check` accepts it (exit 0) with an explicit warning so you never deploy a dead floor by
accident:

```bash
$ fak policy --check examples/loader-properties/empty.json
OK  examples/loader-properties/empty.json  (manifest valid; every deny cites a closed-vocabulary reason)
…
NOTE: nothing is affirmatively allowed — this is the fail-closed
empty floor; EVERY call resolves to DEFAULT_DENY.
```

And it behaves as warned — any call against it is denied:

```bash
$ fak preflight --policy examples/loader-properties/empty.json --tool read_file --args '{}'
verdict=DENY reason=DEFAULT_DENY by=monitor
```

### 3. ROUND-TRIP STABLE

`fak policy --dump` emits the built-in default as a manifest; feeding it back through
`fak policy --check` parses to the **identical** floor:

```bash
$ fak policy --dump > /tmp/default.json
$ fak policy --check /tmp/default.json
OK  /tmp/default.json  (manifest valid; every deny cites a closed-vocabulary reason)
…
```

**Why this matters:** `--dump` is the on-ramp for property 2 — it's how an adopter gets
the complete default floor to edit. A future `--dump` change that broke round-trip (emitted
a manifest `--check` parsed *differently*) would silently change every floor edited from a
dump, with no error to catch it. `TestRoundTrip` (`internal/policy/policy_test.go`) pins
the exactness so that regression can't ship.

### A note on the round-trip command

POLICY.md writes the round-trip as a pipe — `fak policy --dump | fak policy --check`. That
is conceptual shorthand: on v0.34.0 the loader is `fak policy --check <FILE>` and it reads
a **file argument**, not stdin (a bare `--check` with no file blocks waiting on input). The
runnable round-trip therefore dumps to a temp file and checks the file, which is what
[`run.sh`](run.sh) does. The guarantee POLICY.md states is exactly the one demonstrated;
only the plumbing differs from the one-liner.

## Honest scope

This demo does not claim every policy is secure because it parses. It proves loader
properties only: fail-loud malformed inputs, replace-not-merge semantics, and round-trip
stability of the dumped default floor.

## Files

| file | what it is |
|---|---|
| [`run.sh`](run.sh) | one-command witness: build `fak` (or use `$FAK_BIN`) → run all five `policy --check` cases → check exit codes/errors → exit code |
| [`bad-unknown-field.json`](bad-unknown-field.json) | `"allows"` (typo for `"allow"`) → `unknown field` refuse |
| [`bad-deny-reason.json`](bad-deny-reason.json) | a `deny` reason outside the closed vocabulary → refuse |
| [`bad-posture.json`](bad-posture.json) | `"posture": "fail_open_please"` → refuse |
| [`bad-malformed.json`](bad-malformed.json) | truncated JSON → `invalid manifest` refuse |
| [`empty.json`](empty.json) | `{}` — valid, but the warned maximally-paranoid floor |
| [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md) | a full captured run on v0.34.0, every exit code + error message |

Related: [`POLICY.md` "Safety properties of the
loader"](../../POLICY.md#safety-properties-of-the-loader) and "The closed refusal
vocabulary"; [`CLAIMS.md`](../../CLAIMS.md) (the load-bearing, fail-loud capability floor);
[`internal/policy/`](../../internal/policy/) — the loader and its green tests
(`TestRoundTrip`, `TestLoadedPolicyIsLoadBearing`, `TestUnknownDenyReasonRejected`,
`TestUnknownPostureRejected`);
[`../self-modify-floor/`](../self-modify-floor/README.md) and
[`../witness-gate/`](../witness-gate/README.md) (sibling floor demos).
Issue [#201](https://github.com/anthony-chaudhary/issues/201) is the bug about one of these
properties (unknown posture) failing; on v0.34.0 it is resolved — witness 2c here carries
the fixed behavior.

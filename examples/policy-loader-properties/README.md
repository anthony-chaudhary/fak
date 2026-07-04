# fak kernel — policy loader safety properties

**A malformed policy file should fail at load time, not at the first bad call in
production.** [`POLICY.md`](../../POLICY.md) "Safety properties of the loader"
documents three structural guarantees plus one explicit warning:

```
   fail-loud on config errors        an unknown field/reason/posture is a
                                      FATAL startup error, never a silent fallback

   replace, not merge                a loaded manifest IS the whole floor —
                                      --dump gives you the complete default to edit

   round-trip stable                 fak policy --dump | fak policy --check
                                      is exact — no drift between emit and parse

   empty manifest {} is valid        but explicitly WARNED — the maximally
                                      paranoid floor where every call is denied
```

This demo runs all four as `fak policy --check` witnesses — no model, no network,
deterministic exit codes. Expected runtime: five `fak policy` calls complete in seconds.

## Run it

```bash
./examples/policy-loader-properties/run.sh
```

It needs only the `fak` binary on your `PATH` (or set `FAK_BIN=/path/to/fak`). Full
captured run: [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

## The five witnesses

### 1. Round-trip stable

`fak policy --dump` emits the built-in default; `fak policy --check` on that same
output must accept it exactly — a future `--dump` change that broke round-trip would
silently change every deployed floor built from it:

```bash
fak policy --dump > dumped.json
fak policy --check dumped.json
# OK ... (manifest valid; every deny cites a closed-vocabulary reason)
```

### 2. Fail-loud — unknown field

[`bad-unknown-field.json`](bad-unknown-field.json) has `"allows"` where the schema
wants `"allow"` — the exact typo named in POLICY.md's fail-loud clause:

```bash
fak policy --check examples/policy-loader-properties/bad-unknown-field.json
# fak policy: policy ...: invalid manifest: json: unknown field "allows"
# exit 1
```

Go's `json.Decoder` with `DisallowUnknownFields` refuses the typo outright instead of
silently dropping it and falling back to a more permissive default.

### 3. Fail-loud — unknown deny reason

[`bad-unknown-reason.json`](bad-unknown-reason.json) cites `"NOT_A_REAL_REASON"` for a
`deny` entry — not a member of the closed refusal vocabulary:

```bash
fak policy --check examples/policy-loader-properties/bad-unknown-reason.json
# fak policy: policy ...: unknown deny reason(s): exfiltrate="NOT_A_REAL_REASON"; valid reasons: ...
# exit 1
```

### 4. Fail-loud — unknown posture value

[`bad-unknown-posture.json`](bad-unknown-posture.json) sets `"posture": "yolo_mode"` —
only `fail_closed` and `admit_and_log` exist:

```bash
fak policy --check examples/policy-loader-properties/bad-unknown-posture.json
# fak policy: policy ...: unknown posture "yolo_mode" (want fail_closed|admit_and_log)
# exit 1
```

### 5. Empty manifest — valid but warned

[`empty.json`](empty.json) (`{}`) is the maximally paranoid floor: nothing is
affirmatively allowed, so every call resolves to `DEFAULT_DENY`. It is valid — `--check`
exits 0 — but prints an explicit `NOTE` so you never deploy it by accident, mistaking
silence for a working policy:

```bash
fak policy --check examples/policy-loader-properties/empty.json
# OK ... (manifest valid; every deny cites a closed-vocabulary reason)
# ...
# NOTE: nothing is affirmatively allowed — this is the fail-closed
# empty floor; EVERY call resolves to DEFAULT_DENY.
# exit 0
```

## Why "replace, not merge" doesn't get its own fixture

The three fail-loud cases and the empty-manifest case are each a single bad/edge
manifest `fak policy --check` can adjudicate directly. "Replace, not merge" is a
property of *how* a manifest is applied, not a shape `--check` alone can witness — the
loaded manifest **is** the whole floor with no partial overlay onto the built-in
default. `--dump` is what makes this safe in practice: it hands you the *complete*
default to edit from, so trimming a section from your file can't silently inherit
whatever the binary's built-in table happened to say. The round-trip witness (#1) is
the closest structural proof this demo can run: the dumped manifest, unmodified,
re-parses to the identical floor — evidence that dump and load speak the same complete
schema, not partial fragments.

## Files

| file | what it is |
|---|---|
| `run.sh` | one-command launcher: runs the five witnesses, asserts each exit code |
| `bad-unknown-field.json` | fixture: unknown JSON field (`"allows"` typo) |
| `bad-unknown-reason.json` | fixture: `deny` entry citing a non-vocabulary reason |
| `bad-unknown-posture.json` | fixture: `"posture"` set to a value that isn't `fail_closed`/`admit_and_log` |
| `empty.json` | fixture: `{}`, the valid-but-warned maximally paranoid floor |
| `EXAMPLE-OUTPUT.md` | a captured run |

The round-trip witness (#1) doesn't ship a static fixture — it dumps the binary's
built-in default fresh at run time and checks that exact output, so it can never drift
from whatever the current default actually is.

Related: [`POLICY.md`](../../POLICY.md) "Safety properties of the loader",
`internal/policy` (`TestRoundTrip`, `TestLoadedPolicyIsLoadBearing`,
`TestUnknownDenyReasonRejected`).

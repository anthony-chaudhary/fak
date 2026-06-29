# Captured run — loader properties

Real `fak policy --check` / `fak policy --dump` output on `fak` **v0.34.0**, run from
the repo root. The loader is a pure function of the manifest bytes — the same file always
yields the same verdict and the same exit code. The `--check` line echoes the path you
passed it (relative below, because each command was invoked with a relative path).

Two flag facts worth stating up front, because they differ from the one-liner in
POLICY.md "Safety properties of the loader":

- `fak policy --check <FILE>` takes a **file argument**; it does not read stdin. The
  POLICY.md shorthand `fak policy --dump | fak policy --check` is conceptual — the
  runnable round-trip dumps to a file and checks the file (witness 1 below).
- A failed `--check` writes its named error to **stderr** and exits **non-zero**. A
  passing `--check` writes the admitted floor to **stdout** and exits **0**. That
  exit code is the whole point: a misconfigured manifest fails here, at load, not at
  the first bad call in production.

---

## Property 1 — ROUND-TRIP STABLE (exit 0)

`--dump` writes the built-in default; `--check` parses it back to an identical, valid floor.

```console
$ fak policy --dump > /tmp/default.json
$ fak policy --check /tmp/default.json
OK  /tmp/default.json  (manifest valid; every deny cites a closed-vocabulary reason)

posture            : fail_closed
allow (exact)      : 10 tool(s)
allow (prefix)     : read_, get_, search_, list_, lookup_, find_, calc
deny (explicit)    : 2 tool(s)
                     exfiltrate -> SECRET_EXFIL
                     shell_rm_rf -> POLICY_BLOCK
self-modify globs  : internal/abi/, internal/kernel/, internal/adjudicator/, internal/architest/, internal/shipgate/, dos.toml, .dos/, fak/internal/
redact arg fields  : password, secret, api_key, token, authorization
arg rules          : 0 rule(s)
ifc safe sinks     : (none)
ifc authorize      : 0 rule(s)
ifc sources        : 0 tool(s)
rate limit         : (none — inert)
$ echo $?
0
```

Why this matters: `--dump` is how an adopter gets the *complete* default floor to edit
from (the **replace-not-merge** property — see property 3). If a future change to `--dump`
emitted a manifest that `--check` could not parse back to the same floor, deployed floors
would silently drift from what the binary thinks the default is. `TestRoundTrip`
(`internal/policy/policy_test.go`) pins this exactness.

---

## Property 2 — FAIL-LOUD on config errors (exit non-zero, named error)

A malformed or unknown manifest is a **fatal load error**. fak never silently falls back
to a more permissive default — the failure mode it exists to prevent is "the file was
ignored and the agent ran with default permissions."

### 2a. Unknown JSON field — `"allows"` instead of `"allow"` (the classic typo)

```console
$ fak policy --check examples/loader-properties/bad-unknown-field.json
fak policy: policy examples/loader-properties/bad-unknown-field.json: invalid manifest: json: unknown field "allows"
$ echo $?
1
```

The decoder runs with `DisallowUnknownFields` (`internal/policy/policy.go` `ParseManifest`),
so a misspelled key is a hard error — not a key that's silently dropped, leaving `allow`
empty and the floor unexpectedly permissive-by-default.

### 2b. Unknown deny reason — a code outside the closed refusal vocabulary

```console
$ fak policy --check examples/loader-properties/bad-deny-reason.json
fak policy: policy examples/loader-properties/bad-deny-reason.json: unknown deny reason(s): exfiltrate="TOTALLY_MADE_UP_REASON"; valid reasons: DEFAULT_DENY, LEASE_HELD, MALFORMED, MISROUTE, OVERSIZE, POLICY_BLOCK, RATE_LIMITED, RESULT_SECRET_DISCOVERED, SECRET_EXFIL, SELF_MODIFY, TRUST_VIOLATION, UNKNOWN_TOOL, UNWITNESSED
$ echo $?
1
```

Every `deny` reason must be a name from the closed refusal vocabulary, so a refusal always
cites a verifiable code and a deny-loopback can derive a disposition from it. The error
lists the full valid set. (`TestUnknownDenyReasonRejected`.)

### 2c. Unknown posture value — not `fail_closed` | `admit_and_log`

```console
$ fak policy --check examples/loader-properties/bad-posture.json
fak policy: policy examples/loader-properties/bad-posture.json: unknown posture "fail_open_please" (want fail_closed|admit_and_log)
$ echo $?
1
```

An unknown posture is **refused, not coerced** to a default. This is the case the issue
flagged as "#201 territory" — on v0.34.0 it is already resolved: the loader rejects an
unknown posture loudly rather than silently accepting it. (`TestUnknownPostureRejected`.)

### 2d. Malformed JSON (truncated) — `invalid manifest`, not an empty permissive floor

```console
$ fak policy --check examples/loader-properties/bad-malformed.json
fak policy: policy examples/loader-properties/bad-malformed.json: invalid manifest: unexpected EOF
$ echo $?
1
```

A structurally broken file fails the parse outright. It does **not** degrade to an empty
(or default) floor — the bytes that don't parse are an error, full stop.

---

## Property 3 — REPLACE-NOT-MERGE, and the empty `{}` floor (valid, but warned)

A loaded manifest **is** the whole floor — it replaces the default, it is not merged into
it. The extreme of this is the empty manifest `{}`: it is *valid*, and because nothing is
merged in, it allows **nothing**. `--check` accepts it (exit 0) but prints an explicit
warning so you never deploy a dead floor by accident.

```console
$ fak policy --check examples/loader-properties/empty.json
OK  examples/loader-properties/empty.json  (manifest valid; every deny cites a closed-vocabulary reason)

posture            : fail_closed
allow (exact)      : 0 tool(s)
allow (prefix)     : (none)
deny (explicit)    : 0 tool(s)
self-modify globs  : (none)
redact arg fields  : (none)
arg rules          : 0 rule(s)

NOTE: nothing is affirmatively allowed — this is the fail-closed
empty floor; EVERY call resolves to DEFAULT_DENY.
ifc safe sinks     : (none)
ifc authorize      : 0 rule(s)
ifc sources        : 0 tool(s)
rate limit         : (none — inert)
$ echo $?
0
```

And the floor behaves exactly as the warning promises — any call against it is denied:

```console
$ fak preflight --policy examples/loader-properties/empty.json --tool read_file --args '{}'
fak: loaded capability floor from examples/loader-properties/empty.json
verdict=DENY reason=DEFAULT_DENY by=monitor
```

This is the "maximally paranoid floor" from POLICY.md: an empty manifest is the safe
default precisely *because* replace-not-merge means an omitted allow-list grants nothing,
rather than inheriting the binary's baked-in allows.

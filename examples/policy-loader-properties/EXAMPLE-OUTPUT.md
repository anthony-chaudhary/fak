# Captured run — policy loader safety properties

Captured with `fak` 0.37.0 via `bash examples/policy-loader-properties/run.sh`. Paths
below are shown relative to the repo root; the round-trip witness uses a fresh temp file
each run, so its printed path will differ from this capture.

## 1/5 — round-trip stable (`dump | check` is exact)

```
$ fak policy --dump > dumped.json
$ fak policy --check dumped.json
OK  dumped.json  (manifest valid; every deny cites a closed-vocabulary reason)

posture            : fail_closed
allow (exact)      : 10 tool(s)
allow (prefix)     : read_, get_, search_, list_, lookup_, find_, calc
deny (explicit)    : 2 tool(s)
                     exfiltrate -> SECRET_EXFIL
                     shell_rm_rf -> POLICY_BLOCK
egress deny hosts  : (none)
research egress    : (none)
self-modify globs  : internal/abi/, internal/kernel/, internal/adjudicator/, internal/architest/, internal/shipgate/, dos.toml, .dos/, fak/internal/
redact arg fields  : password, secret, api_key, token, authorization
arg rules          : 0 rule(s)
ifc safe sinks     : (none)
ifc authorize      : 0 rule(s)
ifc sources        : 0 tool(s)
rate limit         : (none — inert)
isolation          : (none — dial unset; placement fails closed)
tool runtime       : (none — fold defaults apply)
inherited launch   : (none — child inherits nothing)
exit=0
```

## 2/5 — fail-loud: unknown field (`"allows"` typo for `"allow"`)

```
$ fak policy --check examples/policy-loader-properties/bad-unknown-field.json
fak policy: policy examples/policy-loader-properties/bad-unknown-field.json: invalid manifest: json: unknown field "allows"
exit=1
```

## 3/5 — fail-loud: unknown deny reason

```
$ fak policy --check examples/policy-loader-properties/bad-unknown-reason.json
fak policy: policy examples/policy-loader-properties/bad-unknown-reason.json: unknown deny reason(s): exfiltrate="NOT_A_REAL_REASON"; valid reasons: DEFAULT_DENY, LEASE_HELD, MALFORMED, MISROUTE, OVERSIZE, POLICY_BLOCK, RATE_LIMITED, RESULT_SECRET_DISCOVERED, SECRET_EXFIL, SELF_MODIFY, TRUST_VIOLATION, UNKNOWN_TOOL, UNWITNESSED
exit=1
```

## 4/5 — fail-loud: unknown posture value

```
$ fak policy --check examples/policy-loader-properties/bad-unknown-posture.json
fak policy: policy examples/policy-loader-properties/bad-unknown-posture.json: unknown posture "yolo_mode" (want fail_closed|admit_and_log)
exit=1
```

## 5/5 — empty manifest `{}` — valid but warned

```
$ fak policy --check examples/policy-loader-properties/empty.json
OK  examples/policy-loader-properties/empty.json  (manifest valid; every deny cites a closed-vocabulary reason)

posture            : fail_closed
allow (exact)      : 0 tool(s)
allow (prefix)     : (none)
deny (explicit)    : 0 tool(s)
egress deny hosts  : (none)
research egress    : (none)
self-modify globs  : (none)
redact arg fields  : (none)
arg rules          : 0 rule(s)

NOTE: nothing is affirmatively allowed — this is the fail-closed
empty floor; EVERY call resolves to DEFAULT_DENY.
ifc safe sinks     : (none)
ifc authorize      : 0 rule(s)
ifc sources        : 0 tool(s)
rate limit         : (none — inert)
isolation          : (none — dial unset; placement fails closed)
tool runtime       : (none — fold defaults apply)
inherited launch   : (none — child inherits nothing)
exit=0
```

## Summary

```
All 5 loader-property witnesses matched their expected exit code.
```

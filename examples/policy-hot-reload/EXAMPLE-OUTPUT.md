# Captured run

A real run of [`run.sh`](run.sh), color stripped:

```console
$ examples/policy-hot-reload/run.sh
[reload] starting gateway: fak serve http://127.0.0.1:8080  --policy /tmp/tmp.UTMg5Q5Y2E/policy.json  (floor A: deny delete_account)
[reload] gateway healthy (PID 119870): {"engine":"inkernel","model":"mock","ok":true,"planner":"mock"}

[reload] 1) current verdict under floor A:
  ✓ POST /v1/fak/adjudicate delete_account -> DENY
[reload] 2) raise warm in-process state: quarantine a poisoned result onto trace sess-hot-reload-1
  ✓ GET /v1/fak/trace/sess-hot-reload-1 (IFC ledger raised) -> quarantined
[reload] 3) no-restart baseline: start_time_unix=1782249140 uptime=0.7587809s PID=119870

[reload] 4) operator edits the served floor: A -> B (also allow delete_account)
[reload] 5) validate B BEFORE reloading (fail-loud gate, per POLICY.md):
  ✓ fak policy --check $SERVED -> valid
[reload] 6) hot-swap the floor in the running process:
  ✓ POST /v1/fak/policy/reload -> reloaded:true
[reload] 7) the SAME call now resolves against floor B:
  ✓ POST /v1/fak/adjudicate delete_account -> ALLOW
[reload] 8) warm state survived the swap (ledger NOT dropped):
  ✓ GET /v1/fak/trace/sess-hot-reload-1 still tainted after reload -> quarantined
[reload] 9) no-restart proof: start_time_unix=1782249140 uptime=1.6609922s PID=119870
  ✓ start epoch unchanged (1782249140 == 1782249140) and PID 119870 still live — NO restart

[reload] 10) fail-loud over the wire: a BROKEN manifest is refused, last-good floor holds
  ✓ reload of a malformed manifest -> 400 (refused: unknown field "allows")
  ✓ delete_account still ALLOW (B held — no silent fallback) -> ALLOW

[reload] all witnesses passed — floor swapped DENY->ALLOW in-process; IFC ledger + start epoch survived; bad manifest refused.
```

## What the capture proves

- **The floor swapped in-process.** The identical `delete_account` call was `DENY`
  under floor A (step 1) and `ALLOW` under floor B (step 7) — the only thing that
  changed between them was the `POST /v1/fak/policy/reload` in step 6.
- **No restart.** `start_time_unix` is `1782249140` both before (step 3) and after
  (step 9) the reload; `uptime_seconds` only rose (`0.76s` → `1.66s`); the PID
  (`119870`) stayed live throughout. A restart would have changed all three.
- **Warm state survived.** The session quarantined in step 2 was **still**
  `quarantined` in step 8 — the IFC ledger was never dropped. (A fresh process
  would report the clean `trusted` default.)
- **Fail-loud held.** A malformed manifest (the typo `"allows"`) was rejected with
  `400`, and floor B remained in force (`delete_account` still `ALLOW`) — the
  gateway never silently fell back to a more permissive default.

The raw verdict bodies behind the witnesses (for reference):

```jsonc
// step 1 — POST /v1/fak/adjudicate {"tool":"delete_account"} under floor A
{"verdict":{"kind":"DENY","reason":"POLICY_BLOCK","by":"monitor","disposition":"TERMINAL"},"trace_id":"gw-2"}

// step 6 — POST /v1/fak/policy/reload
{"reloaded":true,"source":"/tmp/tmp.UTMg5Q5Y2E/policy.json","summary":"posture            : fail_closed\nallow (exact)      : 2 tool(s)\n..."}

// step 7 — same call, now under floor B
{"verdict":{"kind":"ALLOW","by":"monitor"},"trace_id":"gw-6"}

// step 10 — reload of {"allows":["x"]}
// HTTP 400
{"error":{"message":"policy reload failed: policy /tmp/.../policy.json: invalid manifest: json: unknown field \"allows\"","type":"invalid_request_error"}}
```

> The `mock` engine/model in the health line is the in-kernel default `fak serve`
> boots with no `--base-url` — this example needs no upstream, so the verdict
> surface and lifecycle routes answer deterministically without one.

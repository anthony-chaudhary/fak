# Issue #9714 — Qwen incumbent launchd ownership HOLD

Verdict: **HOLD**. The preserved Qwen3.6 incumbent is healthy and stable, but
the expected `com.fak.qwen36-model` launchd job still does not own it. The
expected job and its LaunchAgent plist were absent at preflight. The one port
8090 listener was owned by the same alternate supervisor recorded in the prior
issue HOLD.

The read-only identity check bound the incumbent to command SHA-256
`a0c13c8d7e84fa92c76db532db0a6fed13b3b42313a75e10b8f69e252f76a91d`
and the alternate owner to public-safe label SHA-256
`b567298df044cecdac3cf921d7cb971e4665db7b5407fb634647a910e3abbdb7`.
Both `/health` and `/v1/models` returned HTTP 200 and the inventory advertised
only `qwen3.6-27b`.

Ninety-one read-only samples spanning 109 seconds retained one listener, the
same command digest, the same alternate-owner digest, HTTP 200 for both routes,
and the same model alias. This is evidence that the untouched incumbent
remained stable; it is explicitly not post-restore credit because no lifecycle
operation was admitted.

## Fail-closed boundary

The live issue authorizes a lifecycle drill through the expected service
identity only after that identity binds the exact listener. It does not
authorize reconfiguring or booting out the proven alternate supervisor. The
expected job was absent, so no GPU lease was acquired, no process was signaled,
no bootout or bootstrap was attempted, and no service configuration changed.
No Qwen3.8 artifact was touched and no model arm ran.

The next admissible action is an explicit service-owner decision that supplies
a reviewed definition for `com.fak.qwen36-model` and authorizes migration from
the already-proven alternate launchd owner. After that migration, reacquire the
canonical GPU lease and repeat the issue's TERM-only stop/restore drill, binding
the exact PID and command before every lifecycle action and requiring at least
90 continuous seconds of post-restore health and model stability.

## Readback

```console
go test ./docs/_witnesses/issue-9714-qwen36-launchd-ownership -count=1 -v
```

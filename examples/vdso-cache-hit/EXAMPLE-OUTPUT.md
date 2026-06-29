# Captured run — vDSO cache-hit walkthrough

A real capture of the two traces replayed through a prebuilt `fak` (`fak --version` →
`0.34.0`) on Windows. `fak run --trace` is a pure offline replay — no model, no server,
no network — so this output is deterministic: the same trace always yields the same
`vdso_hits` count and the same per-call `by=` column.

```
$ fak run --trace examples/vdso-cache-hit/cache-friendly.json
[ 0] get_reservation_details      verdict=ALLOW     by=monitor   status=OK
[ 1] get_reservation_details      verdict=ALLOW     by=vdso      status=OK

summary: submits=2 vdso_hits=1 engine_calls=1 denies=0 transforms=0 quarantines=0
```

Call `[0]` is the first read — the cache is empty, so it is a fresh call (`by=monitor`)
and its result is what fills the tier-2 content cache. Call `[1]` is the identical read:
the `(tool, canonical-args, world-version)` key now resolves, so it is served from the
cache — `by=vdso`, and `vdso_hits=1`.

```
$ fak run --trace examples/vdso-cache-hit/invalidating.json
[ 0] get_reservation_details      verdict=ALLOW     by=monitor   status=OK
[ 1] book_reservation             verdict=ALLOW     by=monitor   status=OK
[ 2] get_reservation_details      verdict=ALLOW     by=monitor   status=OK

summary: submits=3 vdso_hits=0 engine_calls=3 denies=0 transforms=0 quarantines=0
```

Same first read at `[0]`. Call `[1]` is `book_reservation` — a write-shaped completion
(`readOnlyHint=false`, `destructive=true`) — which bumps the world-version. So when the
identical read arrives at `[2]`, its old cache key (bound to the pre-write world-version)
is unreachable: it MISSES, runs fresh (`by=monitor`), and `vdso_hits=0`.

The contrast between the two `vdso_hits` lines is the demo: `1` vs `0` for the same pair
of reads, the only difference being whether a mutation landed in between. That is the
soundness invariant — **a cache hit equals a fresh call** — made visible. The cache
would rather miss than serve a value the world has moved past.

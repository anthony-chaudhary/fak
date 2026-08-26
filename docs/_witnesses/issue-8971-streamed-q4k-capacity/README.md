# Issue 8971 — streamed Q4_K capacity receipt

This packet records the canonical no-`FAK_Q4K_FREE_CPU` startup control on the
36 GiB M3 Pro. It is a refusal witness, not an admission claim: the native serve
reached readiness and executed one generated token through the in-kernel Metal
forward, but peak swap grew by 7,681,930,690 bytes.

The fail-closed bound is 47,244,640,256 bytes (44 GiB), computed as the next
whole GiB at or above the 36 GiB host plus the observed peak swap delta. This
does not claim that 44 GiB hardware was run; it is the minimum admission bound
derived from the measured 36 GiB failure. A future admission receipt may replace
it only with the same exact artifact, environment, cache displacement, native
identity, zero-fallback, and zero-positive-swap contract.

The receipt retains only scrubbed host and file identities. The two displacement
sources were sequentially hashed and totaled 48,368,448,672 bytes. The bounded
watcher sent one TERM to the exact pre-run owner of port 8090, sent no unmatched
signal, stopped, and restored the same owner command identity. Both `/health`
and `/v1/models` returned HTTP 200 at restoration and again after 30 seconds.
An earlier interactive restoration exited when its owning runner closed and was
rejected; the recorded restoration is the permanent launchd-owned service and
includes the full 992-second recovery interval.

Read back the hash-bound packet without loading a model:

```console
fak native-performance --capacity-receipt \
  docs/_witnesses/issue-8971-streamed-q4k-capacity/canonical-no-free-cpu.json
```

# Issue #9097 — resident Metal GDN production canary

Verdict: **REJECT**. The opt-in same-binary path reached the fak-owned Metal GDN
sequence implementation with direct resident Q4_K projections, but the exact
P=32,800/T=8 request produced no response before the predeclared paging guard.
The selector remains off by default.

The frozen candidate was built from unattached commit
`87b92938232ccf24d290d4c2ef08e16cf55cc0d3` (tree
`60a6c1f5f3de7687fc39d95093451dcb1b54ce83`) with a clean build-vcs identity.
It used the 17,106,775,008-byte Qwen3.8-27B Q4_K_M artifact at SHA-256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
The generated `long-context-needle-v1` prompt and request are reproducible from
the checked-in corpus and are pinned by `receipt.json` and the readback test.

## Result

- The candidate process logged `engine=inkernel`, `backend=metal`,
  `forward_path=metal/qwen35-gdn-preprojected-sequence-v1`, `q4k=true`; no
  llama.cpp or runtime fallback participated.
- The request started at `2026-08-26T13:49:29Z`. It was terminated by the
  paging guard at `2026-08-26T13:51:23Z` with no response headers.
- The third consecutive over-limit sample measured 47,244,640,256 bytes of OS
  footprint, 7,071,285,248 bytes of RSS, 36% system-free memory, and
  14,780,664,381 bytes of swap growth.
- The complete run peaked at 47,053,850,128 bytes by `/usr/bin/time -l`,
  47,244,640,256 bytes sampled footprint, and 19,071,762,432 bytes by
  `/usr/bin/time -l` maximum resident set size. The minimum observed
  `memory_pressure -Q` system-free percentage was 28%; it is not a
  memorystatus measurement. Actual memorystatus percentage was unbound, which
  is an independent reason this run cannot pass the acceptance envelope.
  Targeted kernel pressure/memorystatus log events were zero.
- Teardown used TERM only. The prior port-8090 owner was restored with its exact
  command digest and returned both `/health` and `/v1/models` HTTP 200 for 90
  continuous seconds before the hardware lease was released.

`FAK_GGUF_LOAD_WORKERS=12` is the host's unchanged default
(`GOMAXPROCS=12`). The already-landed #9059 ablation rejected one-worker load
as slower and higher-footprint on this exact host, so this issue ran no second
canary. The next measured boundary is the 12 GiB paging ceiling: production
cannot be accepted on this 36 GiB envelope until the same exact request
completes without crossing it and while capturing the actual memorystatus
source and retaining the stated footprint, event, identity, and restoration
guards.

## Readback

```console
go test ./docs/_witnesses/issue-9097-qwen38-metal-gdn-production -count=1
```

# Qwen3.8 paged-swap codec dogfood — 2026-08-29

Issue: #9617
Spine: `bc95ccc313b6c6d3c841f1b1e5bc0fe99aee70a7` (`internal/model`)

## Result

The production-sized Qwen3.8 paged-swap codec round-tripped repository-derived state exactly.

```text
repo_files=18234
repo_bytes=157160205
repo_digest=891bdfe72c9a7291
payload_bytes=158488532
round_trip_exact=true
```

The repository input is the content plus sorted manifest of Git-tracked regular files at or below 128 KiB in the live checkout. The witness hashes each file body, relative path, and size, then uses that digest to fill every serialized K, Kraw, V, convolution, and recurrent float plane. This binds the exercised production-shaped state to the repository's current live work without treating a static tensor fixture as dogfood.

## Reproduce

Run from the repository root under WSL, per the Windows test fence:

```bash
FAK_QWEN38_SWAP_DOGFOOD=1 GOTOOLCHAIN=auto \
  go test ./internal/model \
  -run '^TestQwen38PagedSwapDogfoodRepoRoundTrip$' \
  -count=1 -v
```

The test is opt-in because it materializes and compares the 158,488,532-byte production envelope. The default package test run only compiles and skips it.

## Defects

No codec defect surfaced. The payload matched the declared operating envelope and every serialized float bit survived restore, so no defect issue was filed.

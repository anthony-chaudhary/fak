# Support witness ingestion and GPU dispatch packet

**Date:** 2026-08-15 · **Issue:** [#6896](https://github.com/anthony-chaudhary/fak/issues/6896)

## Shipped ingestion contract

`fak-support-witness/1` binds an exact support tuple to witness identity, state, proof tier, authority, artifact SHA-256, captured environment, reproduction command, observation/expiry window, source commit, payload SHA-256, baselines, fallback, and penalty.

`supportgraph.Ingest` rejects malformed or digest-mismatched witnesses, is idempotent by witness ID, retains historical evidence, and expires prior evidence when the same artifact/backend/kernel path is observed under a changed runtime or hardware baseline. `cmd/supportwitness` updates graph JSON from a witness; human support tables should be generated from this source rather than edited as a second authority.

## Real-hardware status

The sanctioned GCP probe was run on 2026-08-15:

```text
project=dos-rlvr-admit-20260608
H100 1g/8g: PROVISIONABLE in us-central1
L4: NO_QUOTA in us-central1
T4: NO_QUOTA in us-central1
recommended: a3-mega-h100
```

`python tools/gcp_bench.py --dry-run --tier a3-high-h100-1g` planned an `a3-highgpu-1g` in `us-central1-a`, CUDA 12.9 image, bounded two-hour lifetime, automatic teardown, and estimated ~$11/hour. It was not applied because that driver does not emit `fak-support-witness/1`; spending on it would not satisfy #6896.

The private bridge binary and Slack token are present, but `dgxbridge doctor --json` reports the control channel missing. Authorized operator packet:

```powershell
$channel = '<private control channel id>'
dgxbridge doctor -channel $channel --json
dgxbridge request -channel $channel -wait -timeout 45m -cmd '<checkout origin/main; run tools/run_485_acceptance_on_gpu.sh; capture nvidia-smi and sha256 of the log>'
```

Normalize the returned execution into `fak-support-witness/1` with exact commit, environment, artifact/log digest, reproduction command, and expiry, then ingest:

```bash
go run ./cmd/supportwitness -graph internal/supportgraph/testdata/awq.json -witness W.json -out G.json
go test ./internal/supportgraph
```

No real support edge is claimed yet. Existing support fixtures remain explicitly synthetic.

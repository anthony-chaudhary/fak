# GCP Blackwell (B200/GB200) quota — state + request lineage

> Tracks GitHub issue **#16** — *bench(gcp): Blackwell B200 quota DENIED (granted 0)*.
> This is the "Full state + request lineage" doc the issue links to.
>
> **Provenance note (read this first).** The grant numbers below are **OBSERVED**
> — they are relayed from Google's Cloud Quotas API as of a timestamp; fak does
> not control them and re-filing does not change them. The tool behavior and the
> tier shape are **WITNESSED** — read straight from the code in this repo. The two
> are kept separate on purpose: a denied grant is not a fak failure, and this doc
> does not (and cannot) grant quota. The unblock is an **operator action**.

## Why Blackwell is gated

The flagship head-to-head run wants an `a4-highgpu-8g` VM — **8× NVIDIA B200
(Blackwell)**, compute capability `sm_100`. Two independent quotas gate every
such launch in project `example-gcp-project` (the public placeholder; a real run
needs `--project`/`--contact`):

| Quota | Cloud Quotas `quotaId` | Dimensions | Why it gates |
|---|---|---|---|
| Global GPU ceiling | `GPUS-ALL-REGIONS-per-project` | — | master cross-region cap on all GPUs |
| Per-region B200 family | `GPUS-PER-GPU-FAMILY-per-project-region` | `{region: us-central1, gpu_family: NVIDIA_B200}` | the specific B200-in-a-region gate |

Both must reach **8** for an 8× B200 VM. There is no standalone `NVIDIA_B200_GPUS`
quotaId in Cloud Quotas v1 — B200 is a `gpu_family` *dimension* on the family
quota. (Verified live 2026-06-20: `NVIDIA_B200` is a verbatim, increase-eligible
token. See `tools/gcp_quota_request.py:127` `build_requests`.)

## Observed state (relayed from GCP — 2026-06-20T19:51Z, per the migrated tracker)

| Quota | Preferred | Granted | State |
|---|---|---|---|
| `GPUS-ALL-REGIONS` (global ceiling) | 8 | **1** | request **denied** |
| `NVIDIA_B200` in `us-central1` (per-region family) | 8 | **0** | request **denied** |

Both gates are below the floor for the 8× B200 VM, so **Blackwell is not
provisionable** in this project today. The grant is asynchronous and currently
denied; **re-filing will not change it.**

## What the tooling does (witnessed from code)

- `python tools/gcp_quota_request.py --dry-run` — prints the two POST bodies/URLs,
  submits nothing. A live `--submit`/`--status` **refuses** the
  `example-gcp-project` / `example.com` placeholders (`gcp_quota_request.py:257`);
  pass a real `--project` and `--contact`.
- `python tools/gcp_quota_request.py --status` — polls grant status and writes a
  durable `status-<stamp>.json` evidence file **into this directory**
  (`experiments/gcp-quota/`, the tool's `EVIDENCE_DIR`).
- `python tools/gcp_bench.py --blackwell` — strict: probes only B200/GB200 and
  **STOPs** with the request-quota message when neither is provisionable
  (`gcp_bench.py:729`). This is the correct, honest refusal — not a failure.

The `a4-b200` tier is pinned in `tools/gcp_accel.py:59`: `a4-highgpu-8g`,
8× B200, 180 GB HBM3e each (1,440 GB total), 224 vCPU, ~3968 GB host RAM,
compute capability `100`, zones `us-central1-b` / `us-east5-a` / `europe-west4-b`,
≈ $90/hr.

## Operator unblock (this is the human action — an agent cannot do it)

Pick one:

1. **Escalate the quota request** in the GCP console: *IAM & Admin → Quotas →*
   filter `NVIDIA_B200` (us-central1) and `GPUS-ALL-REGIONS`, raise both to ≥ 8.
2. **Obtain a Blackwell reservation** (or DWS flex-start) — on-demand B200 is
   frequently capacity-gated even with quota.
3. **Run on a different project/account** that already holds B200 quota.

## Re-check (reproducible)

```sh
python tools/gcp_quota_request.py --status \
  --family NVIDIA_B200 --region us-central1 \
  --project <REAL_PROJECT> --contact <REAL_CONTACT>
```

Look for `granted=8` (or ≥ the VM's GPU count) on **both** rows. Each poll drops a
`status-<stamp>.json` here as the evidence trail.

## Done when (issue #16)

- [ ] B200 **and** the global GPU ceiling are granted ≥ 8 in some project, **or**
      an alternative B200-capable project/account is identified. *(operator-gated)*
- [ ] The flagship Blackwell head-to-head is launched:
      `python tools/gcp_bench.py --blackwell --engine all` (arch threads to
      `sm_100` automatically). *(needs live B200 hardware + a working fak-cuda
      device path — depends on the fak-cuda re-run issue)*

Until both boxes are checked the issue stays open: this doc records the blocker,
the lineage, and the exact re-check/escalation steps so the operator (or the next
agent that inherits live B200 quota) can resume without re-deriving any of it.

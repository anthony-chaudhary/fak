# Witness: RTX 5090 sm_120 CUDA acceptance roster — pre-gate census and blocked run (2026-10-28)

`[PHYSICAL_WITNESSED]` Physical-node census and pre-gate evidence for the T3-HW
CUDA acceptance run on the Raider laptop (goal session
`pi-goal-rtx5090-desktop-readiness-20261028`). **The 9-family `make cuda-accept`
roster did NOT run**: at cutoff both source prerequisites were unlanded on public
main and the GPU was held by the standing `fak` service well past the 30-minute
wait cap. Everything below that could be physically probed was probed; nothing is
claimed beyond it. Window: 2026-10-28 06:06–07:26 UTC.

## Node identity (probed over SSH, host-key pin verified)

- Host `raider-fak.local` (mDNS flaky this window — first connect timed out; all
  subsequent work used fallback IP `192.168.1.170`). Both names resolve to the
  same ed25519 host key
  `AAAAC3NzaC1lZDI1NTE5AAAAIHoL59UVpNRk0VD2hypRV+8CBSOkhosKReMOHy6g8TP6`,
  matching the pin in `fak-private/cmd/fak-raider-bench/main.go` L30–34.
  Identity key `C:\Users\USER\.ssh\id_ed25519_prod_to_laptop` (SHA256:2qE4rJeAQ97XQKllwhjx3Gr27FaBD5jDLoUcw+FViJo).
- Arch Linux, kernel `7.2.11-arch1-1`, user `fak`; RTT ~1.4 s.
- GPU: `RTX 5090 Laptop GPU`, UUID `GPU-2a2a837f-c82d-0f22-4538-e3a063903380`,
  PCI `00000000:01:00.0`, driver `615.78.08`, 24463 MiB total — matches expectations.
  Raw CSV: [`nvidia-smi-csv.txt`](nvidia-smi-csv.txt).
- Toolchain: CUDA 13.4 V13.4.92 (`/opt/cuda/bin/nvcc`, not on default PATH —
  raw: [`nvcc-version.txt`](nvcc-version.txt)), cuBLAS 13.8.0.4
  (`libcublas.so.13.8.0.4`, `libcublasLt.so` present), Go 1.27.3 linux/amd64.
- Disk: 706 G free on `/home` and `/`.

## Step 1 — source sync: BLOCKED (no witnessed SHA to sync)

Plan: `.git` measured 9.0 G (> 2 G cap), so transfer would have been a clean
`git archive` tarball of `origin/main` (pre-measured 210 M at `1d46cb93fe`)
extracted to `~/fak-public` — never a dirty working-tree tar, to keep peer WIP
out of the synced tree. **No transfer was made**, because neither required symbol
was on public main at any poll:

- Polled `origin/main` every 5 min, 06:06 → 07:23 UTC (14 fetches; formal 40-min
  cap passed, extended while the coordinator reported the lanes landing imminently).
  Log: [`poll-log-origin-main.txt`](poll-log-origin-main.txt).
- `TestCUDADeviceReceipt`: absent from `internal/compute/` on main at every poll
  (also absent tree-wide).
- `run_nvfp4_w4a4_family`: absent from `tools/cuda_acceptance.sh` on main at every
  poll. Present only as **uncommitted** WIP in the witness lane worktree
  (`.worktrees/ticket-rtx5090-sm120-kernel-hw-witness-20261027`, +146 lines to
  `tools/cuda_acceptance.sh`) — unwitnessed, not usable as sync source.
- Main advanced `1d46cb93fe` → `7219a9f438` → `7b47b8acfb` during the window;
  all landed commits unrelated (goalregistry fix, Qwen3.5-hybrid perf, dos ignore,
  ticket docs, hwutil telemetry, GSQ-RCO docs, devcmd ledger).
- Synced SHA: **N/A** (no checkout materialized; node still holds only the stale
  209 M tarball `~/raider-source-517ad375.tar.gz`, rev `517ad375`, 2026-10-25).

## Step 2 — weights: DONE and byte-verified

`Qwen/Qwen2.5-1.5B-Instruct` safetensors snapshot downloaded on the node via
`curl` from huggingface.co (direct reachability confirmed; no proxy needed) into
`/srv/models/Qwen2.5-1.5B-Instruct`. All 7 files match HF CDN sizes exactly
(`model.safetensors` = 3087467144 B; manifest: [`weights-manifest.txt`](weights-manifest.txt)).
Disclosed host mutation: passwordless `sudo` used only for
`mkdir -p /srv/models/Qwen2.5-1.5B-Instruct && chown fak:fak` (directory is
root-owned). No packages installed; no services touched.

## Step 3 — compile check: NOT RUN

Depends on step 1: the `cuDriverGetVersion` cgo wiring it is meant to prove is
not on main at cutoff (verified absent in `internal/compute/` on
`7b47b8acfb`), so compiling current main would not answer T1.

## Step 4 — acceptance run: NOT RUN (two blockers, both outside agent authority)

1. **Source blocker** (above): the 9-family roster incl. NVFP4 W4A4 is not on
   public main, so there is no witnessed tree to run `make cuda-accept` from.
2. **GPU held**: the standing `fak` service (`systemctl is-active fak` =
   `active`, pid 813 `/usr/local/bin/fak`) pinned **22848/24463 MiB continuously
   from 06:06Z through 07:25Z** (>75 min; wait cap was 30 min at 5-min polls —
   log: [`gpu-busy-poll-log.txt`](gpu-busy-poll-log.txt)). This is the resident
   Qwen3.8-27B serve, not a competing worker: the `owned-rtx5090-raider`
   maintain lease was released 2026-10-28T04:23Z and no eGPU is attached
   (boot `d003c8a2`, consistent with the 10-27 census). Per protocol the process
   was neither killed nor the service stopped/restarted; an operator decision is
   required (verify no dependent sessions, then stop/evict).

Per-family verdict table (verbatim status — SKIP is not PASS, and none ran):

| # | Family | Verdict @ 2026-10-28 07:25Z |
|---|--------|------------------------------|
| 1 | base-forward (#479, f32 SGEMM) | NOT_RUN (GPU held by fak service; roster not on main) |
| 2 | async-hostxfer (#482) | NOT_RUN (same) |
| 3 | cuda-graph (#483) | NOT_RUN (same) |
| 4 | fp16 HGEMM (#484) | NOT_RUN (same) |
| 5 | quant Q8_0/Q4_K (#485) | NOT_RUN (same) |
| 6 | flash-attention (#486) | NOT_RUN (same) |
| 7 | GLM-DSA | NOT_RUN (same) |
| 8 | AWQ 4-bit (#926) | NOT_RUN (same) |
| 9 | NVFP4 W4A4 native (sm_120, cuBLASLt>=12.8.4) | NOT_RUN (same; family wrapper itself unlanded) |

## Caveats

- mDNS `raider-fak.local` failed to resolve once at window start; fallback IP
  used throughout (host key identical to pin, so no MITM concern recorded).
- `git rev-parse HEAD` on the node: not applicable — no fresh checkout exists
  (see step 1). The last *synced* artifact on the node remains rev `517ad375`.
- The unlanded-lane state means this receipt documents readiness, not
  qualification: sm_120 hardware acceptance is **not yet** evidenced.

## Conclusion and next checkable steps

Node is ready (identity, toolchain, weights, disk all witnessed green); the run
is gated on two external actions.

1. Coordinator: land `TestCUDADeviceReceipt` (internal/compute) and
   `run_nvfp4_w4a4_family` (tools/cuda_acceptance.sh) on public main. Verify:
   `git grep -n TestCUDADeviceReceipt origin/main -- internal/compute/` and
   `git grep -n run_nvfp4_w4a4_family origin/main -- tools/cuda_acceptance.sh`.
2. Operator: confirm no active fak sessions, then free the GPU (stop the `fak`
   service or evict the resident model) — lease already released, nothing else
   needs the device.
3. Re-dispatch this packet: `git archive` main at the landed SHA (~210 M),
   scp + extract to `~/fak-public`, record `git rev-parse HEAD`, then
   `FAK_CUDA_ARCH=sm_120 FAK_CUDA_ACCEPT_HF=/srv/models/Qwen2.5-1.5B-Instruct
   make cuda-accept` (PATH/CUDA_HOME = /opt/cuda); append the verbatim
   per-family verdicts and full log to this directory.

## Status update - 2026-10-28T14:50Z (post-cutoff re-probe; goal pi-goal-rtx5090-sm120-cuda-accept-rerun)

The two source prerequisites named in Step 1 have since landed on public main; device availability and the runs themselves remain unverified. This section distinguishes the resolved from the open.

| Prerequisite | Landed commit | Verification (re-probed 2026-10-28 ~14:26Z) |
| --- | --- | --- |
| Typed sm_120 device receipt (TestCUDADeviceReceipt, internal/compute) | e7a506c76c | git merge-base --is-ancestor e7a506c76c origin/main exit 0; git grep -n TestCUDADeviceReceipt origin/main -- internal/compute/ -> cuda_device_receipt_test.go:14 |
| Ninth acceptance family (run_nvfp4_w4a4_family, tools/cuda_acceptance.sh) | 99d9a5b383 | git merge-base --is-ancestor 99d9a5b383 origin/main exit 0; git grep -n run_nvfp4_w4a4_family origin/main -- tools/cuda_acceptance.sh -> lines 35/60/72 |

Public main at re-probe: 88b4db37b7a596ce6d2259c209ff7d25abe4881d. The Step-1 source blocker is therefore RESOLVED: a witnessed tree now exists to sync and run from.

Still open (unchanged from cutoff):
- Device availability: the standing fak service still holds the GPU (re-probed 14:25Z: 22,862/24,463 MiB used, utilization 0%, service active, PID 15768, boot d1026348-79cd-485e-a9b5-c8f5da69e578, unit sha256 b6195d3593d9cfa0fcffa2d6caa7b82cdb1488823d0ec62d66f2e8603e517b40, drop-ins 70-workspace-admission/85-cuda-device-init/90-raider-qualified-model). A peer maintain lease hwl-66b7787b64ae49fc (holder raider-coldprefill-goal) fences the node until 2026-10-28T17:53:11Z; no exclusive action is taken before its expiry. Re-probe also found the service idle (0 active engine requests; single loopback probe connection) and that host router inference routes through the cloud modular backend (live TCP census 14:40Z: router outbound only to 207.210.107.x:443), so stopping it under our own lease breaks no known agent session.
- Execution: all nine families remain NOT_RUN; no cold-prefill or decode measurement exists yet. The Step-3 re-dispatch recipe below is unchanged and now executable end-to-end once the lease fence clears.

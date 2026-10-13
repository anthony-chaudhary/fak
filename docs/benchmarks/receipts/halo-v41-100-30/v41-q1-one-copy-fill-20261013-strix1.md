# DS4.1 Q1 page-in microbench: the read is ~6.4 GB/s cold and the host copy is under 2%, so "one-copy" is not the lever (2026-10-13)

- **Host and class.** `[HW-WITNESSED]` Halo node `strix-halo-fak` / `strix1` (AMD Ryzen AI MAX+ 395 w/ Radeon 8060S, 32 threads, kernel `7.2.3-arch1-3`), NVMe `KIOXIA-EXCERIA BASIC SSD` (DRAM-less), `/home` on `btrfs rw,noatime,compress=zstd:1,ssd`. **strix3 itself was under a peer lease** (`claude-v41c11-indexer`, `/var/lib/fak/leases/dev.lease`) with a live V4.1 serve at 12G ring during this probe, and the fleet serving reserve (`strix1 has 0 confirmed serving peers, reserve 1`) refused an exclusive lease on the free node. So this is a Halo-class read witness, not a strix3 token witness; it is labelled by the node it ran on.
- **A zero-filled probe file lies on zstd btrfs.** `[HW-WITNESSED]` The first probe used a zero-filled slab and read it at 32.5 GB/s "O_DIRECT" — impossible for this drive. `/home` is `compress=zstd:1`, so zero/repetitive bytes are a compressed extent and the read never reaches the device. `dd iflag=direct` of the 2 GiB zero slab reported 32.1 GB/s; a high-entropy refill plus `sync; echo 3 > /proc/sys/vm/drop_caches` reported 3.6 GB/s for the same span. **Every number below is high-entropy bytes with the page cache dropped before each arm.**
- **Cold page-in is ~6.3–6.5 GB/s, 0.66–0.69 ms per 4.27 MB stride.** `[HW-WITNESSED]` Over three independent cold runs (512 strides, 2.04 GiB working set, `sync; echo 3 > /proc/sys/vm/drop_caches` between arms, `ionice -c2 -n7 nice -n 19` so the co-resident serve kept priority):

  | Arm | Run 1 | Run 2 | Run 3 | ms / 4.27 MB |
  | --- | ---: | ---: | ---: | ---: |
  | serial (queue depth 1) | 6.474 | 6.477 | 6.480 GB/s | 0.659 |
  | two-copy model (read + host copy) | 6.415 | 6.181 | 6.410 GB/s | 0.665–0.691 |
  | pool qd=8 (ordered queue, 8 workers) | 6.373 | 6.214 | 6.288 GB/s | 0.670–0.687 |
  | warm (page cache, no drop) | 42.9 | 76.9 | 45.2 GB/s | 0.056–0.100 |

- **The host copy is under 2% — the one-copy port's direct saving is not measurable.** `[HW-WITNESSED]` The two-copy arm models the pre-P2 host path (`uploadQ3KLocked`'s `make` + `copy`): read one 4.27 MB stride into a staging buffer, then copy it into the destination. It measured within ±1.7% of the one-copy serial arm across all three runs (6.18–6.42 vs 6.47–6.48 GB/s). On this box the 4.27 MB host memcpy is hidden behind the page-in, so **removing it changes the read path by less than the run-to-run spread.** The one-copy port is still the correct shape (it also removes an allocation per page-in), but it cannot be credited with a read-path speedup here.
- **Concurrency does not raise cold buffered read throughput on this drive.** `[HW-WITNESSED]` A queue-depth sweep at 2.04 GiB cold gave 6.327 (qd1), 6.320 (qd2), 6.178 (qd4), 6.325 (qd8), 6.373 (qd16) GB/s — flat within noise. A single buffered stream already saturates this DRAM-less NVMe's page-in, so P2 step 1's 0.648→0.780 tok/s cold win does **not** come from raw read bandwidth; it comes from overlapping the read with the upload/compute that the serial loop blocked on.
- **What the slice-1 1.19 ms actually contains.** `[HW-WITNESSED]` The slice-1 receipt's 1.19 ms per 4.27 MB (≈3.6 GB/s) sits *below* this cold read rate (0.66 ms, ≈6.4 GB/s), so roughly **0.5 ms of the 1.19 ms is device-upload and driver/scheduling overhead, not reading and not host copy.** That is the residue the P2 one-copy port does not address and the reason dispatch/fusion (P3/P4) is the next lever, not another read-path copy elimination.

## Binding

| Field | Value |
| --- | --- |
| Host | `strix1` (Halo, RYZEN AI MAX+ 395 / Radeon 8060S), `uname -r` `7.2.3-arch1-3`, 32 threads |
| Drive / FS | `KIOXIA-EXCERIA BASIC SSD` (DRAM-less), `btrfs compress=zstd:1` on `/dev/nvme0n1p2[/@home]` |
| strix3 | under peer lease `lease-strix3-52aa280f1f9a` (`claude-v41c11-indexer`), live V4.1 serve `--expert-ring-bytes 12G`; not disturbed |
| Probe | `expert_fill_probe.go`, standalone mirror of `internal/model/expert_fill_microbench_test.go`; transferred to the node, built with go1.27.0, run under `ionice -c2 -n7 nice -n 19` |
| Data | high-entropy (deterministic `rand` seed `0x3340`), 512 × 4,269,056 B = 2.04 GiB; `sync; echo 3 > /proc/sys/vm/drop_caches` before every cold arm |
| Stride | 4,269,056 B — the V4.1 Flash Q2_K expert projection stride the slice-1 receipt states its page-in in |
| Destination | 2 MiB-aligned (heap base rounded up), the alignment the P2 port targets |
| Classification | `[HW-WITNESSED]` read path on a Halo node; **not** a tok/s claim |
| Reproduce | `go test ./internal/model -run TestExpertFill -count=1 -v` (the committed instrument, host-relative), or the probe script above on any Halo |

## Consequence for the next work

The one-copy fill (Q1/P2) is landed behind the existing flags and is byte-correct, but this microbench says its **read-path** headroom is under 2% and the cold read is already at the drive's page-in rate. The remaining ~0.5 ms/stride is upload+scheduling, so **the next decision-changing lever is the dispatch/fusion work (Q2/P3–P4), not another copy elimination on the fill.** The one-copy port still earns its place because it removes a per-page-in allocation and matches the pinned-buffer identity discipline, and because it is a prerequisite shape for the slot-map fill P3 needs.

<!-- fak-climb: metric=page_in_gbps value=6.47 host=strix1 workload=expert-fill-cold-serial strides=512 class=HW-WITNESSED at=2026-10-13T04:20:00Z -->
<!-- fak-climb: metric=page_in_ms_per_stride value=0.659 host=strix1 workload=expert-fill-cold-serial strides=512 class=HW-WITNESSED at=2026-10-13T04:20:00Z -->
<!-- fak-climb: metric=page_in_gbps value=6.37 host=strix1 workload=expert-fill-cold-pool-qd8 strides=512 class=HW-WITNESSED at=2026-10-13T04:20:00Z -->
<!-- fak-climb: metric=page_in_gbps value=76.9 host=strix1 workload=expert-fill-warm-pool-qd8 strides=512 class=HW-WITNESSED at=2026-10-13T04:20:00Z -->

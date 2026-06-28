# EXAMPLE-OUTPUT — a captured `./run.sh`

A verbatim run of [`run.sh`](run.sh) over [`sample-transcript.jsonl`](sample-transcript.jsonl),
exercising all six `fak debug` inspection surfaces with Go-only prerequisites (no model, no
network). `✓` lines are the witnesses the script asserts; `[cdb]` lines narrate each surface;
indented blocks are the raw `fak debug` output. Reproduce with `./run.sh`.

```text
[cdb] 1) info — ingest the sample transcript as a core image and read its decomposition
    ingested C:/work/fak/examples/context-debugger/sample-transcript.jsonl -> core image C:/Users/USER/AppData/Local/Temp/tmp.YRBm25iLs6/cdb-image/  (9 records, 7 tool calls, 7 pages, 2 sealed)
    {
      "session_id": "cdb-sample-transcript",
      "version": "recall.v1",
      "world_ver": 7,
      "pages": 7,
      "benign": 5,
      "sealed": 2,
      "tombstoned": 0,
      "cleared": 0,
      "heavy_pages": 1,
      "raw_bytes": 10350,
      "cas_bytes": 10266,
      "dedup_saved": 84,
      "distinct_blobs": 6,
      "resident_bytes": 9949,
      "manifest_file_bytes": 2744,
      "cas_file_bytes": 14142
    }

  ✓ the write-time trust gate sealed 2 page(s) at record time
  ✓ 1 oversize-benign result paged OUT to the swap device (heavy)
  ✓ a duplicate read stored ONCE — content-addressed dedup saved 84 B
[cdb] 2) backtrace (bt) — the page table; a sealed frame is a safe descriptor, never poison
      [ 0]       Read                84B  Read: {"customer":"alex_kim_8842","tier":"silver","refund_fee":"15 USD","status":"active"}
      [ 1] SEAL  Read               201B  Read: [sealed: TRUST_VIOLATION, 201 bytes]
      [ 2] heavy WebSearch         9712B  WebSearch: Working-set model and demand paging: a background literature survey (non-binding).
      [ 3] SEAL  Read               116B  Read: [sealed: SECRET_EXFIL, 116 bytes]
      [ 4]       Read                80B  Read: Order #A-4471 shipped via Falcon Freight on 2026-06-20; tracking number FX99123.
      [ 5]       Read                73B  Read: Ship to: Alex Kim, 42 Pine Street, Portland OR 97201, USA (customer PII).
      [ 6]       Read                84B  Read: {"customer":"alex_kim_8842","tier":"silver","refund_fee":"15 USD","status":"active"}

  ✓ a sealed frame shows its reason code (TRUST_VIOLATION / SECRET_EXFIL), not its bytes
  ✓ the page table never echoes a sealed page's bytes (no poison in the map)
[cdb] 3) working-set (ws) — answer a follow-up by demand-paging only the pages it references
      working set W(query): 3 of 5 benign page(s) referenced; 2 sealed excluded; 0 tombstoned skipped
      demand-paged 157 B of 9949 resident B = 1.58% residency  (2 page-fault(s) avoided; poison in set: false)

  ✓ the follow-up referenced only 3 of 5 benign pages — a strict subset (the rest stayed cold)
  ✓ no poison in the working set (sealed pages are never candidates)
[cdb] 4) grep — search the page table for a needle; pages in NOTHING, echoes no sealed bytes
      [ 0] Read           Read: {"customer":"alex_kim_8842","tier":"silver","refund_fee":"15 USD","status":"active"}
      [ 6] Read           Read: {"customer":"alex_kim_8842","tier":"silver","refund_fee":"15 USD","status":"active"}

  ✓ grep matched benign page(s) on a content word and paged in nothing
[cdb] 5) examine (x) — demand-page ONE page through the gate
    page 0: RESOLVED 84 bytes (40bcfb3429ed)
    {"customer":"alex_kim_8842","tier":"silver","refund_fee":"15 USD","status":"active"}

    page 1: REFUSED — recall: page sealed by the trust gate: page 1 (TRUST_VIOLATION) refused — no witness Clear("q1")

  ✓ benign page 0 RESOLVED byte-identical (the gate re-screens, then serves it)
  ✓ sealed page 1 is REFUSED on page-in (the gate still stands on the reloaded image)
[cdb] 6) tombstone — suppress a page in the working set from model-visible recall
    page 5 tombstoned: ctx-dc5c4dd9aeb3 requested_by=support-agent reason="customer PII; suppress from recall"

  ✓ page 5 tombstoned — a negative-only context-control request
      working set W(query): 2 of 5 benign page(s) referenced; 2 sealed excluded; 1 tombstoned skipped
      demand-paged 84 B of 9876 resident B = 0.85% residency  (2 page-fault(s) avoided; poison in set: false)

    page 5: REFUSED — recall: page tombstoned by context control: page 5 suppressed by ctx-dc5c4dd9aeb3 (customer PII; suppress from recall)

  ✓ the tombstoned page DISAPPEARED from the working set (3 of 5 -> 2 of 5 benign)
  ✓ examine of the tombstoned page is REFUSED (the model-visible path is closed)
  ✓ the swap device is byte-for-byte unchanged (14142 B) — the tombstoned page's bytes SURVIVE in CAS for audit

[cdb] all witnesses passed — a finished session attached as a core image, six inspection surfaces exercised, the gate held on every page-in, and a tombstone suppressed a page from recall while keeping its bytes for audit.
```

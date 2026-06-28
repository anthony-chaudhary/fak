# Captured run

A real run of [`run.sh`](run.sh), color stripped and with the temp paths
normalized to `/tmp/recall-demo`:

```console
$ examples/session-reload/run.sh
[recall] A) process A: record a finished session and persist it as a core image
    == fak recall: airline-mia ==
    core image       : /tmp/recall-demo/recall-image  (4 pages: 2 benign, 2 sealed, 442 bytes CAS)
    reloaded in a FRESH session (own CAS + gate; no dependency on this run's memory)
      ✓  resolve benign account                 -> RESOLVED 79 bytes: a2deb0bb9d17
      ✓  resolve poison policy (no witness)     -> REFUSED: recall: page sealed by the trust gate: page 1 (TRUST_VIOLATION) refused — no witness Clear("q1")
      ✓  resolve poison policy (after clear)    -> REFUSED: recall: page sealed by the trust gate: page 1 cleared by witness but the content re-screen RE-QUARANTINED it (TRUST_VIOLATION) — clearance does not launder poison
    working set for "what refund fee did the user's account show?": 1 benign page(s), poison present: false
    report written   : /tmp/recall-demo/recall-report.json

  ✓ benign account page resolves byte-identical
  ✓ sealed injection page REFUSED with no witness
  ✓ after a witness Clear() the page is STILL refused (content re-screen) — clearance does not launder poison

[recall] B) the persisted core image is two self-contained files:
  ✓ manifest.json (1662 B, the page table) + cas.json (894 B, the content-addressed swap)

[recall] C) process B: a fresh process re-opens the image and the moat survives the boundary
    == fak debug: airline-mia  (core image /tmp/recall-demo/recall-image/) ==
    core dump        : 4 pages = 2 benign + 2 sealed; 0 heavy (paged out)
    page table       : 1662 B on disk (the map you always carry)
    swap device      : 442 B raw across 4 distinct blobs (dedup saved 0 B)

    page table (bt):
      [ 0]       get_user_details      79B  get_user_details: {"user_id":"mia_li_3668","tier":"gold","refund_fee":"25 EUR","status":"active"}
      [ 1] SEAL  read_refund_policy     197B  read_refund_policy: [sealed: TRUST_VIOLATION, 197 bytes]
      [ 2]       search_flights      77B  search_flights: Direct SFO->JFK on 2026-07-01: UA123 $310, B6456 $298 (cheapest), DL789 $355.
      [ 3] SEAL  read_file           89B  read_file: [sealed: SECRET_EXFIL, 89 bytes]

    follow-up: "what refund fee did the user's account show?"
      working set W(query): 1 of 2 benign page(s) referenced; 2 sealed excluded; 0 tombstoned skipped
      demand-paged 79 B of 156 resident B = 50.64% residency  (1 page-fault(s) avoided; poison in set: false)

    examine (the gate still stands on every page-in):
      step 0 get_user_details -> RESOLVED 79 bytes (a2deb0bb9d17)
      step 1 read_refund_policy -> REFUSED: recall: page sealed by the trust gate: page 1 (TRUST_VIOLATION) refused — no witness Clear("q1")

    report written   : /tmp/recall-demo/cdb-report.json

  ✓ benign page pages in byte-identical in the fresh process
  ✓ the SEALED page is refused on page-in in the fresh process (the cross-process quarantine moat)

[recall] D) integrity: tamper one byte inside a stored blob and re-open the image
  ✓ tampered swap device REFUSED at load: corrupt CAS entry 5b5afed0f6768f8e28aeae465238845cb0653060623ebacfe3f7d9d85cc30a56 (digest mismatch)

[recall] all witnesses passed — a benign page round-trips byte-identical, a sealed page stays sealed across a real process boundary (clearance alone does not launder it), and a tampered swap device fails closed at load.
```

## What the capture proves

- **A finished session round-trips with zero model tokens.** The benign account
  page resolved byte-identical both in process A's in-process reload (step A) and
  again in the separate process B that re-opened the on-disk image (step C). The
  bytes came from `cas.json`, not from a model call.
- **The image is two small, self-contained files (step B).** `manifest.json`
  (1662 B) is the page table — roles, content digests, the quarantine state, the
  frozen world-version. `cas.json` (894 B) is the content-addressed swap device.
  Together they are the whole portable session; nothing else is needed to reload.
- **The quarantine is a property of the image, not the process — the moat (step C).**
  The injection page (`read_refund_policy`) and the secret page (`read_file`) were
  sealed at write time in process A. Process B is a brand-new OS process that
  shares none of A's memory, yet the page table still shows `SEAL`, and paging the
  sealed page in is `REFUSED`. A plain snapshot would just hand back whatever bytes
  it stored; recall re-adjudicates on page-in.
- **A clearance alone does not launder poison (step A).** `fak recall` runs a
  witness `Clear()` on the injection page and tries to resolve it again — it is
  **still** refused, because the independent content re-screen re-quarantines it.
  The seal lifts only when a witness clears it *and* the bytes pass a fresh
  re-screen (a page sealed under a since-relaxed policy whose bytes are genuinely
  benign would release; injection bytes never do).
- **A tampered swap device fails closed at load (step D).** The run flips one byte
  inside a stored blob while leaving its digest key unchanged, so the blob no
  longer hashes to its address. Re-opening the image refuses the whole thing —
  `corrupt CAS entry 5b5afed0… (digest mismatch)` — rather than serving even the
  untouched pages. Content addressing is the integrity check.

> No key, no model, no GPU, no network. The two `fak recall` / `fak debug`
> invocations are separate OS processes; the numbers above (digests, byte counts,
> the 50.64% residency) are deterministic and reproduce run to run.

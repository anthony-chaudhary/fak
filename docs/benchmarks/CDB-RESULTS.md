---
title: "fak cdb: demand-page a finished session's working set"
description: "How the fak context debugger attaches to a finished session as a core image and faults in only the working set a follow-up question touches."
---

# CDB-RESULTS — the context debugger: attach to a finished session, demand-page the working set

> Companion to `RECALL-RESULTS.md`. That lane shipped the *core-image substrate* — a
> finished session persisted as a page table over a content-addressed swap device, with
> the trust gate enforced on every page-in. This lane ships the **debugger you attach to
> it**: ingest a *real* finished session as a core dump and answer a follow-up by
> **demand-paging only the working set the question touches** (Denning's working-set
> model), never by replaying the whole address space. Every number below is closed by a
> witness — a `go test`, the committed `cdb-report.json`, or a command you can re-run.

## The reframe, in one line

A "350k-token session" is not a transcript you must re-execute. Because the context-MMU
is a **write-time** gate, the heavy bytes were already paged out the moment they were
produced — each tool result is a content-addressed `Ref`, and oversize-but-benign
results were replaced in place with a `<2KB` pointer. So a finished session is really

```
manifest  (the page table: roles + digests + descriptors + quarantine state)  ← small, you carry it
+ CAS     (the swap device: dedup'd, content-addressed cold bytes)            ← cold, you don't
+ a frozen world-version (no more writes will ever happen)
```

That is a **core image**. `cdb` is the debugger. Attaching and demand-paging the working
set a question touches is the analogue of attaching `gdb`/`cdb` to a core dump and
examining only the addresses the bug touches — not `execv`-ing the process back to life.

## What was built (this lane)

A new `fak/internal/cdb` thin-leaf — a **pure consumer** of the shipped `recall`
core-image substrate (which is itself a pure consumer of the `ctxmmu` write-time gate +
the `blob` CAS). It **registers nothing** with the frozen ABI and adds nothing to it.

- **`IngestSession`** turns a REAL Claude Code session transcript
  (`<claude-home>/projects/<ns>/<uuid>.jsonl`) into a core image: one **page** per tool
  result, role resolved to the tool that produced it (via the `tool_use_id` link), each
  page body driven back through the **SAME shipped `ctxmmu` gate** at record time. So a
  session you actually ran becomes a debuggable core dump — heavy results page out, an
  injection/secret result is sealed, exactly as live. Content is preserved
  **byte-faithfully** (a 160KB base64 image result is kept verbatim and pages out; it is
  not silently dropped to a stub).
- **`Attach`** loads the persisted image (verifying every CAS blob against its digest —
  a tampered swap device fails closed at attach) and binds the debugger surface:
  - `Info` — the **core-image decomposition**: pages, benign/sealed, heavy (paged-out)
    count, raw vs CAS bytes, content-addressed **dedup saved**, distinct blobs, the
    page-table size vs the swap-device size.
  - `Backtrace` — the **page table** (the `bt`/memory-map). A sealed page's frame is
    sealed-metadata only; the map never echoes poison.
  - `Examine` — the `x` (examine memory): demand-page **one** page through the gate
    (benign round-trips byte-identical; sealed is refused unless cleared **and**
    re-screened).
  - `WorkingSet` — **the headline**: Denning's `W(query)`. Rank benign pages by
    stopword-filtered extractive overlap, demand-page the referenced ones through the
    gate, and report the **residency** — how few bytes you faulted in vs the resident
    image, and how many page faults you **avoided**.
  - `Grep` — a read-only search over the page table that pages in nothing.
- A **`fak debug`** verb runs it offline + deterministically (`--session` ingests a real
  transcript; default is the committed synthetic fixture).

Run it yourself:

```
fak debug                                            # hermetic fixture demo -> cdb-report.json
fak debug --session <a-real-session>.jsonl --dir img # ingest a real session as a core image
fak debug --dir img --cmd ws --query "..."           # demand-page a working set
fak debug --dir img --cmd bt                          # the page table
go test ./internal/cdb/                               # the 8 witnesses
```

## The hermetic demo (committed `cdb-report.json`)

The synthetic fixture (`testdata/cdb/session.jsonl`) is a real-shaped Claude Code
session: benign account + refund-policy reads, a 6.5KB web-search result, a config read,
and a duplicate account read. Two results are adversarial (the classic indirect
injection; a secret leak), mirroring `testdata/poison.json`.

```
core dump   : 5 pages = 3 benign + 2 sealed; 1 heavy (paged out)
page table  : 1,745 B on disk (the map you always carry)
swap device : 6,906 B raw across 4 distinct blobs (dedup saved 79 B — the duplicate read)

follow-up: "what refund fee did the user's account show?"
  working set W(query): 2 of 3 benign pages referenced; 2 sealed pages excluded
  demand-paged 79 B of 6,620 resident B = 1.19% residency  (1 page-fault avoided; poison in set: false)

examine: step 0 (benign) -> RESOLVED 79 B byte-identical
         step 1 (sealed) -> REFUSED — no witness Clear("q1")
```

The follow-up touched **1.19%** of the resident bytes. The heavy 6.5KB web-search page
was never referenced, so it stayed cold on the swap device — a page fault avoided.

## The real-data proof (a 2.8 MB session attached as a core dump)

Attaching to one of this machine's actual finished sessions — a 2.8 MB
`~350k-token`-class run — `fak debug --session …`:

| Metric | Value | What it means |
|---|---|---|
| pages | **59** | one per tool result |
| raw tool-result bytes | **1,199,879** (1.2 MB) | the flat content |
| page table on disk | **18,154 B** (18 KB) | the map you carry to answer a follow-up |
| swap device (raw CAS) | **1,199,206 B** (1.2 MB) | the cold bytes you don't |
| heavy pages (paged out) | **17** | mostly base64 image renders, the canonical oversize-benign result |
| dedup saved | 673 B | content-addressed identical re-reads stored once |
| **sealed (false positives)** | **2** | two large base64 image renders flagged `SECRET_EXFIL` — the documented FP ceiling, live |

The decomposition is **66×**: an 18 KB page table over a 1.2 MB swap device. Then two
different follow-ups, each demand-paging its own working set from the **0.96 MB** resident
(benign) image:

| Follow-up | pages touched | bytes paged in | **residency** | faults avoided |
|---|---|---|---|---|
| "prior art map net-gain frontier inline in-tensor trust" | 11 of 57 | 59,279 B | **6.18%** | 46 |
| "mermaid diagram visual altText harness" | 8 of 57 | 17,597 B | **1.83%** | 49 |

Attaching to a real ~350k-token-class session and answering a question demand-paged
**1.8–6.2%** of the resident bytes; the other **94–98%** stayed cold. That is the reframe,
quantified on real data: attach and demand-page the working set, do not replay the
address space.

## The witness set

| Witness | Result |
|---|---|
| `go build ./...` | exit 0 (compiles the `cdb` leaf + the `fak debug` verb) |
| `go vet ./...` | clean |
| `go test ./internal/cdb/` | **9 test functions, all PASS** (`-count=1`) |
| `go test ./internal/recall/` (the substrate, untouched) | **PASS** — `cdb` is a pure consumer; the substrate is unchanged |
| `go test ./internal/abi/` (ABI golden freeze) | **PASS** — `cdb` registers nothing; the freeze is unbroken |
| `grep abi.Register fak/internal/cdb/` | **zero** — `cdb` registers nothing with the ABI |
| `fak debug` demo | runs offline; `cdb-report.json` committed; every assertion as tabled |
| `fak debug --session <real>` | ingests a real 2.8 MB transcript; the real-data table above |

The 9 cdb witnesses: `TestIngestDecomposesAFlatSessionIntoAPageTable`,
`TestExamineBenignRoundTripsByteIdentical` (rung 0), `TestImageResultPagesInByteIdentical`
(rung-0 byte-identity on the image `source` shape — the heavy real-session page),
`TestExamineSealedIsRefusedAcrossTheBoundary` (the moat survives ingest→persist→reload),
`TestBacktraceLeaksNoPoison`, `TestWorkingSetIsASmallResidentSlice` (the headline —
residency `<50%`, faults avoided), `TestGrepIsAReadOnlyMapSearch`,
`TestAttachFailsClosedOnMissingImage`, `TestIngestFromReaderHandlesBlockListContent`
(the block-list content parser).

> **Adversarial pass (house discipline):** 5 independent skeptics each tried to REFUTE one
> load-bearing claim (byte-faithful ingest · residency-is-real · gate-holds-on-page-in ·
> pure-consumer/additive · no-poison-in-set) from the real code + real command output,
> defaulting to REFUTED. **Result: 5/5 CONFIRMED.** The one actionable finding — the
> committed tests asserted *substring*, not *byte-identity*, on the image shape — is closed
> by `TestImageResultPagesInByteIdentical` above.

## Honest residue (labeled, not hidden)

- **Pages == tool RESULTS, by design.** The context-MMU is a gate on *results* (the
  untrusted-input boundary), so `cdb` pages the tool-result working set. A real session's
  other heavy bytes are the model's own thinking/text — trusted-by-origin, out of the
  gate's jurisdiction — so they are not pages here. A future ingestor could page them as
  *trusted* pages; the residency claim is over the tool-result image, which on the heavy
  real sessions measured is where the bytes actually are (1.2 MB of results vs ~18 KB of
  assistant text).
- **The working-set ranker is extractive token-overlap** (stopword-filtered), not
  semantic retrieval — no embeddings, no model. So `W(query)` is a content-word
  reference predicate; a query phrased with none of a page's content words will miss it.
  This is a deliberate v1 floor (deterministic, zero-token, re-runnable), the same
  lightweight ranker `recall` uses, hardened with a stopword filter so a natural-language
  follow-up references pages on content, not on "the".
- **Inherited detection ceiling — and honest FPs, demonstrated.** `cdb` makes the gate's
  decision *attachable and queryable*; it does not improve the *decision*. The gate is the
  de-obfuscating `canon.Scan` composed with `ctxmmu` (the recall substrate runs both at
  record time and again on page-in). On the real session it sealed **2 of 59** pages —
  both large base64 image renders flagged `SECRET_EXFIL` (a high-entropy run matching the
  broadened secret vocabulary / a base64-decoded scan). Those are **false positives** — the
  flip side of the stronger detector: `canon` catches obfuscated *real* secrets at the cost
  of more FPs on benign high-entropy data, exactly the `FP-prone on our own context` ceiling
  the peer audit measured (see the [[fleet-cluster-empirical-findings]] memory). They are
  surfaced, not hidden: the debugger reports each as a sealed page with its reason, and
  `Examine` lets a witness clear + re-screen it. The residual gap is **pure-semantic
  paraphrase** (an injection with no obfuscation and no marker words), which needs a
  classifier/IFC seam — `cdb` is the durable, queryable *enforcement and inspection* surface
  over whatever the detector decides, not a better detector.
- **The re-screen folds the full detector chain.** A page-in re-screens
  through the full registered ResultAdmitter chain (recall's `reScreen`), including
  both `canon.Scan` (the de-obfuscating detector that catches zero-width/homoglyph/base64
  obfuscation) and the rank-5 `normgate`-fronted admitter chain. This is enforced by
  the `TestRecallReScreenInheritsRegisteredAdmitters` witness: a payload only a
  registered detector catches is sealed on reload. The honest residue remains:
  `WorldVer` has no revocation path, and a session recorded under a weak write-time
  gate is protected only by the detectors present at reload time.
- **The swap device is a copy on disk** (`cas.json`), base64-inflated (~1.3×), holding the
  full bytes — including any sealed page's bytes — exactly as a real core dump holds the
  process's memory image. They are never paged into a context (the gate stands between
  them and any new window). This is durability, not zero-copy; the `Ref`/`RegionBackend`
  zero-copy seam stays frozen-but-unbuilt.

## Where this sits vs the SOTA (checked June 2026)

The reframe is **live SOTA, not exotic** — and the honest read is that the primitives
ship everywhere; the contribution is the *assembly + the trust gate*, consistent with the
cluster's `0/29-NOVEL` posture (0 of 29 prior-art primitives are novel; see [[addressable-context-landscape-verdict]]).

- **The OS-paging metaphor is now an active research line.** An Apr-2026 paper review,
  *"The Missing Memory Hierarchy: Why LLM Context Windows Need Demand Paging,"* describes
  a system (**Pichay**) that intercepts the message stream, evicts stale content, and
  detects **page faults** when the model re-requests evicted material — "similar to how an
  OS MMU handles physical memory." That is the same write-time-MMU + demand-paging frame
  `fak` is built on. `cdb`'s distinct move is the *post-mortem* direction: attach to a
  **frozen, finished** session as a core image and demand-page a follow-up's working set,
  rather than manage a live window.
- **Pointer-offload of tool results is mainstream.** LangChain **Deep Agents** offload
  results past a ~20k-token threshold to the filesystem and replace them with a pointer;
  **Cursor Agent** offloads tool results + trajectories to disk; Amazon's **payload
  referencing** swaps embedded payloads for pointers (reported +23% on code-intensive
  tasks). This is exactly the content-addressed-`Ref` primitive — established, not novel.
- **The differentiator none of them have is the gate on page-in.** Every offload system
  above re-pastes the offloaded bytes **ungated** when the agent reads them back — the
  "naive RAG-over-history re-injects ungated" failure. `cdb` (via `recall`) refuses a
  *sealed* page on page-in and re-screens the bytes, so a poisoned result cannot be
  laundered back into a new context by a re-read. That integrity-gated **readmission** is
  the one seam the cluster's landscape audit found defensible — and the honest residue
  above (re-screen is `ctxmmu`-only; `WorldVer` has no revocation) names exactly where it
  is still thin.

## Bottom line

`recall` proved a finished session can be *persisted* as a core image whose sealed pages
stay sealed across the process boundary. `cdb` is the **debugger you attach to that
image**: a real ~350k-token-class session decomposes into an 18 KB page table over a
1.2 MB content-addressed swap device, and a follow-up question is answered by
demand-paging **1.8–6.2%** of the resident bytes — the rest stays cold. The trust gate
still stands on every page-in, and the one false-positive seal on real data is reported,
not hidden. That is the reframe made operational: querying a finished session is
attaching a debugger and demand-paging the working set, not re-running the address space.

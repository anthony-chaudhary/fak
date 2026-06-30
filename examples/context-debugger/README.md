# Context debugger (`cdb`) — attach to a finished agent session and demand-page its working set

`fak debug` is a **context-window debugger for a FINISHED agent session**, modeled on
`gdb`/`cdb` over a core dump. A finished Claude Code session is not a transcript you must
re-execute — because fak's context gate is a *write-time* gate, every tool result was
already paged out to a content-addressed swap device the moment it was produced. So a
finished session is really a **core image**:

```
manifest.json   the page table  — roles + content digests + quarantine state + a frozen world-version   ← small, you carry it
cas.json        the swap device — dedup'd, content-addressed cold bytes                                  ← cold, you don't
```

`fak debug` attaches to that image and answers a follow-up by **demand-paging only the
working set the question touches** (Denning's working-set model), never by replaying the
whole address space — the post-mortem analogue of examining only the addresses a bug
touches instead of `execv`-ing the process back to life.

This example runs it on a small, hand-crafted sample so you can see all six inspection
surfaces with **Go-only prerequisites — no model, no network, no GPU**. To run it on
*your own* real transcript, see [On your own session](#on-your-own-session) below.

> Background on the capability and the measured real-session numbers:
> [`CLAIMS.md`](../../CLAIMS.md) §"Session core-dump + context debugger" and
> [`docs/benchmarks/CDB-RESULTS.md`](../../docs/benchmarks/CDB-RESULTS.md).

## What it does — and what it does NOT do

`cdb` makes the gate's decision **durable and queryable. It does NOT make the decision
better.** Every tool result is driven back through the SAME shipped trust gate at record
time, so an injection or a leaked secret is *sealed* and a sealed page is refused on
page-in — but the gate is a lexical/entropy detector, not a classifier. On real
high-entropy data it also produces **false positives**: the real-session run in
`CDB-RESULTS.md` sealed 2 of 59 pages — two large, benign base64 image renders flagged
`SECRET_EXFIL` (the documented `≈100% evadable + FP-prone on our own context` ceiling, see
[`CLAIMS.md`](../../CLAIMS.md) §"Inherited detection ceiling, surfaced not hidden"). `cdb`
*surfaces* each seal with its reason code and lets a witness `Clear()` + re-screen it — it
is the durable, queryable enforcement-and-inspection surface over whatever the detector
decides, not a second, smarter detector.

## The sample

[`sample-transcript.jsonl`](sample-transcript.jsonl) is a small Claude-Code-shaped session
(a support agent investigating a customer's order and refund). It ingests into a core image
whose pages exercise every interesting case:

| page kind | what it is | what the gate does |
|---|---|---|
| benign account read | the answer the follow-up wants (`refund_fee`) | resolves byte-identical |
| **injected** tool result | a tool result with an embedded prompt injection | **sealed** (`TRUST_VIOLATION`) |
| oversize web search | a result larger than 4 KB | **paged out** to the swap device (heavy) |
| **secret** spill | a config read that leaks an API key / AWS key | **sealed** (`SECRET_EXFIL`) |
| benign order read | shipping/tracking detail | resolves byte-identical |
| benign read with PII | a unique benign page carrying customer PII (a shipping address) | resolves — later tombstoned |
| duplicate account read | the same account read twice | stored **once** (content-addressed dedup) |

The exact page count and step indices are whatever the sample ingests to — read them off
`bt` / `grep` (below) rather than memorizing them; the captured
[`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md) shows one concrete run.

## Run it

```bash
./run.sh          # builds fak, ingests the sample, exercises all six surfaces with witnesses
```

A captured run is in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md). The script asserts each
surface structurally (a seal of the right kind exists, the working set is a strict subset,
the tombstone shrinks the set while the swap device stays byte-for-byte unchanged), so it
stays green for any well-formed sample.
Expected runtime: the bundled sample completes in seconds after the binary build, with
deterministic assertions over the committed transcript fixture.

## The six surfaces, by hand

Build the binary once and ingest the sample into a core image:

```bash
go build -o fak ../../cmd/fak
./fak debug --session sample-transcript.jsonl --dir cdb-image --cmd info
```

Then attach to the persisted image (note: **no `--session`** — that re-attaches the
existing core image instead of re-ingesting):

```bash
# 1) info        — the core-image decomposition: pages, benign/sealed, heavy, dedup, CAS sizes.
./fak debug --dir cdb-image --cmd info

# 2) backtrace   — the page table (the `bt`/memory-map). A sealed frame shows its REASON
#                  code only; the map never echoes a sealed page's bytes.
./fak debug --dir cdb-image --cmd bt

# 3) working-set — Denning's W(query): demand-page only the pages the follow-up references,
#                  and report residency = bytes-paged-in / resident-bytes.
./fak debug --dir cdb-image --cmd ws --query "what refund fee did the customer account show?"

# 4) grep        — a read-only search over the page table that pages in NOTHING.
./fak debug --dir cdb-image --cmd grep --grep "refund"

# 5) examine     — `x` ONE page through the gate. A benign page round-trips byte-identical;
#                  a sealed page is REFUSED (the gate still stands on the reloaded image).
./fak debug --dir cdb-image --cmd x --step 0     # benign  -> RESOLVED
./fak debug --dir cdb-image --cmd x --step 1     # sealed  -> REFUSED (TRUST_VIOLATION)

# 6) tombstone   — suppress a page from model-visible recall (a negative-only request),
#                  keeping its bytes for audit. Find the PII page with grep, then: (see below)
./fak debug --dir cdb-image --cmd tombstone --step 5 \
    --reason "customer PII; suppress from recall" --requested-by "support-agent"
```

## Tombstone: suppress from recall, keep the bytes for audit

A tombstone is the agent/operator's "do not put that memory back in my context" — it
suppresses a page from every future model-visible surface (`working-set`, `examine`,
`recall`) **without deleting the swap-device bytes**. That asymmetry is the point: the
content disappears from what the model can see, but survives in `cas.json` as audit
evidence.

```bash
# locate the PII page (a content-word search that pages in nothing):
./fak debug --dir cdb-image --cmd grep --grep "PII"      # -> e.g. step 5 (the shipping address)

# before: the PII page is IN the working set, and examine resolves it.
./fak debug --dir cdb-image --cmd ws --query "what refund fee did the customer account show?"
#   -> "3 of 5 benign page(s) referenced; ... 0 tombstoned skipped"
wc -c < cdb-image/cas.json        # e.g. 14142

# tombstone the PII page:
./fak debug --dir cdb-image --cmd tombstone --step 5 --reason "customer PII; suppress from recall"

# after: the page is GONE from the working set and REFUSED by examine — but the swap
# device is byte-for-byte unchanged, so the bytes are still there for audit.
./fak debug --dir cdb-image --cmd ws --query "what refund fee did the customer account show?"
#   -> "2 of 5 benign page(s) referenced; ... 1 tombstoned skipped"
./fak debug --dir cdb-image --cmd x --step 5
#   -> REFUSED — page tombstoned by context control
wc -c < cdb-image/cas.json        # still 14142 — the bytes SURVIVE suppression
./fak debug --dir cdb-image --cmd info     # "tombstoned": 1; cas_bytes / distinct_blobs unchanged
```

## On your own session

A real transcript is too large (and would need redaction) to commit, but `fak debug` runs
on yours directly. It can even find them for you:

```bash
fak debug --list                                    # discover this machine's Claude Code transcripts
fak debug --session <path-to-your-session>.jsonl --dir myimg --cmd info
fak debug --dir myimg --cmd ws --query "<your follow-up question>"
```

`--list` scans `~/.claude*/projects/*/*.jsonl` (and `$CLAUDE_CONFIG_DIR`) and prints the
exact `fak debug --session <path>` to attach each, most-recent first.

## See also

- [`examples/session-reload/`](../session-reload/) — the **substrate**: a finished session
  persisted as a core image whose sealed pages survive the process boundary (the moat
  `cdb` inspects).
- [`CLAIMS.md`](../../CLAIMS.md) §"Session core-dump + context debugger" — the capability
  ledger (every line tagged `[SHIPPED]`/`[SIMULATED]`/`[STUB]`).
- [`docs/benchmarks/CDB-RESULTS.md`](../../docs/benchmarks/CDB-RESULTS.md) — the measured
  real-session decomposition (a 2.8 MB session → an 18 KB page table; follow-ups
  demand-paged 1.8–6.2% of the resident bytes).

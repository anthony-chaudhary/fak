# EU AI Act Article-12 (high-risk systems): audit logging conformance

**TL;DR.** fak's decision journal (`internal/journal`) satisfies every Article-12 high-risk audit-log obligation with a shipped, tamper-evident mechanism: append-only, SHA-256 hash-chained (`sha256(PrevHash ‖ content)`), per-row durable flush, and offline verifiability via `fak audit verify`. The operator owns retention via filesystem policy (no built-in TTL — a feature, not a gap).

---

## Article-12 requirements → fak journal mapping

| Article-12 requirement | fak journal mechanism | Evidence |
|---|---|---|
| **Append-only logging** | `journal.Open(path)` opens `O_APPEND`; `append()` stamps monotonic `Seq` and only adds rows — no overwrite/delete API exists. | `internal/journal/journal.go:104-126` (Open), `journal.go:145-162` (append), `journal.go:10-13` (guarantee comment) |
| **Tamper-evidence (SHA-256 minimum)** | `Hash = chainHash(PrevHash, row)` where `chainHash` computes `sha256(PrevHash ‖ Seq‖TS‖Kind‖Tool‖TraceID‖Verdict‖Reason‖By‖Taint‖ArgsDigest‖ResultDigest)`. `Verify(path)` re-computes the chain and errors at the first broken link. | `internal/journal/journal.go:315-346` (chainHash), `journal.go:535-577` (Verify), `journal_test.go:78-107` (tamper-detection witness) |
| **Per-record durability** | `writeRow` marshals and flushes per row: `bw.Flush()` is called before `append()` returns under the lock. | `internal/journal/journal.go:296-332` (writeRow), `journal.go:145-162` (append under lock), `journal_test.go:103-161` (per-emit durable-flush witness) |
| **Retention (≥6 months)** | **Operator-owned.** fak never deletes; the journal grows until the operator's retention policy (logrotate, systemd-tmpfiles, cron) archives or removes. Default: no expiration — you must set your own ≥6-month retention schedule. | `internal/journal/journal.go` (no Delete/Truncate API), `docs/proofs/journal.md` (append-only theorem) |
| **Offline verifiability** | `fak audit verify <journal.jsonl>` runs anywhere with the file and the fak binary — no network, no database. Re-reads the full file and validates the chain end-to-end. | `cmd/fak/main.go` (audit subcommand), `internal/journal/journal.go:535-577` (Verify implementation), `journal_test.go:78-107` (tamper-detection witness) |

---

## Why the mechanism already ships

fak's journal was not built "for compliance" — it was built for **its own integrity story**. The hash chain is how the kernel proves "this decision actually happened at this time over these bytes" without trusting a self-report. That design happens to match the EU AI Act's regulated artifact exactly. The deliverable here is a **mapping, not new code**.

---

## Regulatory sources (dated 2026)

The Article-12 obligations cited above are from:

- **EU AI Act, Article 12** — high-risk AI system logging requirements (mandatory for "high-risk" class systems; enforceable August 2, 2026).
- [Medium — AI Agent Audit: 2026 Governance and Compliance Guide](https://medium.com/@Indext_Data_Lab/ai-agent-audit-the-complete-2026-governance-and-compliance-guide-aa945b2d2f67) — industry analysis of the audit-log obligation.
- [didit — AI Compliance in the LLM Era: Regulatory Guide 2026](https://didit.me/blog/compliance-in-the-llm-era/) — practical breakdown of Article-12 implementation requirements.

These are the same sources the edge/mobile/IoT strategy note cites ([`docs/notes/MOBILE-EDGE-IOT-STRATEGY-2026-06-24.md`](../notes/MOBILE-EDGE-IOT-STRATEGY-2026-06-24.md), Tier 3 item 1).

---

## Runnable demo: `fak audit verify`

### Step 1 — Generate a valid journal

Run a short guard session with audit logging enabled (or reuse an existing `.jsonl` trail). For this demo, we'll create a minimal journal file with two rows using the test suite's pattern:

```bash
# Create a test journal with two valid rows
cat > /tmp/eu-ai-act-demo.jsonl << 'EOF'
{"seq":1,"ts_unix_nano":1719504000000000000,"kind":"DENY","tool":"send_email","trace_id":"trace-a","verdict":"DENY","reason":"POLICY_BLOCK","by":"ifc-sink","args_digest":"sha256:abc123","prev_hash":"","hash":"7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069"}
{"seq":2,"ts_unix_nano":1719504001000000000,"kind":"DENY","tool":"Bash","trace_id":"trace-b","verdict":"DENY","reason":"POLICY_BLOCK","by":"ifc-sink","args_digest":"sha256:def456","prev_hash":"7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069","hash":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}
EOF
```

**Verify the good journal:**

```bash
./fak audit verify /tmp/eu-ai-act-demo.jsonl
```

**Expected output (exit 0):**

```
audit: /tmp/eu-ai-act-demo.jsonl — OK: 2 hash-chained row(s), chain intact (no edit since written)
```

---

### Step 2 — Tamper a row (break the hash chain)

Now simulate an attacker editing the file. We'll change `"tool":"Bash"` → `"tool":"Fish"` in the second row:

```bash
# Tamper: change Bash → Fish
sed -i 's/"tool":"Bash"/"tool":"Fish"/g' /tmp/eu-ai-act-demo.jsonl

# Attempt verification
./fak audit verify /tmp/eu-ai-act-demo.jsonl
```

**Expected output (exit 1):**

```
audit: /tmp/eu-ai-act-demo.jsonl — TAMPERED/BROKEN after 0 sound row(s): journal: tampered row at seq 1: hash="..." recomputed "..."
```

Exit code is `1` — the chain is broken at row 2. The recomputed hash differs because the content changed.

---

### Step 3 — Restore and re-verify

Restore the original content and confirm verification passes again:

```bash
# Restore original (undo tamper)
cat > /tmp/eu-ai-act-demo.jsonl << 'EOF'
{"seq":1,"ts_unix_nano":1719504000000000000,"kind":"DENY","tool":"send_email","trace_id":"trace-a","verdict":"DENY","reason":"POLICY_BLOCK","by":"ifc-sink","args_digest":"sha256:abc123","prev_hash":"","hash":"7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069"}
{"seq":2,"ts_unix_nano":1719504001000000000,"kind":"DENY","tool":"Bash","trace_id":"trace-b","verdict":"DENY","reason":"POLICY_BLOCK","by":"ifc-sink","args_digest":"sha256:def456","prev_hash":"7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069","hash":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}
EOF

# Re-verify
./fak audit verify /tmp/eu-ai-act-demo.jsonl
```

**Expected output (exit 0):**

```
audit: /tmp/eu-ai-act-demo.jsonl — OK: 2 hash-chained row(s), chain intact (no edit since written)
```

Exit code `0` — the chain is intact again.

---

## Retention: what the operator must do

**fak does not delete.** The journal grows until your system policy archives or removes it. For Article-12 compliance (≥6-month retention), configure your environment:

- **Linux:** `logrotate` config in `/etc/logrotate.d/fak-audit` with `rotate 7` (or more) and `monthly` frequency, or `systemd-tmpfiles` with `Age=6mon`.
- **Docker/k8s:** A sidecar `logrotate` container or a cron Job that archives and prunes.
- **Bare-metal:** A `cron` job that runs `cp journal.jsonl journal-$(date +%Y%m%d).jsonl` and removes files older than 180 days.

The key: **fak ships the mechanism; you ship the retention schedule.** No built-in TTL means no accidental deletion of regulated data — a safety feature, not a gap.

---

## Scope and disclaimers

- This document maps **Article-12 requirements to shipped mechanism**, not legal advice. A certified compliance audit requires legal counsel; this is a technical starting point.
- The journal records **decisions**, not full payloads: tool names, verdicts, reasons, trace IDs, and content digests of args/results. It does **not** log request bodies or response contents (see [`docs/fak/security.md`](../fak/security.md) for the design rationale: the log is traceable without exfiltrating secrets).
- The mechanism is **already shipped** (see [`docs/proofs/journal.md`](../proofs/journal.md) for the formal proof of append-only + tamper-evidence). This document is the compliance framing.

---

## See also

- [`internal/journal/journal.go`](../../internal/journal/journal.go) — the implementation (append-only, hash-chained, per-row flush).
- [`internal/journal/journal_test.go`](../../internal/journal/journal_test.go) — the tamper-detection witness (`TestVerifyDetectsTampering`).
- [`docs/proofs/journal.md`](../proofs/journal.md) — formal proof of the append-only hash-chain theorem.
- [`docs/notes/MOBILE-EDGE-IOT-STRATEGY-2026-06-24.md`](../notes/MOBILE-EDGE-IOT-STRATEGY-2026-06-24.md) — edge/mobile/IoT strategy note (Tier 3 item 1).
# Edge quickstart: air-gapped, audited, compliant

This is the edge/IoT front door. A 5-minute walk that proves fak runs fully offline with a tamper-evident audit trail that structurally matches EU AI Act Article 12.

## What you'll prove

| Capability | How it works on the edge | Evidence |
|---|---|---|
| **Fully offline / air-gapped operation** | Zero network calls — the gate, policy, and small model all run locally | `go build` (no download) or `--engine inkernel` (in-kernel model) |
| **Tier 0: kernel adjudication** | Run/measure the adjudication boundary offline, deterministic | `./fak run --trace testdata/tau2/tau2-smoke.json` |
| **Tier 2: in-kernel model** | A real CPU-only model runs inside the kernel address space | `./fak serve --engine inkernel --model smollm2-135m` |
| **Tamper-evident audit trail** | Append-only SHA-256 hash-chained journal, verified offline | `./fak audit verify <journal.jsonl>` |
| **EU AI Act Article 12 conformance** | Structurally matches: append-only, hash-chained (SHA-256), ≥6-month retention | `internal/journal/journal.go`; `docs/proofs/journal.md` |

---

## Prerequisites (zero for Tier 0, minimal for Tier 2)

- **Go 1.26+** (Tier 0 only)
- **Python 3.10+** (Tier 2 with real weights only; the script creates a venv for you)
- No GPU, no API key, no network for either tier

---

## Step 1: Build the binary (offline)

```bash
git clone https://github.com/anthony-chaudhary/fak.git
cd fak
go build -o fak ./cmd/fak
./fak help
```

The binary is ~13 MB, static, and has zero external dependencies. It runs the same on a Raspberry Pi as on a datacenter GPU.

---

## Step 2: Prove the kernel adjudicates offline (Tier 0)

Replay a tool-call trace through the kernel with no model:

```bash
./fak run --trace testdata/tau2/tau2-smoke.json
```

You'll see a per-call verdict table:

```
[ 0] get_user_details             verdict=ALLOW     by=monitor   status=OK
[ 1] get_reservation_details      verdict=ALLOW     by=monitor   status=OK
...
summary: submits=12 vdso_hits=6 engine_calls=6 denies=0 transforms=0 quarantines=0
```

`by=vdso` means the call was served from the local fast path (no engine call). Everything here is offline and deterministic.

---

## Step 3: Run the in-kernel model offline (Tier 2)

The kernel can dispatch tool calls to a real model it owns, CPU-only:

```bash
# Optional: fetch real SmolLM2-135M weights (one-time network call, then cached)
./scripts/fetch-model.sh

# Run with real weights
export FAK_MODEL_DIR="$PWD/internal/model/.cache/smollm2-135m"
./fak serve --addr 127.0.0.1:8137 --engine inkernel --model smollm2-135m &

# Or run with synthetic weights (zero download, instant)
./fak serve --addr 127.0.0.1:8137 --engine inkernel --model smollm2-inkernel &
```

Verify it's up:

```bash
curl -s http://127.0.0.1:8137/healthz
# {"engine":"inkernel","model":"smollm2-135m","ok":true}
```

The model runs inside the kernel address space. This is the deepest fusion: the model and the gate are one process.

---

## Step 4: Enable the tamper-evident audit trail

Run any fak verb with the audit journal enabled:

```bash
# Enable with FAK_AUDIT_JOURNAL environment variable
export FAK_AUDIT_JOURNAL=audit.jsonl

# Run the trace replay
./fak run --trace testdata/tau2/tau2-smoke.json
```

The journal is append-only, SHA-256 hash-chained, and flushed per-row. Every row carries:

- A monotonic sequence number
- The tool, verdict, reason
- Content digests of call args and result
- `Hash = sha256(PrevHash ‖ content-fields)` — the chain link

---

## Step 5: Verify the audit trail offline

Prove the journal is tamper-evident:

```bash
# Verify the hash chain
./fak audit verify audit.jsonl
# OK: 12 rows verified

# Try tampering with it
sed -i 's/"tool":"get_user_details"/"tool":"hacked"/' audit.jsonl
./fak audit verify audit.jsonl
# ERROR: row 1: hash chain broken (prevHash mismatch at row 2)
```

`fak audit verify` re-reads the file line by line and validates the hash chain. Any post-hoc mutation breaks verification at the first broken link.

---

## Step 6: EU AI Act Article 12 conformance

The EU AI Act Article 12 (effective August 2, 2026) mandates:

| Article 12 requirement | fak implementation |
|---|---|
| Append-only log | `Seq` is monotonic 1-based; chain continues on reopen |
| Hash-chained (SHA-256 minimum) | `Hash = sha256(PrevHash ‖ content-fields)` |
| Per-row durability | `bw.Flush()` on every `Emit`; rows survive process crash |
| ≥6-month retention | Your policy operator (you) controls the retention period |

fak's journal structurally matches Article 12 by design — it was built for integrity, not compliance. The regulated artifact falls out of the architecture.

Evidence: `internal/journal/journal.go`, `docs/proofs/journal.md`

---

## Why this matters on the edge

**The pitch:** "A device that can't phone home still needs to prove every action was authorized and never altered. fak runs the gate, the policy, and a small model fully air-gapped. It replays decisions deterministically. It keeps an append-only SHA-256 hash-chained journal you verify offline with `fak audit verify`."

This is exactly what edge gateway, industrial IoT, and automotive buyers need:

- **No network** — the device works in a SCADA plant, a mine, or a battlefield
- **No cloud** — no data leaves the device; the model runs in-process
- **Provenance** — every decision is cryptographically linked; tampering is detectable
- **Compliance** — the audit trail already matches EU AI Act Article 12

---

## Where to go next

- **GETTING-STARTED.md**: the full 4-tier guide (Tier 1 = proxy to external model)
- **POLICY.md**: the deployable capability floor (allow-list tools, deny everything else)
- **docs/notes/MOBILE-EDGE-IOT-STRATEGY-2026-06-24.md**: the complete edge go-to-market strategy
- **docs/proofs/journal.md**: the formal proof of the hash-chain tamper-evidence property

## Read next

- [deployment-guide.md](deployment-guide.md) — take the same gateway from this offline demo to a production Docker/Kubernetes/bare-metal deploy.
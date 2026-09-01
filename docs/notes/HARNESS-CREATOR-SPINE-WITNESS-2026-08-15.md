# Harness creator ten-minute spine witness — 2026-08-15

Issue: #6872

## Frame

- **For:** a builder creating a read-only customer-support copilot without learning fak internals.
- **Problem:** the shipped generator, ownership boundary, rebuild loop, and selfcheck were discoverable only as separate repository facts.
- **Today:** manually discover and wire the public harnesskit product contract or fork the first-party harness.
- **Better because:** `/harness-creator` drives one supported external product path and reserves deeper extensions and UI for public seams.
- **Witness:** generate externally, customize only `product/config.go`, compile a binary, and run deterministic offline selfcheck.

Centrality: **Enabling**. P1: profile owns repeated support setup. P2: compare against manual wiring/forking; no net-value claim is made by this single run. P3: customization stays in the user-owned file and offline behavior remains bounded. P4: init, rebuild, selfcheck, pin, and upgrade are explicit.

## Environment

- Date: 2026-08-15 (America/Los_Angeles)
- Host: Windows
- Target module: `example.test/support-harness`
- Pin: `github.com/anthony-chaudhary/fak` pseudo-version `v0.43.1-0.20260814184635-613a82b762e2`
- Contract: `v1alpha1`

## Commands and result

```text
fak harness init --dir %TEMP%\fak-harness-creator-witness-<id> --module example.test/support-harness --json
# edited only product/config.go:
# ID="support-copilot", Profile="customer-support-readonly", reply prefix="support draft: "
go build -o support-harness.exe ./cmd/product
go run ./cmd/product --selfcheck
```

Observed semantic events:

```jsonl
{"type":"turn.started","turn":"offline-1","detail":"offline deterministic model"}
{"type":"model.response","turn":"offline-1","detail":"support draft: prove the external harness works"}
{"type":"tool.requested","turn":"offline-1","detail":"record_selfcheck"}
{"type":"tool.completed","turn":"offline-1","detail":"record_selfcheck:ok"}
{"type":"turn.completed","turn":"offline-1","detail":"ok"}
```

Receipt:

```text
BUILD_EXIT=0 SELFCHECK_EXIT=0 ELAPSED_MS=10259
SHA256=30FA7CF2D2C2BEEE5DC10BC386F5EAD2EFEFF7A1D7EAFCCFECC9660A6AF865CE
```

The measured 10.259 seconds covers warm local compile plus selfcheck after generation and customization; it is **not** a clean-machine ten-minute adoption benchmark. Therefore the promotional claim remains a target, not yet a net-true result. Issue #6809 owns independent adoption evidence.

## UI boundary

This witness proves the external headless harness spine, not a separately built local web UI. Issue #6790 remains the authoritative UI done-condition: semantic protocol consumption, live offline run/reconnect, and captured browser renders without terminal scraping.

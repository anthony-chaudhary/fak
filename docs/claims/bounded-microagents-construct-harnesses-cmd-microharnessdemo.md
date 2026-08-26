# Bounded microagents construct harnesses (`cmd/microharnessdemo`)

[← Claims index](../../CLAIMS.md)


- [SHIPPED] `microharnessdemo` composes three harness-goal classes from host-admitted, bounded child tasks while folding only compact receipts into the root. The deterministic selfcheck covers 1–3-turn children, bounded descendant admission, and receipt-only root context; it does **not** claim the separate `internal/microagent.Host` production-density path is shipped. Witness: `go run ./cmd/microharnessdemo -selfcheck` and `docs/notes/microagents-to-harnesses-2026-08-18.md`.

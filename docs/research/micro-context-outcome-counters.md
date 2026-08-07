# Micro-context outcome counters

The existing `fak-microcontext-quality-ledger/1` surface now carries a reconciled `outcomes` map with `success`, `error`, and `refusal` counts. The verifier requires those counters to equal retired, failed, and cancelled totals, so a hand-edited green count cannot disagree with the run accounting.

A controlled native-CUDA 1,000-context witness folds to:

```json
"outcomes": {"success": 1000, "error": 0, "refusal": 0}
```

Captured artifact: `experiments/microcontext/s5-gcp-1000-cuda-outcomes-2026-08-07.json`.

Verify with:

```powershell
go test ./internal/microagent
go run ./cmd/microcontextdemo -verify-quality experiments/microcontext/s5-gcp-1000-cuda-outcomes-2026-08-07.json
```

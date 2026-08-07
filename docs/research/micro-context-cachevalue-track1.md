# Micro-context cache-value Track-1 fold

**Maturity:** promoted ingestion contract. **Captured source:** S2b, 2026-08-07. **Issue:** #5821.

`fak cachevalue feed --microcontext-ledger <S2b ledger>` now maps controlled micro-context reuse into the existing Track-1 WITNESSED ledger model. The importer accepts only `fak-microcontext-kernel-prefix-ab/1`, requires the response usage counters to reconcile exactly with endpoint-native reused/prompt/turn counters, requires the `net-true` fixture verdict and a 64-hex base fingerprint, and rejects synthetic scheduler ledgers. It never emits a Track-2 provider-dollar row.

For `experiments/microcontext/s2b-gcp-inkernel-prefix-ab-pass-2026-08-07.json`, the status card renders:

```text
Track 1 realized reuse 0.866 (new)
Track 1 current: 2026-W32 86.6% reuse, 1 session(s), 8 multi-turn turn(s)
Track 2 current: no OBSERVED-$ rows
Fleet aggregate: rows 0, exit sessions 0, saved 4859 token-equiv
```

The Track-1 row identifies provider `fak-inkernel`, mechanism `microcontext_radix_prefix`, and session type `microcontext-shared-base`; its context key is the captured base SHA-256. The 4,859 reused tokens are endpoint-witnessed token-equivalents, not billed savings. Existing cache-value honesty text continues to exclude the vs-naive re-prefill multiple and keeps Track 1 and Track 2 side by side.

## Verify

```powershell
go test ./internal/cachevaluereport ./cmd/fak -run 'MicroContext|CachevalueFeed'
fak cachevalue feed --microcontext-ledger experiments/microcontext/s2b-gcp-inkernel-prefix-ab-pass-2026-08-07.json --dry-run --source agent
```

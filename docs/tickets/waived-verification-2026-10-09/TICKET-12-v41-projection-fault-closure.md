# TICKET-12: V4.1 compressor and indexer projection fault-closure contract regressed

## Current state

At fak `4cbddeafffb`, four tests share one failure family:

- `TestV41CompressorProjectionUnclassifiedFailureCloses` and `TestV41IndexerProjectionUnclassifiedFailureCloses` fail in every matmul and read sub-case (wkv, wgate, wq_b, wk, weights_proj) with "selected failure changed rollback or leaked transient" (`v41_compressor_projection_test.go:914`).
- `TestV41CompressorProjectionUnknownAndClosedPanicIdentity` and `TestV41IndexerProjectionUnknownAndClosedPanicIdentity` fail with "unknown panic changed Session closure priority" (`v41_compressor_projection_fault_test.go:150`, `v41_indexer_projection_fault_test.go:150`).

An injected projection failure no longer leaves session rollback and transient ownership unchanged. Likely sources are the waived commits that moved compressed reader ownership and transactions, such as 76c848b446f (compressed reader ownership and BF16 boundaries), 9ca0b5ccf59 (bounded lazy CPU retention transactions), and ac66aa968ad (transactional ratio-one composition). Bisect to confirm.

Parent: #13764.

## Working spine

1. Bisect one representative sub-case (`attn.compressor.wkv.weight/matmul`).
2. Restore the contract: an unclassified failure closes the session without changing rollback or leaking transients, and an unknown panic keeps its closure priority.
3. Fix it once in the shared helper path, so both compressor and indexer pass.

## Witness

`go test -count=1 -timeout 60m -run 'V41(Compressor|Indexer)Projection(UnclassifiedFailureCloses|UnknownAndClosedPanicIdentity)' ./internal/model/` exit 0.

## Done condition

- [ ] All four tests green on trunk, and the fix cites the culprit sha.

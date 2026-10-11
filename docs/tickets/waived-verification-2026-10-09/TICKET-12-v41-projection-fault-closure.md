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

- [x] All four tests green on trunk, and the fix cites the culprit sha.

## Resolution

Culprit: 82b4d3ce6cb. The code is correct and the tests were wrong, the same finding as #13795.

- 82b4d3ce6cb made the selected compressor/indexer boundary close the Session on any unclassified failure while re-panicking the original value ("cannot leave this selected owner live"). `UnknownAndClosedPanicIdentity` still pinned the earlier contract from 305768ee805/85e0ca21540, where an unknown panic left the Session open. It now requires closure, a `*BackendForwardOperationError` latch, and, when the Session was already closed, the first latch keeping priority.
- `v41GuardedProjectionPanicContract` was added by 82b4d3ce6cb but never ran. Its `len(b.live) != 0` check counted the seven model-owned immutable norm weights that the Session borrows. `Session.Close` leaves those staged by design (`hal.go`, `borrowedHALW`). The leak check now excludes only those borrowed residents. Rollback state was unchanged in every sub-case.

# TICKET-08: V4.1 compressed-window composition and session fallback red after waived landing

## Current state

Two tests fail at fak `0016a2a6a9f`:

- `TestV41CompressedWindowComposition` (`internal/model/v41_compressed_window_test.go:81`): "layer 0 position 0 missing combined sparse payload ... N:4 idx [0 -1 -1 -1]".
- `TestV41IncrementalSessionFallback/non-plain-role` (`internal/model/v41_incremental_session_test.go:272`): "session with a non-plain layer role reported eligible, want fallback".

Suspect waived commits: ba378e227f7 "add own windows to compressed attention contraction"; 81573100ff6 "enforce active V4.1 attention roles"; related ac66aa968ad, 76c848b446f.

Parent: #13764.

## Working spine

1. Bisect both reds.
2. Restore the combined sparse payload for own-window positions.
3. Make non-plain layer roles fall back from incremental eligibility.
4. Add fail-before/pass-after tests.

## Witness

`go test -count=1 -timeout 60m -run 'V41CompressedWindow|V41IncrementalSession' ./internal/model/` exit 0.

## Done condition

- [ ] Both tests are green on trunk, and the fix commit cites the covered shas.

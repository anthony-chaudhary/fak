# TICKET-10: absent V4.1 mHC mix refuses with a geometry error instead of a named missing tensor

## Current state

At fak `a512e07ef31`, `TestV41ForwardAbsentMHCMixStillRefuses` (`internal/model/v41_mhc_geometry_test.go:104`) fails. A model with no mHC mix tensor is still refused, but with "no admitted mHC geometry (require flattened=false)". The test wants the refusal to name the missing tensor. That makes the refusal less actionable: an operator sees a geometry problem, not a missing weight.

Likely source: a recent waived mHC or geometry commit by oss-maintainer-12 after `24033b4080f`. Candidates include 6a54495ecd4 (transposed mHC device projection); bisect to confirm.

The same mHC geometry family also has `TestV41MHCProjectionRawFullAndReduced/full-q2` red at `4cbddeafffb` (`v41_mhc_projection_test.go:426`: "layer two lost independent nonzero stream 1"). Fix both here.

Parent: #13764. Found by the TICKET-01 worker; it is outside TICKET-01's group.

## Working spine

1. Bisect the red.
2. Put the missing-tensor check before geometry admission, so an absent mix yields the named missing-tensor refusal.
3. Pin the refusal with a sentinel or `errors.Is` check, not by matching error text.

## Witness

`go test -count=1 -timeout 60m -run 'V41ForwardAbsentMHC|V41MHC' ./internal/model/` exit 0.

## Done condition

- [ ] An absent mHC mix yields the named missing-tensor refusal; the test is green on trunk, and the fix commit cites the culprit sha.

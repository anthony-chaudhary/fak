package vdso

// proofs_witness_test.go — deterministic witness tests CLOSING open math-proof
// obligations for internal/vdso. See fak/docs/proofs/00-METHOD.md.
//
// OPEN [integrity-epoch-advances-monotonically]:
//   The integrity (trust) epoch advances monotonically: a refutation (Revoke of a
//   non-empty witness) strictly increases TrustEpoch by exactly 1; an empty-witness
//   Revoke is a no-op (epoch unchanged); and across a sequence of N refutations the
//   epoch is strictly increasing and never decreases.
//   mechanism: revoke.go:91 (atomic.AddUint64(&v.trustEpoch,1)),
//              revoke.go:136 (TrustEpoch loads the clock),
//              vdso.go:85 (Revoke entry; empty-witness early return at revoke.go:86).
//
// The existing revoke_test.go asserts the +1 bump and the empty no-op as ISOLATED
// single steps; what is NOT yet asserted is the SEQUENCE invariant — that over a long
// run of refutations (including a re-revoke of an already-refuted witness, and empty
// no-ops interspersed) the epoch is the exact running count of non-empty Revokes, is
// strictly monotone across each real refutation, and never decreases. This file
// witnesses that sequence property with a fixed-seed deterministic driver.

import (
	"fmt"
	"math/rand"
	"testing"
)

// TestProof_IntegrityEpochMonotonicSequence is the sequence witness. Over N steps it
// drives a deterministic (fixed-seed) mix of:
//   - non-empty Revoke (a real refutation) — must bump TrustEpoch by EXACTLY 1,
//   - empty-witness Revoke — must be a strict no-op (epoch unchanged, Revocations
//     unchanged),
//   - re-Revoke of an already-refuted witness — must STILL bump (each refutation event
//     is its own integrity tick, revoke.go has no "already revoked" short-circuit on
//     the epoch path),
//
// and after each step asserts:
//   - epoch never decreased (monotone),
//   - epoch advanced iff the step was a non-empty Revoke (strictly, by 1),
//   - the final epoch equals the running count of non-empty Revokes,
//   - Revocations() (the integrity-bus event count) tracks the same count.
func TestProof_IntegrityEpochMonotonicSequence(t *testing.T) {
	const N = 2000
	v := New(64)

	rng := rand.New(rand.NewSource(0x5150)) // fixed seed => fully deterministic run

	if v.TrustEpoch() != 0 {
		t.Fatalf("fresh VDSO TrustEpoch=%d, want 0", v.TrustEpoch())
	}

	var refutations uint64 // running count of non-empty Revoke calls = expected epoch
	prev := v.TrustEpoch()

	// A small witness alphabet so re-revocation of an already-refuted witness happens
	// often — that path must still tick the epoch.
	witnesses := []string{"commit-a", "commit-b", "commit-c", "lease-7", "blob-deadbeef"}

	for i := 0; i < N; i++ {
		// ~1 in 5 steps is an empty-witness no-op; the rest are real refutations.
		empty := rng.Intn(5) == 0

		var witness string
		if !empty {
			witness = witnesses[rng.Intn(len(witnesses))]
		}

		beforeRevocations := v.Revocations()
		evicted := v.Revoke(witness)
		after := v.TrustEpoch()

		// (1) Monotonicity: the integrity clock NEVER decreases, on any step.
		if after < prev {
			t.Fatalf("step %d: TrustEpoch DECREASED %d -> %d (witness=%q)", i, prev, after, witness)
		}

		if empty {
			// (2) Empty-witness Revoke is a total no-op on the integrity axis.
			if evicted != 0 {
				t.Fatalf("step %d: empty Revoke evicted=%d, want 0", i, evicted)
			}
			if after != prev {
				t.Fatalf("step %d: empty Revoke moved TrustEpoch %d -> %d, want unchanged", i, prev, after)
			}
			if v.Revocations() != beforeRevocations {
				t.Fatalf("step %d: empty Revoke moved Revocations %d -> %d, want unchanged",
					i, beforeRevocations, v.Revocations())
			}
		} else {
			// (3) A real refutation STRICTLY increases the epoch by exactly 1.
			refutations++
			if after != prev+1 {
				t.Fatalf("step %d: non-empty Revoke(%q) moved TrustEpoch %d -> %d, want +1",
					i, witness, prev, after)
			}
			if v.Revocations() != beforeRevocations+1 {
				t.Fatalf("step %d: non-empty Revoke(%q) moved Revocations %d -> %d, want +1",
					i, witness, beforeRevocations, v.Revocations())
			}
			// The witness is now (still) marked refuted regardless of re-revocation.
			if !v.Revoked(witness) {
				t.Fatalf("step %d: witness %q not marked refuted after Revoke", i, witness)
			}
		}

		// (4) The epoch is exactly the running refutation count at every step.
		if after != refutations {
			t.Fatalf("step %d: TrustEpoch=%d != running refutation count %d", i, after, refutations)
		}
		prev = after
	}

	// Whole-run closure: final epoch == total non-empty Revokes == Revocations().
	if got := v.TrustEpoch(); got != refutations {
		t.Fatalf("final TrustEpoch=%d, want %d (total refutations)", got, refutations)
	}
	if got := v.Revocations(); uint64(got) != refutations {
		t.Fatalf("final Revocations()=%d, want %d", got, refutations)
	}
	if refutations == 0 {
		t.Fatal("non-vacuous guard: the deterministic run produced zero refutations")
	}
}

// TestProof_IntegrityEpochStrictlyIncreasingNonDecreasing isolates the two halves of
// the monotonicity claim on a clean back-to-back sequence (no randomness), so the
// witness is also readable as a direct invariant: K consecutive distinct refutations
// give epochs 1,2,...,K with each strictly greater than the last and none ever smaller.
func TestProof_IntegrityEpochStrictlyIncreasingNonDecreasing(t *testing.T) {
	const K = 256
	v := New(64)

	prev := v.TrustEpoch() // 0
	for k := 1; k <= K; k++ {
		w := fmt.Sprintf("witness-%d", k) // distinct each time
		v.Revoke(w)
		cur := v.TrustEpoch()
		if cur <= prev {
			t.Fatalf("refutation %d: TrustEpoch not strictly increasing (%d -> %d)", k, prev, cur)
		}
		if cur < prev {
			t.Fatalf("refutation %d: TrustEpoch decreased (%d -> %d)", k, prev, cur)
		}
		if cur != uint64(k) {
			t.Fatalf("after %d distinct refutations TrustEpoch=%d, want %d", k, cur, k)
		}
		prev = cur
	}
	if v.TrustEpoch() != uint64(K) {
		t.Fatalf("final TrustEpoch=%d, want %d", v.TrustEpoch(), K)
	}
}

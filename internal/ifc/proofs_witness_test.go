package ifc

// proofs_witness_test.go — deterministic witnesses for OPEN math-proof obligations
// against internal/ifc. See fak/docs/proofs/00-METHOD.md.
//
// OPEN (1) [taint-join-semilattice]: The abi.Ref.Taint join — implemented as a
// taintRank-max via Ledger.Raise over the closed 3-element order
// Trusted < Tainted < Quarantined — is a join-semilattice: it has an identity
// (an unseen trace reads Trusted), it is monotone (a Raise never lowers a mark),
// idempotent, commutative, and associative.
//
// mechanism: ifc.go:65 (taintRank), ifc.go:122 (Ledger.Raise), abi/types.go:82
// (TaintLabel closed lattice).
//
// Strategy: the Ledger is the operational realization of the join. We pin a PURE
// reference join (max by taintRank) and then prove, EXHAUSTIVELY over the closed
// 3-element carrier, that (a) Ledger.Raise realizes exactly that reference join,
// (b) the reference join obeys every semilattice law, and (c) Level/identity and
// monotonicity hold on the live Ledger. The carrier is finite (3 elements), so the
// exhaustive sweep over all elements / pairs / triples is a COMPLETE proof, not a
// sample.

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// carrier is the closed 3-element taint lattice, in ascending restrictiveness.
var carrier = []abi.TaintLabel{abi.TaintTrusted, abi.TaintTainted, abi.TaintQuarantined}

// refJoin is the SPECIFICATION join: the most-restrictive of a, b by taintRank.
// Ledger.Raise must realize exactly this when applied to a trace's current mark.
func refJoin(a, b abi.TaintLabel) abi.TaintLabel {
	if taintRank(a) >= taintRank(b) {
		return a
	}
	return b
}

// ledgerJoin computes a∨b through the LIVE Ledger: seed a trace at mark a (by
// raising from the Trusted identity), then Raise(b), and read the resulting mark.
// This is the operational join the IFC control plane actually uses.
func ledgerJoin(a, b abi.TaintLabel) abi.TaintLabel {
	l := NewLedger()
	const tr = "trace"
	// Seed to `a`. Identity is Trusted (rank 0), so a single Raise(a) sets it to a.
	l.Raise(tr, a)
	if got := l.Level(tr); got != a {
		// Seeding must be faithful for the rest of the proof to bind; surfaced by
		// the caller via a mismatch against refJoin if it ever isn't.
		_ = got
	}
	l.Raise(tr, b)
	return l.Level(tr)
}

// TestTaintJoinLedgerRealizesSpec — the LIVE Ledger.Raise realizes the reference
// join exactly, over every ordered pair of the closed carrier. This is the bridge
// lemma: it lets the algebraic laws proven on refJoin transfer to the real Ledger.
func TestTaintJoinLedgerRealizesSpec(t *testing.T) {
	for _, a := range carrier {
		for _, b := range carrier {
			want := refJoin(a, b)
			got := ledgerJoin(a, b)
			if got != want {
				t.Fatalf("Ledger.Raise(%s) over mark %s = %s; spec join = %s",
					taintName(b), taintName(a), taintName(got), taintName(want))
			}
		}
	}
}

// TestTaintJoinIdentity — an UNSEEN trace reads the join identity (Trusted), and
// joining anything with Trusted (the bottom/identity element) is a no-op: a∨⊥==a.
func TestTaintJoinIdentity(t *testing.T) {
	// Unseen trace is the identity element Trusted (ifc.go:147), NOT the enum zero
	// value TaintTainted (abi/types.go:86) — this distinction is load-bearing.
	if got := NewLedger().Level("never-raised"); got != abi.TaintTrusted {
		t.Fatalf("unseen trace = %s; want identity Trusted", taintName(got))
	}
	if abi.TaintTrusted == abi.TaintLabel(0) {
		t.Fatalf("guard: Trusted is the enum zero value — identity claim would be vacuous")
	}
	for _, a := range carrier {
		// a ∨ ⊥ == a on the live ledger (raising a fresh trace to a from Trusted).
		if got := ledgerJoin(abi.TaintTrusted, a); got != a {
			t.Fatalf("Trusted ∨ %s = %s; want %s", taintName(a), taintName(got), taintName(a))
		}
		if got := ledgerJoin(a, abi.TaintTrusted); got != a {
			t.Fatalf("%s ∨ Trusted = %s; want %s", taintName(a), taintName(got), taintName(a))
		}
	}
}

// TestTaintJoinMonotone — a Raise NEVER lowers a trace's mark: after Raise(b) the
// mark's rank is >= both the prior mark's rank and b's rank (it is an upper bound),
// over every reachable (prior, b) pair on the live Ledger.
func TestTaintJoinMonotone(t *testing.T) {
	for _, prior := range carrier {
		for _, b := range carrier {
			l := NewLedger()
			const tr = "m"
			l.Raise(tr, prior)
			before := l.Level(tr)
			l.Raise(tr, b)
			after := l.Level(tr)
			if taintRank(after) < taintRank(before) {
				t.Fatalf("Raise(%s) LOWERED mark %s -> %s (rank %d < %d)",
					taintName(b), taintName(before), taintName(after),
					taintRank(after), taintRank(before))
			}
			// Upper-bound: the join dominates BOTH operands.
			if taintRank(after) < taintRank(before) || taintRank(after) < taintRank(b) {
				t.Fatalf("Raise result %s is not an upper bound of {%s, %s}",
					taintName(after), taintName(before), taintName(b))
			}
		}
	}
}

// TestTaintJoinIdempotent — a∨a == a on the live Ledger, for every carrier element.
func TestTaintJoinIdempotent(t *testing.T) {
	for _, a := range carrier {
		if got := ledgerJoin(a, a); got != a {
			t.Fatalf("%s ∨ %s = %s; want %s (idempotent)",
				taintName(a), taintName(a), taintName(got), taintName(a))
		}
	}
}

// TestTaintJoinCommutative — Raise(a) then Raise(b) yields the SAME final mark as
// Raise(b) then Raise(a), over every ordered pair, through the live Ledger.
func TestTaintJoinCommutative(t *testing.T) {
	for _, a := range carrier {
		for _, b := range carrier {
			ab := ledgerJoin(a, b)
			ba := ledgerJoin(b, a)
			if ab != ba {
				t.Fatalf("commutativity FAILS: %s∨%s=%s but %s∨%s=%s",
					taintName(a), taintName(b), taintName(ab),
					taintName(b), taintName(a), taintName(ba))
			}
		}
	}
}

// ledgerJoin3 folds three Raises in the given association order through one live
// trace: left = (a∨b)∨c is just sequential Raise(a),Raise(b),Raise(c) (the ledger
// already accumulates a running join). To witness the OTHER association — a∨(b∨c)
// — we precompute (b∨c) on a side ledger and Raise it onto a.
func ledgerJoinLeft(a, b, c abi.TaintLabel) abi.TaintLabel {
	l := NewLedger()
	const tr = "t"
	l.Raise(tr, a)
	l.Raise(tr, b)
	l.Raise(tr, c)
	return l.Level(tr)
}

func ledgerJoinRight(a, b, c abi.TaintLabel) abi.TaintLabel {
	bc := ledgerJoin(b, c)
	return ledgerJoin(a, bc)
}

// TestTaintJoinAssociative — (a∨b)∨c == a∨(b∨c) over every ordered triple of the
// closed carrier, through the live Ledger. Exhaustive over 3^3 = 27 triples.
func TestTaintJoinAssociative(t *testing.T) {
	for _, a := range carrier {
		for _, b := range carrier {
			for _, c := range carrier {
				left := ledgerJoinLeft(a, b, c)
				right := ledgerJoinRight(a, b, c)
				if left != right {
					t.Fatalf("associativity FAILS: (%s∨%s)∨%s=%s but %s∨(%s∨%s)=%s",
						taintName(a), taintName(b), taintName(c), taintName(left),
						taintName(a), taintName(b), taintName(c), taintName(right))
				}
			}
		}
	}
}

package bench

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/negframe"
)

func TestDeMorganEquivalenceFullMechanicalLexicon(t *testing.T) {
	proofs := negframe.MechanicalEquivalenceProofs()
	if len(proofs) == 0 {
		t.Fatal("mechanical proof corpus is empty")
	}
	for _, proof := range proofs {
		if proof.Pattern == "" || proof.Replacement == "" {
			t.Fatalf("incomplete mechanical proof row: %+v", proof)
		}
	}
	if err := VerifyMechanicalEquivalence(proofs); err != nil {
		t.Fatal(err)
	}

	a, b := negframe.Atom("A"), negframe.Atom("B")
	for name, pair := range map[string][2]negframe.BoolExpr{
		"not-and": {negframe.Not(negframe.And(a, b)), negframe.Or(negframe.Not(a), negframe.Not(b))},
		"not-or":  {negframe.Not(negframe.Or(a, b)), negframe.And(negframe.Not(a), negframe.Not(b))},
	} {
		t.Run(name, func(t *testing.T) {
			if got := BooleanEquivalent(pair[0], pair[1]); !got.Equivalent || got.Rows != 4 {
				t.Fatalf("result=%+v", got)
			}
		})
	}
}

func TestDeMorganEquivalenceCatchesMeaningChange(t *testing.T) {
	bad := []negframe.MechanicalEquivalenceProof{{ID: "bad", Pattern: "test", Proven: true, Original: negframe.Not(negframe.Atom("A")), Rewritten: negframe.Atom("A")}}
	if err := VerifyMechanicalEquivalence(bad); err == nil || !strings.Contains(err.Error(), "changes meaning") {
		t.Fatalf("error=%v", err)
	}
}

func TestDeMorganEquivalenceRequiresProofForNewRule(t *testing.T) {
	missing := append(negframe.MechanicalEquivalenceProofs(), negframe.MechanicalEquivalenceProof{Pattern: "new mechanical rule"})
	if err := VerifyMechanicalEquivalence(missing); err == nil || !strings.Contains(err.Error(), "no equivalence proof") {
		t.Fatalf("error=%v", err)
	}
}

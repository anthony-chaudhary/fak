package bench

import (
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/negframe"
)

// EquivalenceResult is the complete truth-table witness for one constraint pair.
type EquivalenceResult struct {
	Equivalent     bool            `json:"equivalent"`
	Rows           int             `json:"rows"`
	Counterexample map[string]bool `json:"counterexample,omitempty"`
}

// BooleanEquivalent enumerates every atom assignment and compares both expressions.
// Invalid expressions fail closed as non-equivalent.
func BooleanEquivalent(a, b negframe.BoolExpr) EquivalenceResult {
	set := map[string]struct{}{}
	if !collectAtoms(a, set) || !collectAtoms(b, set) {
		return EquivalenceResult{}
	}
	atoms := make([]string, 0, len(set))
	for atom := range set {
		atoms = append(atoms, atom)
	}
	sort.Strings(atoms)
	if len(atoms) > 16 {
		return EquivalenceResult{}
	}
	rows := 1 << len(atoms)
	for mask := 0; mask < rows; mask++ {
		env := make(map[string]bool, len(atoms))
		for i, atom := range atoms {
			env[atom] = mask&(1<<i) != 0
		}
		av, aok := evalBool(a, env)
		bv, bok := evalBool(b, env)
		if !aok || !bok || av != bv {
			return EquivalenceResult{Rows: mask + 1, Counterexample: env}
		}
	}
	return EquivalenceResult{Equivalent: true, Rows: rows}
}

func collectAtoms(e negframe.BoolExpr, out map[string]struct{}) bool {
	switch e.Op {
	case negframe.BoolAtom:
		if e.Atom == "" || len(e.Args) != 0 {
			return false
		}
		out[e.Atom] = struct{}{}
		return true
	case negframe.BoolNot:
		return len(e.Args) == 1 && collectAtoms(e.Args[0], out)
	case negframe.BoolAnd, negframe.BoolOr:
		return len(e.Args) == 2 && collectAtoms(e.Args[0], out) && collectAtoms(e.Args[1], out)
	default:
		return false
	}
}

func evalBool(e negframe.BoolExpr, env map[string]bool) (bool, bool) {
	switch e.Op {
	case negframe.BoolAtom:
		v, ok := env[e.Atom]
		return v, ok
	case negframe.BoolNot:
		if len(e.Args) != 1 {
			return false, false
		}
		v, ok := evalBool(e.Args[0], env)
		return !v, ok
	case negframe.BoolAnd, negframe.BoolOr:
		if len(e.Args) != 2 {
			return false, false
		}
		a, aok := evalBool(e.Args[0], env)
		b, bok := evalBool(e.Args[1], env)
		if e.Op == negframe.BoolAnd {
			return a && b, aok && bok
		}
		return a || b, aok && bok
	default:
		return false, false
	}
}

// VerifyMechanicalEquivalence rejects both missing obligations and meaning-changing pairs.
func VerifyMechanicalEquivalence(proofs []negframe.MechanicalEquivalenceProof) error {
	if len(proofs) == 0 {
		return fmt.Errorf("no mechanical rules exported")
	}
	for i, proof := range proofs {
		if !proof.Proven || proof.ID == "" {
			return fmt.Errorf("mechanical rule %d (%s) has no equivalence proof", i, proof.Pattern)
		}
		if result := BooleanEquivalent(proof.Original, proof.Rewritten); !result.Equivalent {
			return fmt.Errorf("mechanical rule %d proof %q changes meaning: counterexample=%v", i, proof.ID, result.Counterexample)
		}
	}
	return nil
}

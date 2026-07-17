package negframe

// BoolOp is the small boolean-constraint vocabulary used to prove that a mechanical
// positive reframe preserves truth. It intentionally models constraints, not prose.
type BoolOp string

const (
	BoolAtom BoolOp = "atom"
	BoolNot  BoolOp = "not"
	BoolAnd  BoolOp = "and"
	BoolOr   BoolOp = "or"
)

// BoolExpr is a dependency-free boolean expression. Atom is set only for BoolAtom;
// Args has one child for BoolNot and two children for BoolAnd/BoolOr.
type BoolExpr struct {
	Op   BoolOp     `json:"op"`
	Atom string     `json:"atom,omitempty"`
	Args []BoolExpr `json:"args,omitempty"`
}

func Atom(name string) BoolExpr  { return BoolExpr{Op: BoolAtom, Atom: name} }
func Not(x BoolExpr) BoolExpr    { return BoolExpr{Op: BoolNot, Args: []BoolExpr{x}} }
func And(a, b BoolExpr) BoolExpr { return BoolExpr{Op: BoolAnd, Args: []BoolExpr{a, b}} }
func Or(a, b BoolExpr) BoolExpr  { return BoolExpr{Op: BoolOr, Args: []BoolExpr{a, b}} }

// MechanicalEquivalenceProof binds one exported mechanical lexicon rule to the
// original and rewritten constraints which a truth-table witness must compare.
type MechanicalEquivalenceProof struct {
	ID          string   `json:"id"`
	Pattern     string   `json:"pattern"`
	Replacement string   `json:"replacement"`
	Original    BoolExpr `json:"original"`
	Rewritten   BoolExpr `json:"rewritten"`
	Proven      bool     `json:"proven"`
}

var mechanicalProofs = map[string][2]BoolExpr{
	// "do not forget" and "do not hesitate" remove a lexical double negative.
	"double-negative": {Not(Not(Atom("action"))), Atom("action")},
	// "no need" and "can skip" both state that the action is not required.
	"optional-action": {Not(Atom("required")), Not(Atom("required"))},
	// "make sure you do not" and "avoid" preserve the prohibited action polarity.
	"prohibition": {Not(Atom("action")), Not(Atom("action"))},
}

// MechanicalEquivalenceProofs exports every mechanical rule, including an explicit
// Proven=false row when its proof ID is absent or unknown. Thus adding a lexicon rule
// without a proof obligation deterministically reds the bench coverage gate.
func MechanicalEquivalenceProofs() []MechanicalEquivalenceProof {
	out := make([]MechanicalEquivalenceProof, 0)
	for _, rule := range rules {
		if rule.Template == "" {
			continue
		}
		pair, ok := mechanicalProofs[rule.ProofID]
		row := MechanicalEquivalenceProof{ID: rule.ProofID, Pattern: rule.Pattern.String(), Replacement: rule.Template, Proven: ok && rule.ProofID != ""}
		if ok {
			row.Original, row.Rewritten = pair[0], pair[1]
		}
		out = append(out, row)
	}
	return out
}

package shipgate

import (
	"fmt"
	"os/exec"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// Decision represents the keep, revert, or escalate verdict for a candidate change.
type Decision uint8

// Decision constants define the possible outcomes of candidate evaluation.
const (
	REVERT Decision = iota
	KEEP
	ESCALATE
)

// String renders the decision as a stable token.
func (d Decision) String() string {
	switch d {
	case KEEP:
		return "KEEP"
	case REVERT:
		return "REVERT"
	case ESCALATE:
		return "ESCALATE"
	}
	return "?"
}

// EvidenceClass names the candidate class proven for a witness.
type EvidenceClass uint8

// EvidenceClass constants specify the verification tier proven for a witness.
const (
	ClassFull EvidenceClass = iota
	ClassDocsOnly
	ClassProofCarrying
)

// String renders the evidence class name.
func (c EvidenceClass) String() string {
	switch c {
	case ClassFull:
		return "ClassFull"
	case ClassDocsOnly:
		return "ClassDocsOnly"
	case ClassProofCarrying:
		return "ClassProofCarrying"
	}
	return "ClassFull"
}

// Profile specifies the required measured-signal subset for an EvidenceClass.
type Profile struct {
	needGain  bool
	needSuite bool
	needTruth bool
}

// EvidenceProfile maps each EvidenceClass to its required verification signals.
var EvidenceProfile = map[EvidenceClass]Profile{
	ClassFull:          {needGain: true, needSuite: true, needTruth: true},
	ClassDocsOnly:      {needTruth: true},
	ClassProofCarrying: {needGain: true, needTruth: true},
}

// ProfileFor returns the verification profile for class c, falling back to ClassFull.
func ProfileFor(c EvidenceClass) Profile {
	if p, ok := EvidenceProfile[c]; ok {
		return p
	}
	return EvidenceProfile[ClassFull]
}

// NeedsCostlyEvidence reports whether the profile requires a metric gain or green suite.
func (p Profile) NeedsCostlyEvidence() bool { return p.needGain || p.needSuite }

// Witness contains measured evaluation signals from external execution.
type Witness struct {
	Class       EvidenceClass
	Metric      string
	Before      float64
	After       float64
	LowerBetter bool
	SuiteGreen  bool
	TruthClean  bool
	improvedBit bool
}

func (w Witness) improved() bool {
	if w.LowerBetter {
		return w.After < w.Before
	}
	return w.After > w.Before
}

// Evaluate computes the keep-or-revert decision based on profile-required witness signals.
func Evaluate(w Witness) (Decision, Witness) {
	p := ProfileFor(w.Class)
	w.improvedBit = (!p.needGain || w.improved()) &&
		(!p.needSuite || w.SuiteGreen) &&
		(!p.needTruth || w.TruthClean)
	if w.improvedBit {
		return KEEP, w
	}
	return REVERT, w
}

// Kept reports whether the candidate achieved a keep verdict.
func (w Witness) Kept() bool { return w.improvedBit }

// ClassifyPaths selects an EvidenceClass based on touched file paths.
func ClassifyPaths(paths []string, isDoc func(string) bool) EvidenceClass {
	if len(paths) == 0 || isDoc == nil {
		return ClassFull
	}
	for _, p := range paths {
		if !isDoc(p) {
			return ClassFull
		}
	}
	return ClassDocsOnly
}

// Gate tracks consecutive non-keeps and trips a breaker after K attempts.
type Gate struct {
	K        int
	nonKeeps int
}

// NewGate constructs a breaker that escalates after k consecutive non-keeps.
func NewGate(k int) *Gate {
	if k <= 0 {
		k = 3
	}
	return &Gate{K: k}
}

// Record updates breaker state with a new decision and returns the effective verdict.
func (g *Gate) Record(d Decision) Decision {
	if d == KEEP {
		g.nonKeeps = 0
		return KEEP
	}
	g.nonKeeps++
	if g.nonKeeps >= g.K {
		return ESCALATE
	}
	return REVERT
}

// ConsecutiveNonKeeps returns the current non-keep count.
func (g *Gate) ConsecutiveNonKeeps() int { return g.nonKeeps }

// ApplyInWorktree creates a detached git worktree, runs apply, and returns any error.
func ApplyInWorktree(repo, dir string, apply func(worktree string) error) error {
	add := exec.Command("git", "-C", repo, "worktree", "add", "--detach", dir)
	windowgate.ConfigureBackgroundCommand(add)
	if out, err := add.CombinedOutput(); err != nil {
		return fmt.Errorf("worktree add: %v: %s", err, out)
	}
	if err := apply(dir); err != nil {
		_ = RemoveWorktree(repo, dir)
		return err
	}
	return nil
}

// RemoveWorktree deletes an isolated worktree.
func RemoveWorktree(repo, dir string) error {
	rm := exec.Command("git", "-C", repo, "worktree", "remove", "--force", dir)
	windowgate.ConfigureBackgroundCommand(rm)
	if out, err := rm.CombinedOutput(); err != nil {
		return fmt.Errorf("worktree remove: %v: %s", err, out)
	}
	return nil
}

// TuneCacheSize evaluates a proposed cache size change against baseline performance.
func TuneCacheSize(baselineKPI, candidateKPI float64, suiteGreen, truthClean bool) (Decision, Witness) {
	return Evaluate(Witness{
		Metric:      "vdso_hit_rate",
		Before:      baselineKPI,
		After:       candidateKPI,
		LowerBetter: false,
		SuiteGreen:  suiteGreen,
		TruthClean:  truthClean,
	})
}

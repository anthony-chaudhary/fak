package opttarget

import (
	"fmt"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/rsiloop"
)

// Direction declares which way a metric improves.
type Direction int

const (
	// HigherBetter: a larger metric wins (e.g. a cache hit rate).
	HigherBetter Direction = iota
	// LowerBetter: a smaller metric wins (e.g. a latency or a debt count).
	LowerBetter
)

// Site names the tunable a target rewrites: a module-relative file path and the
// anchored const identifier whose literal value is the knob. It is the
// declarative form of worktree.go's hard-wired (tunableRelPath, TunableConstName)
// pair — the stable, regex-friendly target the worktree Proposer edits in an
// isolated copy.
type Site struct {
	Path  string `json:"path"`  // module-relative, forward-slash (e.g. "internal/rsiloop/tunable.go")
	Const string `json:"const"` // the anchored const identifier (e.g. "DefaultCacheSize")
}

// GrammarKind tags the bounded candidate space a target sweeps. The space is
// closed BY THE SCHEMA — exactly as an auto-fuser constrains its schedule space —
// so a target can never propose an unbounded free-form edit through this seam.
type GrammarKind string

const (
	// GrammarIntSweep enumerates a fixed list of int values for Site.Const. This
	// is the cache-size demo's space and the Phase 0 grammar; float/enum/
	// log-clustered grammars are named follow-ons.
	GrammarIntSweep GrammarKind = "int-sweep"
)

// Grammar is the bounded candidate generator: pure data naming a kind and the
// values it enumerates.
type Grammar struct {
	Kind GrammarKind `json:"kind"`
	Ints []int       `json:"ints,omitempty"`
}

// Guards are the declared safety bits a target carries: the paths a kept
// candidate is ALLOWED to change (the truth-clean floor a measurer enforces) and
// an intentional-floor marker for a tunable whose current value is a deliberate
// floor rather than a defect to optimize away. Phase 0 carries them as declared
// metadata; the general worktree rewrite (Phase 0.1) and the scheduler (Phase 2)
// read them.
type Guards struct {
	ChangedPaths     []string `json:"changed_paths,omitempty"`
	IntentionalFloor bool     `json:"intentional_floor,omitempty"`
}

// OptTarget is a DECLARED optimization target — the one new noun of the fuser. It
// compiles (Compile) to a rsiloop.Harness, so a target authored in a few lines of
// data earns the full non-forgeable keep-bit for free.
type OptTarget struct {
	Name        string    `json:"name"`
	Metric      string    `json:"metric"`
	Direction   Direction `json:"direction"`
	BaselineRef string    `json:"baseline_ref"`
	Site        Site      `json:"site"`
	Grammar     Grammar   `json:"grammar"`
	// Measurer is the registry key of the probe binding a future loader resolves
	// (Phase 1). Compile itself takes the Measurer VALUE explicitly, so this name
	// is metadata until the registry lands.
	Measurer string `json:"measurer"`
	Guards   Guards `json:"guards"`
}

// Validate reports the first reason a target is not compilable, or nil. A
// malformed declaration is REFUSED at compile time, never silently lowered into a
// harness that measures nothing.
func (t OptTarget) Validate() error {
	if t.Name == "" {
		return fmt.Errorf("opttarget: empty Name")
	}
	if t.Metric == "" {
		return fmt.Errorf("opttarget %q: empty Metric", t.Name)
	}
	if t.BaselineRef == "" {
		return fmt.Errorf("opttarget %q: empty BaselineRef", t.Name)
	}
	if t.Measurer == "" {
		return fmt.Errorf("opttarget %q: empty Measurer", t.Name)
	}
	switch t.Grammar.Kind {
	case GrammarIntSweep:
		if len(t.Grammar.Ints) == 0 {
			return fmt.Errorf("opttarget %q: int-sweep grammar has no values", t.Name)
		}
		if t.Site.Const == "" {
			return fmt.Errorf("opttarget %q: int-sweep grammar needs a Site.Const to rewrite", t.Name)
		}
	default:
		return fmt.Errorf("opttarget %q: unknown grammar kind %q", t.Name, t.Grammar.Kind)
	}
	return nil
}

// candidates lowers the grammar into the ordered rsiloop.Candidate list. The
// label/payload form is IDENTICAL to the hand-wired worktree harness (Label
// "<Const>=<n>", Payload the int), so a compiled target's journal is byte-
// identical to the hand-coded one (the Phase 0 golden witness).
func (t OptTarget) candidates() []rsiloop.Candidate {
	switch t.Grammar.Kind {
	case GrammarIntSweep:
		cs := make([]rsiloop.Candidate, 0, len(t.Grammar.Ints))
		for _, n := range t.Grammar.Ints {
			cs = append(cs, rsiloop.Candidate{
				Label:   t.Site.Const + "=" + strconv.Itoa(n),
				Payload: n,
			})
		}
		return cs
	default:
		return nil
	}
}

// lowerBetter maps the declared Direction onto rsiloop's bool.
func (t OptTarget) lowerBetter() bool { return t.Direction == LowerBetter }

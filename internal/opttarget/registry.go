package opttarget

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/rsiloop"
)

// MeasurerFactory builds the impure measurement seam for a target, given the repo
// root the worktree forks from. It is the one part the registry cannot keep as pure
// data — constructing a Measurer wires the live worktree/probe/suite/truth impls.
type MeasurerFactory func(t OptTarget, repoRoot string) (Measurer, error)

// registry binds a declared OptTarget.Measurer key to the factory that builds its
// seam. It is a CLOSED set: an unknown key is REFUSED by Resolve, never silently
// lowered into a target that measures nothing.
var registry = map[string]MeasurerFactory{
	"worktree-int": newWorktreeIntMeasurer,
}

// Resolve looks up the factory for t.Measurer and builds the Measurer for t,
// forking worktrees from repoRoot. An unknown key is refused with an error naming
// both the offending key and the known keys — a missing binding fails closed.
func Resolve(t OptTarget, repoRoot string) (Measurer, error) {
	factory, ok := registry[t.Measurer]
	if !ok {
		return nil, fmt.Errorf("opttarget %q: unknown measurer %q (known: %s)", t.Name, t.Measurer, knownKeys())
	}
	return factory(t, repoRoot)
}

// KnownMeasurers returns the registry's keys, sorted — the inventory a CLI help or
// audit lists as the closed set of bindable measurers.
func KnownMeasurers() []string {
	keys := make([]string, 0, len(registry))
	for k := range registry {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// knownKeys is the sorted known-measurer keys joined for an error message.
func knownKeys() string { return strings.Join(KnownMeasurers(), ", ") }

// newWorktreeIntMeasurer binds the "worktree-int" key to the rsiloop worktree
// harness via HarnessMeasurer. Scope (Phase 0): the worktree harness rewrites the
// SINGLE tunable rsiloop.TunableRelPath:rsiloop.TunableConstName regardless of the
// target's declared Site, so this factory is faithful ONLY for a target whose Site
// IS that tunable — it guards that match and refuses any other Site rather than
// silently mis-measure. Lowering an ARBITRARY Site's (path, const) rewrite into the
// worktree seam is the named Phase 0.1 follow-on.
func newWorktreeIntMeasurer(t OptTarget, repoRoot string) (Measurer, error) {
	if t.Grammar.Kind != GrammarIntSweep {
		return nil, fmt.Errorf("opttarget %q: worktree-int needs an int-sweep grammar, got %q", t.Name, t.Grammar.Kind)
	}
	// HONESTY GUARD: the worktree harness hard-rewrites
	// rsiloop.TunableRelPath:rsiloop.TunableConstName regardless of t.Site, so a
	// target pointing at any other Site would SILENTLY mis-measure (the harness would
	// edit the demo tunable, not the declared one). Refuse it.
	if t.Site.Path != rsiloop.TunableRelPath || t.Site.Const != rsiloop.TunableConstName {
		return nil, fmt.Errorf(
			"opttarget %q: worktree-int only rewrites %s:%s, but Site is %s:%s; "+
				"generalizing to an arbitrary Site is the named Phase 0.1 follow-on",
			t.Name, rsiloop.TunableRelPath, rsiloop.TunableConstName, t.Site.Path, t.Site.Const)
	}
	h := rsiloop.NewWorktreeHarness(rsiloop.WorktreeConfig{
		Repo:        repoRoot,
		BaselineRef: t.BaselineRef,
		Candidates:  t.Grammar.Ints,
	})
	return HarnessMeasurer{H: h}, nil
}

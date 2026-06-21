// Package steward is the steward population: a set of cheap, single-invariant
// validators that garden the kernel's journal. The design rule (from the ABI):
// a steward NEVER blocks on its own opinion — it returns a violation only with an
// independently-authored witness, else abstains. A meta-steward prunes stewards
// that never fire across a soak, so the population doesn't ossify into dead code.
//
// Stewards here are self-contained (they take their evidence as data or observe
// the event journal), so the package depends only on abi — keeping it a clean
// leaf. Concrete builders cover the v0.1 invariants: secret-in-context,
// lease-disjointness, KPI-regression, and vDSO-soundness.
package steward

import (
	"context"
	"fmt"
	"regexp"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Check is a single invariant probe: it returns (violated, witness).
type Check func(ctx context.Context) (bool, string)

// FuncSteward adapts a named Check to abi.Steward.
type FuncSteward struct {
	name  string
	check Check
}

func NewSteward(name string, c Check) *FuncSteward              { return &FuncSteward{name, c} }
func (s *FuncSteward) Name() string                             { return s.name }
func (s *FuncSteward) Check(ctx context.Context) (bool, string) { return s.check(ctx) }

// Population runs a set of stewards and tracks how often each fires (for the
// meta-steward prune).
type Population struct {
	mu    sync.Mutex
	st    []abi.Steward
	fires map[string]int
}

func NewPopulation(stewards ...abi.Steward) *Population {
	return &Population{st: stewards, fires: map[string]int{}}
}

func (p *Population) Add(s abi.Steward) {
	p.mu.Lock()
	p.st = append(p.st, s)
	p.mu.Unlock()
}

// Sweep runs every steward once and tallies fires. Returns the names that fired
// with their witnesses.
func (p *Population) Sweep(ctx context.Context) map[string]string {
	p.mu.Lock()
	defer p.mu.Unlock()
	fired := map[string]string{}
	for _, s := range p.st {
		if v, w := s.Check(ctx); v {
			p.fires[s.Name()]++
			fired[s.Name()] = w
		}
	}
	return fired
}

// Prune is the meta-steward: after a soak it removes stewards that never fired
// (dead invariants), returning the pruned names (unit 92).
func (p *Population) Prune() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	var pruned []string
	kept := p.st[:0]
	for _, s := range p.st {
		if p.fires[s.Name()] == 0 {
			pruned = append(pruned, s.Name())
			continue
		}
		kept = append(kept, s)
	}
	p.st = kept
	return pruned
}

// Names lists the current steward names.
func (p *Population) Names() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]string, len(p.st))
	for i, s := range p.st {
		out[i] = s.Name()
	}
	return out
}

// ----------------------------------------------------------------------------
// Concrete invariant builders.
// ----------------------------------------------------------------------------

var secretRE = regexp.MustCompile(`(?i)(sk-[a-z0-9]{16,}|AKIA[0-9A-Z]{12,}|ghp_[A-Za-z0-9]{20,})`)

// SecretInContext fires if any byte slice in contextSnapshot() carries a secret
// shape — i.e. the MMU let one through. The witness is the offending snippet.
func SecretInContext(snapshot func() [][]byte) *FuncSteward {
	return NewSteward("no-secret-in-context", func(ctx context.Context) (bool, string) {
		for _, b := range snapshot() {
			if m := secretRE.Find(b); m != nil {
				return true, "secret-shaped bytes admitted to context: " + string(m[:min(8, len(m))]) + "…"
			}
		}
		return false, ""
	})
}

// Lease is a file-tree lease (lane + glob set).
type Lease struct {
	Lane  string
	Trees []string
}

// LeaseDisjointness fires if any two leases share a tree prefix (unit 90).
func LeaseDisjointness(leases func() []Lease) *FuncSteward {
	return NewSteward("lease-disjointness", func(ctx context.Context) (bool, string) {
		ls := leases()
		for i := 0; i < len(ls); i++ {
			for j := i + 1; j < len(ls); j++ {
				for _, a := range ls[i].Trees {
					for _, b := range ls[j].Trees {
						if overlap(a, b) {
							return true, fmt.Sprintf("%s and %s collide on %s/%s", ls[i].Lane, ls[j].Lane, a, b)
						}
					}
				}
			}
		}
		return false, ""
	})
}

// KPIRegression fires if the current p50 regressed past the committed baseline
// by more than tol (unit 91).
func KPIRegression(baseline, current func() float64, tol float64) *FuncSteward {
	return NewSteward("kpi-regression", func(ctx context.Context) (bool, string) {
		b, c := baseline(), current()
		if b > 0 && c > b*(1+tol) {
			return true, fmt.Sprintf("p50 regressed: baseline=%.3g current=%.3g", b, c)
		}
		return false, ""
	})
}

// VDSOSoundness fires if a probe shows a cache hit != a fresh pure call (unit 88).
func VDSOSoundness(probe func() (bool, string)) *FuncSteward {
	return NewSteward("vdso-soundness", func(ctx context.Context) (bool, string) {
		return probe()
	})
}

func overlap(a, b string) bool {
	return a == b || hasPrefix(a, b) || hasPrefix(b, a)
}

func hasPrefix(s, p string) bool {
	return len(s) >= len(p) && s[:len(p)] == p
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func init() {
	// v0.1 registers no stewards by default in the global registry; the bench /
	// run wiring constructs them with live snapshots and adds them to a Population.
	// (Registration via abi.RegisterSteward is available for always-on stewards.)
	abi.RegisterCapability("steward.v1")
}

package steward

import (
	"context"
	"sort"
	"testing"
)

// Unit 87: NewSteward + Population.Sweep reports a firing steward with its witness.
func TestNewStewardAndSweepReportsWitness(t *testing.T) {
	s := NewSteward("x", func(ctx context.Context) (bool, string) { return true, "w" })
	if s.Name() != "x" {
		t.Fatalf("Name() = %q, want %q", s.Name(), "x")
	}
	pop := NewPopulation(s)
	fired := pop.Sweep(context.Background())
	w, ok := fired["x"]
	if !ok {
		t.Fatalf("Sweep() did not report steward %q as fired: %v", "x", fired)
	}
	if w != "w" {
		t.Fatalf("Sweep() witness = %q, want %q", w, "w")
	}
}

// A steward whose check abstains (false) must not appear in the Sweep result.
func TestSweepAbstainingStewardNotReported(t *testing.T) {
	s := NewSteward("quiet", func(ctx context.Context) (bool, string) { return false, "" })
	pop := NewPopulation(s)
	fired := pop.Sweep(context.Background())
	if _, ok := fired["quiet"]; ok {
		t.Fatalf("abstaining steward was reported as fired: %v", fired)
	}
	if len(fired) != 0 {
		t.Fatalf("expected empty fired map, got %v", fired)
	}
}

// Unit 88: VDSOSoundness fires when probe reports a mismatch; abstains otherwise.
func TestVDSOSoundness(t *testing.T) {
	fireProbe := func() (bool, string) { return true, "mismatch" }
	s := VDSOSoundness(fireProbe)
	if s.Name() != "vdso-soundness" {
		t.Fatalf("Name() = %q, want %q", s.Name(), "vdso-soundness")
	}
	v, w := s.Check(context.Background())
	if !v {
		t.Fatalf("VDSOSoundness probe (true) did not fire")
	}
	if w != "mismatch" {
		t.Fatalf("witness = %q, want %q", w, "mismatch")
	}

	cleanProbe := func() (bool, string) { return false, "" }
	v2, w2 := VDSOSoundness(cleanProbe).Check(context.Background())
	if v2 {
		t.Fatalf("VDSOSoundness probe (false) fired unexpectedly, witness=%q", w2)
	}
}

// Unit 89: SecretInContext fires on a secret-shaped token; abstains on clean.
func TestSecretInContext(t *testing.T) {
	dirty := func() [][]byte {
		return [][]byte{[]byte("token sk-abcdef0123456789abcdef0123")}
	}
	s := SecretInContext(dirty)
	if s.Name() != "no-secret-in-context" {
		t.Fatalf("Name() = %q, want %q", s.Name(), "no-secret-in-context")
	}
	v, w := s.Check(context.Background())
	if !v {
		t.Fatalf("SecretInContext did not fire on a secret-shaped token")
	}
	if w == "" {
		t.Fatalf("SecretInContext fired but witness was empty")
	}

	clean := func() [][]byte {
		return [][]byte{[]byte("nothing sensitive here"), []byte("plain text")}
	}
	v2, _ := SecretInContext(clean).Check(context.Background())
	if v2 {
		t.Fatalf("SecretInContext fired on a clean snapshot")
	}
}

// Unit 90: LeaseDisjointness fires when two leases share a tree prefix; abstains
// when fully disjoint.
func TestLeaseDisjointness(t *testing.T) {
	overlapping := func() []Lease {
		return []Lease{
			{Lane: "a", Trees: []string{"internal/abi"}},
			{Lane: "b", Trees: []string{"internal/abi/types"}},
		}
	}
	s := LeaseDisjointness(overlapping)
	if s.Name() != "lease-disjointness" {
		t.Fatalf("Name() = %q, want %q", s.Name(), "lease-disjointness")
	}
	v, w := s.Check(context.Background())
	if !v {
		t.Fatalf("LeaseDisjointness did not fire on prefix-sharing leases")
	}
	if w == "" {
		t.Fatalf("LeaseDisjointness fired but witness was empty")
	}

	disjoint := func() []Lease {
		return []Lease{
			{Lane: "a", Trees: []string{"internal/abi"}},
			{Lane: "b", Trees: []string{"internal/steward"}},
		}
	}
	v2, _ := LeaseDisjointness(disjoint).Check(context.Background())
	if v2 {
		t.Fatalf("LeaseDisjointness fired on fully disjoint leases")
	}
}

// Unit 91: KPIRegression fires when current regresses past baseline*(1+tol);
// abstains when current == baseline.
func TestKPIRegression(t *testing.T) {
	baseline := func() float64 { return 1.0 }
	regressed := func() float64 { return 2.0 }
	s := KPIRegression(baseline, regressed, 0.1)
	if s.Name() != "kpi-regression" {
		t.Fatalf("Name() = %q, want %q", s.Name(), "kpi-regression")
	}
	v, w := s.Check(context.Background())
	if !v {
		t.Fatalf("KPIRegression did not fire when current=2.0 vs baseline=1.0 tol=0.1")
	}
	if w == "" {
		t.Fatalf("KPIRegression fired but witness was empty")
	}

	steady := func() float64 { return 1.0 }
	v2, _ := KPIRegression(baseline, steady, 0.1).Check(context.Background())
	if v2 {
		t.Fatalf("KPIRegression fired when current == baseline")
	}
}

// Unit 92: meta-steward prune removes exactly the steward that never fires.
func TestPrunePopulation(t *testing.T) {
	mkFire := func(name string) *FuncSteward {
		return NewSteward(name, func(ctx context.Context) (bool, string) { return true, "w-" + name })
	}
	dead := NewSteward("dead", func(ctx context.Context) (bool, string) { return false, "" })

	pop := NewPopulation(
		mkFire("alpha"),
		mkFire("beta"),
		mkFire("gamma"),
		mkFire("delta"),
		dead,
	)

	// Sweep twice to accumulate fire tallies.
	pop.Sweep(context.Background())
	pop.Sweep(context.Background())

	pruned := pop.Prune()
	if len(pruned) != 1 {
		t.Fatalf("Prune() returned %v, want exactly [dead]", pruned)
	}
	if pruned[0] != "dead" {
		t.Fatalf("Prune() returned %q, want %q", pruned[0], "dead")
	}

	names := pop.Names()
	for _, n := range names {
		if n == "dead" {
			t.Fatalf("Names() still contains pruned steward %q: %v", "dead", names)
		}
	}
	got := append([]string(nil), names...)
	sort.Strings(got)
	want := []string{"alpha", "beta", "delta", "gamma"}
	if len(got) != len(want) {
		t.Fatalf("Names() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Names() = %v, want %v", got, want)
		}
	}
}

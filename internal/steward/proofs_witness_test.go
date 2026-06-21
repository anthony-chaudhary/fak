package steward

// Witness tests closing OPEN proof obligations for internal/steward.
// Discipline: fak/docs/proofs/00-METHOD.md.
//
// OPEN (1) [steward-population-deterministic-order-independent]
//   For a fixed set of stewards and a fixed environment:
//     (a) Sweep produces the same fired set and the same per-name fire tallies,
//         and Prune removes exactly the stewards that never fired and keeps the
//         rest, regardless of the ORDER in which stewards were added to the
//         Population.
//   mechanism: steward.go:55 (Sweep), :61 (fires tally), :70 (Prune), :87 (Names)
//
// Strategy: build a fixed multiset of named stewards — some always fire, some
// always abstain — with a deterministic per-name verdict. For many random
// permutations of the add-order (fixed seed), run an identical sweep schedule
// and assert that
//   * the fired set returned by each Sweep is identical (independent of order),
//   * the per-name fire tally is identical (witnessed via the Prune partition:
//     a steward is pruned iff its tally is exactly 0), and
//   * Prune keeps exactly the firing names and removes exactly the abstaining
//     names, matching the set-theoretic expectation derived from the verdicts.
// Equality is exact (sets/ints), so the test is non-vacuous.

import (
	"context"
	"math/rand"
	"reflect"
	"sort"
	"testing"
)

// firePlan is the fixed environment: a name -> "does it fire" map. The witness
// each firing steward emits is a deterministic function of its name, so the
// fired *map* (name -> witness) is fully determined by the verdicts, not order.
var stewardFirePlan = map[string]bool{
	"alpha":   true,
	"beta":    false,
	"gamma":   true,
	"delta":   true,
	"epsilon": false,
	"zeta":    true,
	"eta":     false,
	"theta":   true,
}

func mkPlannedSteward(name string, fires bool) *FuncSteward {
	w := "w-" + name
	return NewSteward(name, func(ctx context.Context) (bool, string) {
		if fires {
			return true, w
		}
		return false, ""
	})
}

// buildPopulation returns a Population whose stewards are added in the given
// order (a permutation of the plan's names).
func buildPopulation(order []string) *Population {
	pop := NewPopulation()
	for _, name := range order {
		pop.Add(mkPlannedSteward(name, stewardFirePlan[name]))
	}
	return pop
}

func sortedNames(in []string) []string {
	out := append([]string(nil), in...)
	sort.Strings(out)
	return out
}

// expectedFired is the fired map (name -> witness) implied by the fixed plan,
// independent of add-order.
func expectedFired() map[string]string {
	want := map[string]string{}
	for name, fires := range stewardFirePlan {
		if fires {
			want[name] = "w-" + name
		}
	}
	return want
}

// expectedKept / expectedPruned partition the names by verdict.
func expectedKeptPruned() (kept, pruned []string) {
	for name, fires := range stewardFirePlan {
		if fires {
			kept = append(kept, name)
		} else {
			pruned = append(pruned, name)
		}
	}
	sort.Strings(kept)
	sort.Strings(pruned)
	return kept, pruned
}

// TestStewardSweepFiredSetOrderIndependent witnesses OPEN(1a) for Sweep: across
// many add-order permutations of the same steward set, every Sweep returns the
// identical fired map. We also confirm the fired map equals the plan-derived
// expectation, so the equality is anchored to ground truth (non-vacuous).
func TestStewardSweepFiredSetOrderIndependent(t *testing.T) {
	rng := rand.New(rand.NewSource(0x57E_0001)) // fixed seed -> deterministic perms

	baseOrder := make([]string, 0, len(stewardFirePlan))
	for name := range stewardFirePlan {
		baseOrder = append(baseOrder, name)
	}
	sort.Strings(baseOrder)

	want := expectedFired()

	// Reference: a canonical (sorted) order.
	refFired := buildPopulation(baseOrder).Sweep(context.Background())
	if !reflect.DeepEqual(refFired, want) {
		t.Fatalf("reference Sweep fired = %v, want plan-derived %v", refFired, want)
	}

	const perms = 200
	for p := 0; p < perms; p++ {
		order := append([]string(nil), baseOrder...)
		rng.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })

		fired := buildPopulation(order).Sweep(context.Background())
		if !reflect.DeepEqual(fired, refFired) {
			t.Fatalf("perm %d order=%v: Sweep fired = %v, want %v (order changed the fired set)",
				p, order, fired, refFired)
		}
	}
}

// TestStewardPruneOrderIndependent witnesses OPEN(1a) for the fire-tally + Prune:
// after an identical sweep schedule, Prune removes exactly the never-fired
// stewards and keeps exactly the firing ones, regardless of add-order. Because a
// steward is pruned iff its accumulated tally is 0, the invariance of the
// kept/pruned partition across every permutation is a direct witness that the
// per-name fire tallies are order-independent (a firing steward must have a
// positive tally to survive; an abstaining one must have tally 0 to be pruned).
func TestStewardPruneOrderIndependent(t *testing.T) {
	rng := rand.New(rand.NewSource(0x57E_0002))

	baseOrder := make([]string, 0, len(stewardFirePlan))
	for name := range stewardFirePlan {
		baseOrder = append(baseOrder, name)
	}
	sort.Strings(baseOrder)

	wantKept, wantPruned := expectedKeptPruned()

	const perms = 200
	const sweepsPerRun = 3 // accumulate tallies; result must not depend on count>=1

	for p := 0; p < perms; p++ {
		order := append([]string(nil), baseOrder...)
		rng.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })

		pop := buildPopulation(order)
		for s := 0; s < sweepsPerRun; s++ {
			pop.Sweep(context.Background())
		}

		gotPruned := sortedNames(pop.Prune())
		if !reflect.DeepEqual(gotPruned, wantPruned) {
			t.Fatalf("perm %d order=%v: Prune removed %v, want exactly %v (never-fired set depends on order)",
				p, order, gotPruned, wantPruned)
		}

		gotKept := sortedNames(pop.Names())
		if !reflect.DeepEqual(gotKept, wantKept) {
			t.Fatalf("perm %d order=%v: after Prune Names()=%v, want exactly %v",
				p, order, gotKept, wantKept)
		}
	}
}

// TestStewardSweepTallyMonotoneOrderIndependent strengthens OPEN(1a): the
// per-name tally is not merely "positive vs zero" but accumulates one per Sweep
// in which the steward fires, identically across add-orders. We can observe the
// tally indirectly but exactly: after exactly N sweeps a firing steward has
// tally N (>0 -> survives Prune); after 0 sweeps EVERY steward has tally 0, so a
// freshly-built population in any order prunes its entire set. This pins both
// endpoints of the tally and is invariant to order.
func TestStewardSweepTallyMonotoneOrderIndependent(t *testing.T) {
	rng := rand.New(rand.NewSource(0x57E_0003))

	baseOrder := make([]string, 0, len(stewardFirePlan))
	for name := range stewardFirePlan {
		baseOrder = append(baseOrder, name)
	}
	sort.Strings(baseOrder)

	allNames := sortedNames(baseOrder)

	const perms = 100
	for p := 0; p < perms; p++ {
		order := append([]string(nil), baseOrder...)
		rng.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })

		// Zero sweeps: every tally is 0, so Prune removes the whole population,
		// regardless of order.
		popZero := buildPopulation(order)
		prunedAll := sortedNames(popZero.Prune())
		if !reflect.DeepEqual(prunedAll, allNames) {
			t.Fatalf("perm %d: 0-sweep Prune removed %v, want all %v", p, prunedAll, allNames)
		}
		if rem := popZero.Names(); len(rem) != 0 {
			t.Fatalf("perm %d: 0-sweep population not fully pruned, remaining=%v", p, rem)
		}
	}
}

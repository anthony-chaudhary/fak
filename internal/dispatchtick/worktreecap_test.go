package dispatchtick

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/scoreboard"
)

// worktreeCapInput is a minimal SPAWN-clean preflight: no host resources (so the #1337
// host cap does not bind) and no seat pool (so seat inventory does not bind), leaving the
// effective cap as exactly min(MaxWorkers, leaseTarget) -- today's ceiling. Each test then
// adds only the one term it is about, so a raise can never be confused with a side effect.
func worktreeCapInput(maxWorkers, leaseTarget int) PreflightInput {
	return PreflightInput{
		Workspace:  "/w",
		MaxWorkers: maxWorkers,
		Host:       HostCheck{Safe: true},
		Account:    AccountCheck{Available: true, Tag: "acct"},
		Kernel:     KernelCheck{Alive: IntPtr(0), Target: IntPtr(leaseTarget)},
	}
}

// provenEvidence is a #3180 A/B result that DID earn the raise: isolation on, the
// poison-free verdict, a zero incident count, and a measured peak of n workers.
func provenEvidence(n int) WorktreeIsolation {
	return WorktreeIsolation{
		Enabled:           true,
		Verdict:           WorktreeABProvenVerdict,
		PoisonIncidents:   0,
		ProvenConcurrency: n,
	}
}

// TestWorktreeCapUnearnedRaisesNothing is the core #3185 safety property: every way the
// evidence can fall short leaves the cap at today's ceiling. Isolation being ON is
// necessary but NOT sufficient -- raising on the flag alone is exactly #1334's
// raise-on-belief error, so each near-miss below must still resolve to the old ceiling.
func TestWorktreeCapUnearnedRaisesNothing(t *testing.T) {
	const maxWorkers, leaseTarget = 20, 6
	cases := []struct {
		name string
		iso  WorktreeIsolation
	}{
		{"zero value (isolation off, no evidence)", WorktreeIsolation{}},
		{"isolation off but evidence proven", WorktreeIsolation{Enabled: false, Verdict: WorktreeABProvenVerdict, ProvenConcurrency: 12}},
		{"isolation on but verdict NOT_PROVEN", WorktreeIsolation{Enabled: true, Verdict: "NOT_PROVEN", ProvenConcurrency: 12}},
		{"isolation on but no report at all", WorktreeIsolation{Enabled: true, ProvenConcurrency: 12}},
		{"isolation on, verdict proven, but poison observed", WorktreeIsolation{Enabled: true, Verdict: WorktreeABProvenVerdict, PoisonIncidents: 1, ProvenConcurrency: 12}},
		{"isolation on, verdict proven, but nothing measured", WorktreeIsolation{Enabled: true, Verdict: WorktreeABProvenVerdict, ProvenConcurrency: 0}},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			if earned := WorktreeProvenCap(tc.iso); earned != 0 {
				t.Fatalf("unearned evidence must earn 0, got %d", earned)
			}
			in := worktreeCapInput(maxWorkers, leaseTarget)
			in.WorktreeIsolation = tc.iso
			got := EvaluatePreflight(in)
			if got.Cap != leaseTarget {
				t.Fatalf("cap = %d, want today's ceiling %d", got.Cap, leaseTarget)
			}
			if got.CapTerms.WorktreeCap != 0 {
				t.Fatalf("cap_terms.worktree_cap = %d, want 0 (no raise applied)", got.CapTerms.WorktreeCap)
			}
			if got.CapTerms.Limiting == "worktree" {
				t.Fatal(`limiting = "worktree" without earned evidence`)
			}
		})
	}
}

// TestWorktreeCapProvenRaisesPastOldCeiling is the payoff: with isolation on and the A/B
// harness witnessing zero poison at 12, the fleet may run 12 -- past the old ~6 that the
// reactive lease target pins it to. The raise is the MEASURED number, not a constant.
func TestWorktreeCapProvenRaisesPastOldCeiling(t *testing.T) {
	const maxWorkers, leaseTarget, proven = 20, 6, 12
	in := worktreeCapInput(maxWorkers, leaseTarget)
	in.WorktreeIsolation = provenEvidence(proven)

	got := EvaluatePreflight(in)
	if got.Cap != proven {
		t.Fatalf("cap = %d, want the proven concurrency %d", got.Cap, proven)
	}
	if got.Cap <= leaseTarget {
		t.Fatalf("cap %d did not rise past the old ceiling %d", got.Cap, leaseTarget)
	}
	if got.CapTerms.WorktreeCap != proven {
		t.Fatalf("cap_terms.worktree_cap = %d, want %d", got.CapTerms.WorktreeCap, proven)
	}
	if got.CapTerms.Limiting != "worktree" {
		t.Fatalf("limiting = %q, want %q so an operator sees the evidence set the cap", got.CapTerms.Limiting, "worktree")
	}
}

// TestWorktreeCapDerivesFromEvidenceNotAConstant pins the "never a naked constant bump"
// requirement: the earned cap tracks whatever the harness measured, so a different
// measurement yields a different cap rather than a hard-coded number.
func TestWorktreeCapDerivesFromEvidenceNotAConstant(t *testing.T) {
	for _, proven := range []int{7, 9, 16, 32} {
		in := worktreeCapInput(64, 6)
		in.WorktreeIsolation = provenEvidence(proven)
		if got := EvaluatePreflight(in); got.Cap != proven {
			t.Fatalf("evidence at %d workers: cap = %d, want %d", proven, got.Cap, proven)
		}
	}
}

// TestWorktreeCapBoundedByHardCeilings proves the raise can lift the ARTIFICIAL
// build-poison ceiling but never the physical ones: host capacity (#1337, explicitly out
// of scope for #3185 and still the outer bound), seat inventory, and the operator's own
// configured max. Proven evidence must never overbook the box or an explicit config.
func TestWorktreeCapBoundedByHardCeilings(t *testing.T) {
	t.Run("host capacity bounds the raise", func(t *testing.T) {
		// 16 cores at the default 2-cores-per-worker budget => host cap 8, under the
		// proven 12. The physical box wins.
		in := worktreeCapInput(20, 6)
		in.WorktreeIsolation = provenEvidence(12)
		in.Resources = HostResources{Cores: IntPtr(16)}
		if got := EvaluatePreflight(in); got.Cap != 8 {
			t.Fatalf("cap = %d, want host-bounded 8", got.Cap)
		}
	})
	t.Run("seat inventory bounds the raise", func(t *testing.T) {
		in := worktreeCapInput(20, 6)
		in.WorktreeIsolation = provenEvidence(12)
		in.Seat = SeatCheck{Total: IntPtr(7)}
		if got := EvaluatePreflight(in); got.Cap != 7 {
			t.Fatalf("cap = %d, want seat-bounded 7", got.Cap)
		}
	})
	t.Run("configured max bounds the raise", func(t *testing.T) {
		// An operator who deliberately set --max-workers 8 must not be overshot to 12
		// just because isolation proved out; the raise lifts the lease target, not the
		// explicit ceiling (matching the #3368 forecast-floor convention).
		in := worktreeCapInput(8, 6)
		in.WorktreeIsolation = provenEvidence(12)
		if got := EvaluatePreflight(in); got.Cap != 8 {
			t.Fatalf("cap = %d, want config-bounded 8", got.Cap)
		}
	})
	t.Run("pending contraction still bounds the raise", func(t *testing.T) {
		// A drain in flight reclaims capacity; proven isolation must not admit onto it.
		in := worktreeCapInput(20, 6)
		in.WorktreeIsolation = provenEvidence(12)
		in.ContractionTarget = 5
		if got := EvaluatePreflight(in); got.Cap != 5 {
			t.Fatalf("cap = %d, want contraction-bounded 5", got.Cap)
		}
	})
}

// TestWorktreeCapZeroValueIsByteIdentical guards the fail-safe default: a caller that
// never sets the term gets exactly the pre-#3185 result, field for field.
func TestWorktreeCapZeroValueIsByteIdentical(t *testing.T) {
	base := worktreeCapInput(20, 6)
	base.Resources = HostResources{Cores: IntPtr(16)}
	base.Seat = SeatCheck{Total: IntPtr(9)}

	withTerm := base
	withTerm.WorktreeIsolation = WorktreeIsolation{}

	before, after := EvaluatePreflight(base), EvaluatePreflight(withTerm)
	if before.Cap != after.Cap || before.Verdict != after.Verdict {
		t.Fatalf("zero-value term changed the fold: cap %d->%d verdict %q->%q",
			before.Cap, after.Cap, before.Verdict, after.Verdict)
	}
	if after.CapTerms.Limiting != before.CapTerms.Limiting {
		t.Fatalf("zero-value term changed limiting: %q -> %q", before.CapTerms.Limiting, after.CapTerms.Limiting)
	}
}

// TestWorktreeCapVerdictMatchesScoreboard pins this package's mirrored verdict string to
// what the #3180 A/B fold actually emits. The preflight fold deliberately does not import
// scoreboard (it stays dependency-free), so without this test the two spellings could
// drift and the cap would silently stop ever being earned -- a failure that would look
// exactly like "the evidence never proved out".
func TestWorktreeCapVerdictMatchesScoreboard(t *testing.T) {
	poisonFree := scoreboard.FoldWorktreeAB(
		scoreboard.WorktreeABArm{Resolved: 5, DurationSeconds: 600, PoisonIncidents: 0, PeakConcurrency: 6, WaveID: "w1"},
		scoreboard.WorktreeABArm{Resolved: 5, DurationSeconds: 400, PoisonIncidents: 0, PeakConcurrency: 12, WaveID: "w1"},
	)
	if poisonFree.Verdict != WorktreeABProvenVerdict {
		t.Fatalf("scoreboard poison-free verdict %q != mirrored %q", poisonFree.Verdict, WorktreeABProvenVerdict)
	}

	// And the harness's own report feeds the gate end to end: a poison-free isolated arm
	// earns exactly its measured peak concurrency.
	iso := WorktreeIsolation{
		Enabled:           true,
		Verdict:           poisonFree.Verdict,
		PoisonIncidents:   poisonFree.Isolated.PoisonIncidents,
		ProvenConcurrency: poisonFree.Isolated.PeakConcurrency,
	}
	if earned := WorktreeProvenCap(iso); earned != 12 {
		t.Fatalf("A/B report earned %d, want the isolated arm's peak 12", earned)
	}

	// A poisoned arm must NOT earn a raise through the same path.
	poisoned := scoreboard.FoldWorktreeAB(
		scoreboard.WorktreeABArm{Resolved: 5, DurationSeconds: 600, PoisonIncidents: 0, PeakConcurrency: 6, WaveID: "w1"},
		scoreboard.WorktreeABArm{Resolved: 5, DurationSeconds: 400, PoisonIncidents: 3, PeakConcurrency: 12, WaveID: "w1"},
	)
	poisonedIso := WorktreeIsolation{
		Enabled:           true,
		Verdict:           poisoned.Verdict,
		PoisonIncidents:   poisoned.Isolated.PoisonIncidents,
		ProvenConcurrency: poisoned.Isolated.PeakConcurrency,
	}
	if earned := WorktreeProvenCap(poisonedIso); earned != 0 {
		t.Fatalf("a poisoned A/B arm earned %d, want 0", earned)
	}
}

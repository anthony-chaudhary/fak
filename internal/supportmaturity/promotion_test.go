package supportmaturity

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

// TestWitnessForEveryRung asserts the binding is TOTAL (every closed rung binds to a
// witness) and INJECTIVE (no two rungs share a proof) — the "bind each rung to the
// witness that proves it" deliverable of #1245. The pinned map is the doctrine: if a
// future edit re-points a rung's witness, this fails until the map is updated.
func TestWitnessForEveryRung(t *testing.T) {
	want := map[Rung]WitnessKind{
		M0None:       WitnessNone,
		M1Fenced:     WitnessFence,
		M2Loads:      WitnessPreflight,
		M3Runs:       WitnessProofPath,
		M4Correct:    WitnessOracleInCI,
		M5Optimized:  WitnessBenchCommitted,
		M6Parity:     WitnessSotaParity,
		M7BeyondSOTA: WitnessSotaBeyond,
	}
	seen := map[WitnessKind]Rung{}
	for _, r := range Rungs {
		got := WitnessFor(r)
		exp, ok := want[r]
		if !ok {
			t.Fatalf("rung %s has no expected witness pinned — extend want", r)
		}
		if got != exp {
			t.Fatalf("WitnessFor(%s) = %s, want %s", r, got, exp)
		}
		if prev, dup := seen[got]; dup {
			t.Fatalf("non-injective binding: %s and %s both proven by %s", prev, r, got)
		}
		seen[got] = r
	}
	if len(seen) != len(Rungs) {
		t.Fatalf("bound %d distinct witnesses for %d rungs", len(seen), len(Rungs))
	}
	// An out-of-range rung floors to WitnessNone — the closed-vocabulary default.
	if got := WitnessFor(Rung(len(Rungs))); got != WitnessNone {
		t.Fatalf("out-of-range rung witness = %s, want %s", got, WitnessNone)
	}
}

// nonAuthorBenchWin is a committed-bench witness CONFIRMED by signals the cell's author
// did not write: a strict speedup (Before<After) AND the truth syscall clean. It is the
// kind of evidence that legitimately promotes M4Correct → M5Optimized.
func nonAuthorBenchWin() shipgate.Witness {
	return shipgate.Witness{
		Class:       shipgate.ClassProofCarrying, // a bench is gain + truth, not suite
		Metric:      "tokens_per_sec",
		Before:      100.0,
		After:       150.0,
		LowerBetter: false, // a higher throughput is better
		TruthClean:  true,  // the non-author truth syscall confirmed it
	}
}

// authorOnlyBenchWin is the SAME strict speedup, but carried by author-only evidence:
// the truth syscall is NOT clean (no non-author confirmation). shipgate must refuse it.
func authorOnlyBenchWin() shipgate.Witness {
	w := nonAuthorBenchWin()
	w.TruthClean = false // the only difference: nobody but the author vouches for it
	return w
}

// TestPromoteRefusesAuthorOnlyBenchWin is the epic's golden NEGATIVE cell: a planted
// bench-win carried by author-only evidence does NOT promote (shipgate refuses). The
// strict speedup is real, but with no non-author witness the cell holds at M4.
func TestPromoteRefusesAuthorOnlyBenchWin(t *testing.T) {
	got, dec := Promote(M4Correct, WitnessBenchCommitted, authorOnlyBenchWin(), nil)
	if got != M4Correct {
		t.Fatalf("author-only bench-win promoted %s → %s; must hold (shipgate refuses)", M4Correct, got)
	}
	if dec != shipgate.REVERT {
		t.Fatalf("author-only bench-win decision = %s, want REVERT", dec)
	}
}

// TestPromoteAcceptsNonAuthorBenchWin is the matched POSITIVE cell: the same speedup,
// now confirmed by a non-author witness, advances exactly one rung M4 → M5 with KEEP.
// Without this, an always-refuse Promote would pass the negative test vacuously.
func TestPromoteAcceptsNonAuthorBenchWin(t *testing.T) {
	got, dec := Promote(M4Correct, WitnessBenchCommitted, nonAuthorBenchWin(), nil)
	if got != M5Optimized {
		t.Fatalf("non-author bench-win promoted %s → %s, want %s", M4Correct, got, M5Optimized)
	}
	if dec != shipgate.KEEP {
		t.Fatalf("non-author bench-win decision = %s, want KEEP", dec)
	}
}

func TestPromoteWithRecordCarriesScore(t *testing.T) {
	rec := PromoteWithRecord(M4Correct, WitnessBenchCommitted, nonAuthorBenchWin(), nil)
	if rec.Next != M5Optimized || rec.Decision != shipgate.KEEP || !rec.Kept {
		t.Fatalf("non-author bench-win record = %+v, want M5/KEEP/kept", rec)
	}
	if rec.Score.Name != "support_maturity_promotion" || rec.Score.Grade != "promoted" {
		t.Fatalf("promotion score = %+v, want support_maturity_promotion/promoted", rec.Score)
	}
	if got := promotionScoreComponent(rec.Score, "binding_ok"); got != 1 {
		t.Fatalf("binding_ok score = %.0f, want 1 in %+v", got, rec.Score)
	}
	if got := promotionScoreComponent(rec.Score, "metric_delta"); got != 50 {
		t.Fatalf("metric_delta score = %.0f, want 50 in %+v", got, rec.Score)
	}
	if got := promotionScoreComponent(rec.Score, "advanced"); got != 1 {
		t.Fatalf("advanced score = %.0f, want 1 in %+v", got, rec.Score)
	}

	refused := PromoteWithRecord(M4Correct, WitnessPreflight, nonAuthorBenchWin(), nil)
	if refused.Next != M4Correct || refused.Decision != shipgate.REVERT {
		t.Fatalf("wrong witness record = %+v, want hold/REVERT", refused)
	}
	if refused.Score.Grade != "wrong-witness" {
		t.Fatalf("wrong-witness grade = %q in %+v", refused.Score.Grade, refused.Score)
	}
	if got := promotionScoreComponent(refused.Score, "binding_ok"); got != 0 {
		t.Fatalf("wrong-witness binding_ok score = %.0f, want 0 in %+v", got, refused.Score)
	}
}

// TestPromoteRefusesWrongWitness asserts the BINDING gate: even a flawless non-author
// witness of the WRONG kind for the target rung cannot promote. To reach M5 you need a
// committed bench; a preflight verdict — however clean — is refused before shipgate runs.
func TestPromoteRefusesWrongWitness(t *testing.T) {
	got, dec := Promote(M4Correct, WitnessPreflight, nonAuthorBenchWin(), nil)
	if got != M4Correct {
		t.Fatalf("wrong-witness promotion advanced %s → %s; the binding must refuse it", M4Correct, got)
	}
	if dec != shipgate.REVERT {
		t.Fatalf("wrong-witness decision = %s, want REVERT", dec)
	}
}

// TestPromoteCapsAtTop asserts M7BeyondSOTA has no higher rung to climb to: any
// promotion attempt there — even a perfect witness — is REVERTed and the rung holds.
func TestPromoteCapsAtTop(t *testing.T) {
	got, dec := Promote(M7BeyondSOTA, WitnessSotaBeyond, nonAuthorBenchWin(), nil)
	if got != M7BeyondSOTA {
		t.Fatalf("promotion above the ladder top advanced %s → %s", M7BeyondSOTA, got)
	}
	if dec != shipgate.REVERT {
		t.Fatalf("top-rung decision = %s, want REVERT", dec)
	}
}

// TestPromoteEscalates wires shipgate's breaker into the promotion path: a run of
// refused promotions trips the consecutive-non-keep gate and ESCALATEs to a human — the
// third arm of the reused KEEP/REVERT/ESCALATE vocabulary. The rung never advances on a
// non-keep.
func TestPromoteEscalates(t *testing.T) {
	breaker := shipgate.NewGate(2) // escalate on the 2nd consecutive non-keep
	if _, dec := Promote(M4Correct, WitnessBenchCommitted, authorOnlyBenchWin(), breaker); dec != shipgate.REVERT {
		t.Fatalf("first refused promotion decision = %s, want REVERT", dec)
	}
	got, dec := Promote(M4Correct, WitnessBenchCommitted, authorOnlyBenchWin(), breaker)
	if dec != shipgate.ESCALATE {
		t.Fatalf("second refused promotion decision = %s, want ESCALATE", dec)
	}
	if got != M4Correct {
		t.Fatalf("escalating promotion advanced %s → %s; must hold", M4Correct, got)
	}
}

// TestPromoteBreakerResetsOnKeep asserts a genuine KEEP resets the breaker — a real
// promotion is not penalized by earlier refusals.
func TestPromoteBreakerResetsOnKeep(t *testing.T) {
	breaker := shipgate.NewGate(2)
	Promote(M4Correct, WitnessBenchCommitted, authorOnlyBenchWin(), breaker) // one non-keep
	if _, dec := Promote(M4Correct, WitnessBenchCommitted, nonAuthorBenchWin(), breaker); dec != shipgate.KEEP {
		t.Fatalf("non-author bench-win decision = %s, want KEEP", dec)
	}
	if n := breaker.ConsecutiveNonKeeps(); n != 0 {
		t.Fatalf("breaker not reset by KEEP: %d consecutive non-keeps remain", n)
	}
}

func promotionScoreComponent(score Scorecard, name string) float64 {
	for _, c := range score.Components {
		if c.Name == name {
			return c.Value
		}
	}
	return 0
}

// TestDropOnOracleRed is the epic's golden POSITIVE drop cell: a planted oracle-red
// drops a cell's rung. The M4 witness (the CI oracle) regresses, so M4Correct → M3Runs —
// the cell still runs on the reference but can no longer claim CI-witnessed correctness.
func TestDropOnOracleRed(t *testing.T) {
	if got := Drop(M4Correct, true); got != M3Runs {
		t.Fatalf("red oracle dropped %s → %s, want %s", M4Correct, got, M3Runs)
	}
	if got := Drop(M4Correct, false); got != M4Correct {
		t.Fatalf("clean oracle moved the rung: %s → %s; a clean witness must hold", M4Correct, got)
	}
}

// TestDropSweep asserts the drop rule over the whole ladder: a regressed bound witness
// demotes EXACTLY one rung from every rung above the floor, a clean witness holds every
// rung, and M0None is the floor (a regression cannot fall below it).
func TestDropSweep(t *testing.T) {
	for i, r := range Rungs {
		if got := Drop(r, false); got != r {
			t.Fatalf("clean witness moved %s → %s", r, got)
		}
		dropped := Drop(r, true)
		if r == M0None {
			if dropped != M0None {
				t.Fatalf("regression fell below the floor: %s → %s", r, dropped)
			}
			continue
		}
		if want := Rungs[i-1]; dropped != want {
			t.Fatalf("regressed %s dropped to %s, want exactly one rung down (%s)", r, dropped, want)
		}
	}
}

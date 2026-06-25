package callavoid

import (
	"encoding/json"
	"math"
	"reflect"
	"testing"
)

const eps = 1e-9

func approx(a, b float64) bool { return math.Abs(a-b) <= 1e-6 }

// TestProveMemoSingleUseIsALoss pins the headline of the skeptical gate, the dual of
// vcachechain.ProveRecall's "a single-unit recall is a loss": one access can never
// amortize the capture cost, so it refutes and the loss equals exactly that capture.
func TestProveMemoSingleUseIsALoss(t *testing.T) {
	p := ProveMemo(MemoInput{Accesses: 1, ValidateCost: 0.02, MutationRate: 0.1, CaptureCost: 0.02})
	if p.Status != ProofRefuted {
		t.Fatalf("single use must REFUTE, got %s (%s)", p.Status, p.Reason)
	}
	if p.Decision != "always_execute" {
		t.Errorf("decision = %q, want always_execute", p.Decision)
	}
	if !approx(p.SingleUseLoss, 0.02) {
		t.Errorf("SingleUseLoss = %v, want 0.02 (the capture cost)", p.SingleUseLoss)
	}
	if !approx(p.SavedCost, -0.02) {
		t.Errorf("SavedCost = %v, want -0.02", p.SavedCost)
	}
	if p.CorrectnessDependsOn {
		t.Error("CorrectnessDependsOn must always be false — the law")
	}
}

// TestProveMemoReusePays: with cheap validation and a calm world, a second access
// already clears the gate, and the saved arithmetic is exact.
func TestProveMemoReusePays(t *testing.T) {
	p := ProveMemo(MemoInput{Accesses: 2, ValidateCost: 0.02, MutationRate: 0.1, CaptureCost: 0.02})
	if p.Status != ProofProven {
		t.Fatalf("reuse must PROVE, got %s (%s)", p.Status, p.Reason)
	}
	if p.Decision != "memoize" {
		t.Errorf("decision = %q, want memoize", p.Decision)
	}
	// memo = (1+(k-1)m)(1+c) + (k-1)v = 1.1*1.02 + 0.02 = 1.142; saved = 2 - 1.142.
	if !approx(p.MemoCost, 1.142) {
		t.Errorf("MemoCost = %v, want 1.142", p.MemoCost)
	}
	if !approx(p.SavedCost, 0.858) {
		t.Errorf("SavedCost = %v, want 0.858", p.SavedCost)
	}
	if p.BreakEvenAccesses != 2 {
		t.Errorf("BreakEvenAccesses = %d, want 2", p.BreakEvenAccesses)
	}
}

// TestProveMemoVolatileStateRefutes: when the world almost always mutates between
// accesses, no reuse count can pay — the gate refutes for ALL k and break-even is
// "never". This is the anti-gaming heart: you cannot claim a saving on volatile state.
func TestProveMemoVolatileStateRefutes(t *testing.T) {
	for _, k := range []int{2, 5, 100, 10000} {
		p := ProveMemo(MemoInput{Accesses: k, ValidateCost: 0.02, MutationRate: 0.99, CaptureCost: 0.02})
		if p.Status != ProofRefuted {
			t.Errorf("k=%d volatile: want REFUTED, got %s (%s)", k, p.Status, p.Reason)
		}
		if p.PerReuseNetGain > 0 {
			t.Errorf("k=%d volatile: per-reuse net gain %v should be <= 0", k, p.PerReuseNetGain)
		}
		if p.BreakEvenAccesses != neverBreakEven {
			t.Errorf("k=%d volatile: break-even should be 'never', got %d", k, p.BreakEvenAccesses)
		}
	}
}

// TestProveMemoExpensiveValidationRefutes: a call whose validation costs as much as
// executing it (a non-fingerprintable call) is never worth caching, at any reuse.
func TestProveMemoExpensiveValidationRefutes(t *testing.T) {
	p := ProveMemo(MemoInput{Accesses: 50, ValidateCost: 1.0, MutationRate: 0.05, CaptureCost: 0.02})
	if p.Status != ProofRefuted {
		t.Fatalf("expensive validation must REFUTE, got %s (%s)", p.Status, p.Reason)
	}
	if p.BreakEvenAccesses != neverBreakEven {
		t.Errorf("break-even should be 'never', got %d", p.BreakEvenAccesses)
	}
}

// TestProveMemoBreakEvenMonotone: above the break-even the proof proves; below it it
// refutes — the boundary is real and self-consistent.
func TestProveMemoBreakEvenMonotone(t *testing.T) {
	base := MemoInput{ValidateCost: 0.05, MutationRate: 0.2, CaptureCost: 0.5}
	be := ProveMemo(MemoInput{Accesses: 2, ValidateCost: base.ValidateCost, MutationRate: base.MutationRate, CaptureCost: base.CaptureCost}).BreakEvenAccesses
	if be == neverBreakEven {
		t.Fatal("expected a finite break-even for this calibration")
	}
	below := ProveMemo(MemoInput{Accesses: be - 1, ValidateCost: base.ValidateCost, MutationRate: base.MutationRate, CaptureCost: base.CaptureCost})
	at := ProveMemo(MemoInput{Accesses: be, ValidateCost: base.ValidateCost, MutationRate: base.MutationRate, CaptureCost: base.CaptureCost})
	if below.Status != ProofRefuted {
		t.Errorf("k=be-1=%d must REFUTE, got %s", be-1, below.Status)
	}
	if at.Status != ProofProven {
		t.Errorf("k=be=%d must PROVE, got %s (saved=%v)", be, at.Status, at.SavedCost)
	}
}

// TestAccountAmplification: 4 real dispatches that serve 6 more from the vDSO is a
// 2.5x amplification — a naive agent would have spent 10 round-trips for the same end.
func TestAccountAmplification(t *testing.T) {
	r := Account(Tally{Execute: 4, MemoHit: 6})
	// Each memo hit pays the ValidateFloor (0.01), never free: executed = 4 + 6*0.01 = 4.06.
	if !approx(r.ExecutedTurns, 4.06) || !approx(r.EffectiveTurns, 10) {
		t.Fatalf("executed=%v effective=%v, want 4.06 and 10", r.ExecutedTurns, r.EffectiveTurns)
	}
	if !approx(r.Amplification, 10.0/4.06) {
		t.Errorf("amplification = %v, want %v", r.Amplification, 10.0/4.06)
	}
	if r.Status != "amplifying" || r.Grade != "B" {
		t.Errorf("status/grade = %s/%s, want amplifying/B", r.Status, r.Grade)
	}
	if r.RawTurns != 10 || !approx(r.AvoidedTurns, 10-4.06) {
		t.Errorf("raw=%d avoided=%v, want 10 and %v", r.RawTurns, r.AvoidedTurns, 10-4.06)
	}
	if r.Schema != "fak.callavoid.turns.v1" {
		t.Errorf("schema = %q", r.Schema)
	}
}

// TestAccountProductiveDenyDrivesAmplification: one free productive deny that prunes a
// large futile sub-tree is the "reaches a state a naive path cannot" regime — and the
// fan-out is capped and SURFACED, so amplification is a lower bound, never inflated.
func TestAccountProductiveDeny(t *testing.T) {
	r := Account(Tally{Execute: 1, Redirects: []int{200, 5000}})
	// pruned = 200 + min(5000, 1024) = 1224; one fan-out was clamped.
	if r.RedirectPruned != 1224 {
		t.Errorf("RedirectPruned = %d, want 1224", r.RedirectPruned)
	}
	if r.RedirectCapped != 1 {
		t.Errorf("RedirectCapped = %d, want 1 (the 5000 clamp, surfaced)", r.RedirectCapped)
	}
	if !approx(r.EffectiveTurns, 1225) || !approx(r.ExecutedTurns, 1) {
		t.Errorf("effective=%v executed=%v, want 1225 and 1", r.EffectiveTurns, r.ExecutedTurns)
	}
	if r.Grade != "A" {
		t.Errorf("grade = %s, want A", r.Grade)
	}
	if len(r.Risks) == 0 {
		t.Error("a clamped fan-out must surface a lower-bound risk, not silently inflate")
	}
}

// TestAccountStaleMissRegresses: a window dominated by stale misses is a NET LOSS — the
// cache bet cost more than it saved — and the report says so honestly.
func TestAccountStaleMissRegresses(t *testing.T) {
	r := Account(Tally{StaleMiss: 3, ValidateCost: 0.02, CaptureCost: 0.02})
	// executed = 3*(0.02+1+0.02) = 3.12 ; naive = 3 ; amp = 0.961...
	if !approx(r.ExecutedTurns, 3.12) {
		t.Errorf("executed = %v, want 3.12", r.ExecutedTurns)
	}
	if r.Status != "regressing" || r.Grade != "F" {
		t.Errorf("status/grade = %s/%s, want regressing/F", r.Status, r.Grade)
	}
	if r.Amplification >= 1 {
		t.Errorf("amplification = %v, want < 1", r.Amplification)
	}
}

// TestAccountAllAvoidedIsFinite: a window where every call was served locally is the
// ideal "no tool call" extreme — maximally amplifying (grade A) but FINITE, not +Inf:
// a memo hit still pays the ValidateFloor (a world-version check is never free), so a
// pure-cache window caps at 1/ValidateFloor = 100x (#817). It must not divide-by-zero.
func TestAccountAllAvoided(t *testing.T) {
	r := Account(Tally{MemoHit: 5})
	if math.IsInf(r.Amplification, 1) {
		t.Errorf("all-avoided amplification = +Inf, want finite (a memo hit pays ValidateFloor)")
	}
	if !approx(r.Amplification, 1.0/ValidateFloor) {
		t.Errorf("all-avoided amplification = %v, want %v (1/ValidateFloor)", r.Amplification, 1.0/ValidateFloor)
	}
	if r.Grade != "A" || r.Status != "amplifying" {
		t.Errorf("grade/status = %s/%s, want A/amplifying", r.Grade, r.Status)
	}
}

// TestAccountStaleMissDefaultCostsRegress: at the DEFAULT zero costs a stale miss must
// still read as a net loss (#817). A stale miss is overhead a naive agent never paid;
// the ValidateFloor makes it strictly >1 paid (validate 0.01 + re-run 1), so the window
// regresses instead of reading the old, wrong break-even (which priced it at exactly 1).
func TestAccountStaleMissDefaultCostsRegress(t *testing.T) {
	r := Account(Tally{StaleMiss: 4})
	// executed = 4*(ValidateFloor + 1 + 0) = 4.04 ; naive = 4 ; amp = 4/4.04 < 1.
	if !approx(r.ExecutedTurns, 4*(ValidateFloor+1)) {
		t.Errorf("executed = %v, want %v", r.ExecutedTurns, 4*(ValidateFloor+1))
	}
	if r.Status != "regressing" {
		t.Errorf("status = %s, want regressing (a stale miss is a strict loss)", r.Status)
	}
	if r.Amplification >= 1 {
		t.Errorf("amplification = %v, want < 1", r.Amplification)
	}
}

// TestAccountNonNegativeCountGuards: a negative scalar count can never inflate the
// ratio — it is floored to zero, symmetric with the existing Redirects/cost guards (#817).
func TestAccountNonNegativeCountGuards(t *testing.T) {
	r := Account(Tally{Execute: -3, MemoHit: -2, Repair: -1, StaleMiss: -4, HardDeny: -5})
	if r.RawTurns != 0 {
		t.Errorf("raw turns = %d, want 0 (all negative counts floored)", r.RawTurns)
	}
	if r.MemoHits != 0 || r.StaleMisses != 0 || r.HardDenies != 0 || r.Repairs != 0 {
		t.Errorf("guarded counts = memo:%d stale:%d hard:%d repair:%d, want all 0",
			r.MemoHits, r.StaleMisses, r.HardDenies, r.Repairs)
	}
	if r.Status != "break_even" {
		t.Errorf("status = %s, want break_even (an empty window after guarding)", r.Status)
	}
}

// TestAccountEmptyWindowIsBreakEven: nothing happened — neither amplifying nor
// regressing, and no divide-by-zero.
func TestAccountEmptyWindow(t *testing.T) {
	r := Account(Tally{})
	if math.Abs(r.Amplification-1) > eps {
		t.Errorf("empty amplification = %v, want 1", r.Amplification)
	}
	if r.Status != "break_even" {
		t.Errorf("status = %s, want break_even", r.Status)
	}
}

// TestHardDenyIsSymmetric: a plain fast-reject with no forward guidance adds nothing
// to either side — it must NOT be credited as amplification (only a PRODUCTIVE deny is).
func TestHardDenyIsSymmetric(t *testing.T) {
	with := Account(Tally{Execute: 2, HardDeny: 5})
	without := Account(Tally{Execute: 2})
	if !approx(with.Amplification, without.Amplification) {
		t.Errorf("hard denies changed amplification %v vs %v — they must be symmetric", with.Amplification, without.Amplification)
	}
	if with.HardDenies != 5 || with.RawTurns != 7 {
		t.Errorf("hard denies should still be counted: HardDenies=%d RawTurns=%d", with.HardDenies, with.RawTurns)
	}
}

// TestDeterministic: pure functions — identical input yields byte-identical output.
func TestDeterministic(t *testing.T) {
	in := Tally{Execute: 3, MemoHit: 4, Repair: 1, StaleMiss: 1, HardDeny: 2, Redirects: []int{10, 20}, ValidateCost: 0.01, CaptureCost: 0.01}
	a, b := Account(in), Account(in)
	if !reflect.DeepEqual(a, b) {
		t.Error("Account is not deterministic")
	}
	m := MemoInput{Accesses: 7, ValidateCost: 0.03, MutationRate: 0.15, CaptureCost: 0.2}
	if !reflect.DeepEqual(ProveMemo(m), ProveMemo(m)) {
		t.Error("ProveMemo is not deterministic")
	}
}

// TestJSONRoundTrips: both reports marshal to stable, schema-tagged JSON (the artifact
// shape a scorecard or CI line consumes).
func TestJSONRoundTrips(t *testing.T) {
	rb, err := json.Marshal(Account(Tally{Execute: 1, MemoHit: 1}))
	if err != nil {
		t.Fatalf("marshal TurnReport: %v", err)
	}
	var rr map[string]any
	if err := json.Unmarshal(rb, &rr); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if rr["schema"] != "fak.callavoid.turns.v1" {
		t.Errorf("schema = %v", rr["schema"])
	}
	pb, err := json.Marshal(ProveMemo(MemoInput{Accesses: 3, ValidateCost: 0.02, MutationRate: 0.1, CaptureCost: 0.02}))
	if err != nil {
		t.Fatalf("marshal MemoProof: %v", err)
	}
	var pp map[string]any
	if err := json.Unmarshal(pb, &pp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if pp["correctness_depends_on_hit"] != false {
		t.Errorf("correctness_depends_on_hit = %v, want false", pp["correctness_depends_on_hit"])
	}
}

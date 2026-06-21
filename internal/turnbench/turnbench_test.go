package turnbench

import (
	"context"
	"testing"

	// Blank-import the built-in driver list so the full ABI (resolver, vDSO,
	// adjudicator, ctx-MMU, normgate, ifc, witness, engines) is wired before
	// kernel.New / agent.Configure run inside Run.
	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

const (
	airlineTrace = "../../testdata/turntax/turntax-airline.json"
	happyTrace   = "../../testdata/turntax/turntax-happy.json"
)

// The airline fixture is designed to fire exactly one event per class. These are
// the EXPECTED live-kernel counts; they document the design and catch a
// regression in any rung (grammar, vDSO tiers, ctx-MMU quarantine, policy deny).
const (
	wantGrammar    = 2                                             // two aliased convert_currency calls
	wantVDSOPure   = 2                                             // two calculate calls (tier-1)
	wantVDSODedup  = 3                                             // get_user x2 + search x1 duplicates (tier-2)
	wantVDSOStatic = 2                                             // two list_all_airports calls (tier-3)
	wantQuarantine = 1                                             // one poisoned fetch_policy
	wantDeny       = 1                                             // one delete_account
	wantPass       = 3                                             // first get_user + first search + book_flight
	wantVDSOTotal  = wantVDSOPure + wantVDSODedup + wantVDSOStatic // 7
	wantTurnsSaved = wantGrammar + wantVDSOTotal                   // 9
)

func TestRun_AirlineClassesAreLiveKernelEvents(t *testing.T) {
	rep, err := Run(context.Background(), mustLoad(t, airlineTrace), DefaultCostModel())
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if rep.Provenance.AppVersion == "" {
		t.Fatal("report provenance app_version is empty")
	}
	if rep.Cost.Version != CostModelVersion {
		t.Fatalf("cost model version=%q, want %q", rep.Cost.Version, CostModelVersion)
	}
	for _, l := range rep.Levers {
		if l.Version != BenchmarkConceptVersion {
			t.Fatalf("lever %q version=%q, want %q", l.Name, l.Version, BenchmarkConceptVersion)
		}
	}

	// The consistency guard: the kernel's aggregate counters agree with the
	// per-call classification. This catches a bucketing/wiring drift in the bench;
	// it is NOT an independent oracle (classify reads the same verdicts the counters
	// came from). If it is not "ok" the bench's bookkeeping is internally broken.
	if rep.ConsistencyCheck != "ok" {
		t.Fatalf("consistency check failed: %s", rep.ConsistencyCheck)
	}

	cb := rep.Class
	for _, c := range []struct {
		name      string
		got, want int
	}{
		{"grammar", cb.Grammar, wantGrammar},
		{"vdso_pure", cb.VDSOPure, wantVDSOPure},
		{"vdso_dedup", cb.VDSODedup, wantVDSODedup},
		{"vdso_static", cb.VDSOStatic, wantVDSOStatic},
		{"quarantine", cb.Quarantine, wantQuarantine},
		{"deny", cb.Deny, wantDeny},
		{"pass", cb.Pass, wantPass},
	} {
		if c.got != c.want {
			t.Errorf("class %s = %d, want %d (fixture/mechanism drift)", c.name, c.got, c.want)
		}
	}

	// Counters and the per-call classification must agree (the consistency guard;
	// they read the same verdicts, so this catches bench bucketing drift, not a
	// kernel-verdict error).
	if int(rep.Counters.Transforms) != wantGrammar {
		t.Errorf("Counters.Transforms = %d, want %d", rep.Counters.Transforms, wantGrammar)
	}
	if int(rep.Counters.VDSOHits) != wantVDSOTotal {
		t.Errorf("Counters.VDSOHits = %d, want %d", rep.Counters.VDSOHits, wantVDSOTotal)
	}
	if int(rep.Counters.Quarantines) != wantQuarantine {
		t.Errorf("Counters.Quarantines = %d, want %d", rep.Counters.Quarantines, wantQuarantine)
	}
	if int(rep.Counters.Denies) != wantDeny {
		t.Errorf("Counters.Denies = %d, want %d", rep.Counters.Denies, wantDeny)
	}
}

func TestRun_NetTurnTaxMath(t *testing.T) {
	cm := DefaultCostModel()
	rep, err := Run(context.Background(), mustLoad(t, airlineTrace), cm)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if rep.Net.TurnsSaved != wantTurnsSaved {
		t.Fatalf("Net.TurnsSaved = %d, want %d", rep.Net.TurnsSaved, wantTurnsSaved)
	}
	// The forced/elision split is honest about WHICH turns the baseline demonstrably
	// pays (re-issued read + aliased retry) vs optional tool calls a stronger model
	// could elide (pure + static). They must sum to the total.
	if want := wantGrammar + wantVDSODedup; rep.TurnKinds.Forced != want { // 2 + 3 = 5
		t.Errorf("TurnKinds.Forced = %d, want %d (grammar+dedup)", rep.TurnKinds.Forced, want)
	}
	if want := wantVDSOPure + wantVDSOStatic; rep.TurnKinds.Elision != want { // 2 + 2 = 4
		t.Errorf("TurnKinds.Elision = %d, want %d (pure+static)", rep.TurnKinds.Elision, want)
	}
	if rep.TurnKinds.Forced+rep.TurnKinds.Elision != rep.Net.TurnsSaved {
		t.Errorf("forced(%d)+elision(%d) != turns_saved(%d)", rep.TurnKinds.Forced, rep.TurnKinds.Elision, rep.Net.TurnsSaved)
	}
	if want := wantTurnsSaved * (cm.PromptTokensPerTurn + cm.CompletionTokensPerTurn); rep.Net.TokensSaved != want {
		t.Errorf("Net.TokensSaved = %d, want %d", rep.Net.TokensSaved, want)
	}
	if want := float64(wantTurnsSaved) * cm.ModelTurnLatencyMs; rep.Net.LatencySavedMs != want {
		t.Errorf("Net.LatencySavedMs = %v, want %v", rep.Net.LatencySavedMs, want)
	}
	if rep.Net.DollarsSaved <= 0 {
		t.Errorf("Net.DollarsSaved = %v, want > 0", rep.Net.DollarsSaved)
	}
	if rep.LocalServeNs <= 0 {
		t.Errorf("LocalServeNs = %d, want > 0 (calibration loop measures real time)", rep.LocalServeNs)
	}
}

// The vDSO lever is proven by a REAL ON/OFF path swap, not by arithmetic: with
// the fast path disabled, the vDSO-served calls fall through to the engine, so
// the ONLY turns saved are the grammar repairs — and the difference equals the
// live VDSOHits count.
func TestRun_VDSOAblationIsARealPathSwap(t *testing.T) {
	rep, err := Run(context.Background(), mustLoad(t, airlineTrace), DefaultCostModel())
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if rep.VDSOOffNet.TurnsSaved != wantGrammar {
		t.Errorf("vdso-off turns saved = %d, want %d (grammar only)", rep.VDSOOffNet.TurnsSaved, wantGrammar)
	}
	leverDelta := rep.Net.TurnsSaved - rep.VDSOOffNet.TurnsSaved
	if leverDelta != int(rep.Counters.VDSOHits) {
		t.Errorf("vDSO lever delta = %d, want VDSOHits = %d", leverDelta, rep.Counters.VDSOHits)
	}
}

func TestRun_SafetyFloorIsSeparateFromTurnTax(t *testing.T) {
	rep, err := Run(context.Background(), mustLoad(t, airlineTrace), DefaultCostModel())
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if rep.Safety.InjectionsAdmittedBaseline != wantQuarantine {
		t.Errorf("baseline injections admitted = %d, want %d", rep.Safety.InjectionsAdmittedBaseline, wantQuarantine)
	}
	if rep.Safety.InjectionsAdmittedFak != 0 {
		t.Errorf("fak injections admitted = %d, want 0", rep.Safety.InjectionsAdmittedFak)
	}
	if rep.Safety.DestructiveExecutedBaseline != 1 {
		t.Errorf("baseline destructive executed = %d, want 1 (delete_account)", rep.Safety.DestructiveExecutedBaseline)
	}
	if rep.Safety.DestructiveExecutedFak != 0 {
		t.Errorf("fak destructive executed = %d, want 0", rep.Safety.DestructiveExecutedFak)
	}
}

// The anti-inflation control: a clean happy path saves NOTHING. If this is ever
// non-zero the benchmark is applying a fixed per-call discount, not measuring
// real avoided errors.
func TestRun_HappyPathSavesNothing(t *testing.T) {
	rep, err := Run(context.Background(), mustLoad(t, happyTrace), DefaultCostModel())
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if rep.ConsistencyCheck != "ok" {
		t.Fatalf("consistency check failed: %s", rep.ConsistencyCheck)
	}
	if rep.Net.TurnsSaved != 0 {
		t.Errorf("happy-path turns saved = %d, want 0 (no alias, no dup, no poison)", rep.Net.TurnsSaved)
	}
	if rep.Net.TokensSaved != 0 || rep.Net.DollarsSaved != 0 || rep.Net.LatencySavedMs != 0 {
		t.Errorf("happy-path net non-zero: tokens=%d $=%v ms=%v", rep.Net.TokensSaved, rep.Net.DollarsSaved, rep.Net.LatencySavedMs)
	}
	if rep.Class.Quarantine != 0 || rep.Class.Deny != 0 {
		t.Errorf("happy-path safety events: quarantine=%d deny=%d, want 0/0", rep.Class.Quarantine, rep.Class.Deny)
	}
	if rep.Class.Pass != 3 {
		t.Errorf("happy-path pass = %d, want 3", rep.Class.Pass)
	}
}

func TestWorkloadHash_StableAndIgnoresClassLabels(t *testing.T) {
	tr := mustLoad(t, airlineTrace)
	h1 := tr.WorkloadHash()
	if h1 == "" {
		t.Fatal("empty workload hash")
	}
	if h2 := mustLoad(t, airlineTrace).WorkloadHash(); h1 != h2 {
		t.Errorf("hash not stable across reloads: %q != %q", h1, h2)
	}
	// Relabeling a call's documentation class must NOT change the workload hash
	// (the hash is over tool+args+meta only).
	tr2 := mustLoad(t, airlineTrace)
	for i := range tr2.Calls {
		tr2.Calls[i].Class = "relabeled"
		tr2.Calls[i].Note = "x"
	}
	if got := tr2.WorkloadHash(); got != h1 {
		t.Errorf("hash changed after relabeling class/note: %q != %q", got, h1)
	}
}

func mustLoad(t *testing.T, path string) *Trace {
	t.Helper()
	tr, err := LoadTrace(path)
	if err != nil {
		t.Fatalf("LoadTrace(%q): %v", path, err)
	}
	return tr
}

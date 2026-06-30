package commitrollup

import (
	"reflect"
	"strings"
	"testing"
)

func TestPlanBatchBatchesDisjointCompatibleIntents(t *testing.T) {
	got := PlanBatch([]Intent{
		{
			ID:        "intent-a",
			Submitter: "worker-a",
			Paths:     []string{`internal\gateway\a.go`, "./internal/gateway/b.go"},
			Stamp:     "(fak gateway)",
			Witnesses: []string{"preview:a"},
		},
		{
			ID:        "intent-b",
			Submitter: "worker-b",
			Paths:     []string{"internal/gateway/c.go"},
			Stamp:     "gateway",
			Witnesses: []string{"preview:b"},
		},
	}, Config{})

	if !got.OK {
		t.Fatalf("plan OK = false, refusals=%+v", got.Refusals)
	}
	if got.Schema != Schema || !got.RollupEnabled {
		t.Fatalf("schema/enabled = %q/%v", got.Schema, got.RollupEnabled)
	}
	if got.Stamp != "gateway" || got.Trailer != "(fak gateway)" {
		t.Fatalf("stamp/trailer = %q/%q", got.Stamp, got.Trailer)
	}
	wantIDs := []string{"intent-a", "intent-b"}
	if !reflect.DeepEqual(got.IntentIDs, wantIDs) {
		t.Fatalf("intent ids = %v, want %v", got.IntentIDs, wantIDs)
	}
	wantPaths := []string{"internal/gateway/a.go", "internal/gateway/b.go", "internal/gateway/c.go"}
	if !reflect.DeepEqual(got.UnionPaths, wantPaths) {
		t.Fatalf("union paths = %v, want %v", got.UnionPaths, wantPaths)
	}
	if !reflect.DeepEqual(got.Submitters, []string{"worker-a", "worker-b"}) {
		t.Fatalf("submitters = %v", got.Submitters)
	}
	if !reflect.DeepEqual(got.Witnesses, []string{"preview:a", "preview:b"}) {
		t.Fatalf("witnesses = %v", got.Witnesses)
	}
	if len(got.Refusals) != 0 {
		t.Fatalf("refusals = %+v, want none", got.Refusals)
	}
	for _, want := range []string{"roll up commit intents intent-a, intent-b", "(fak gateway)"} {
		if !strings.Contains(got.Subject, want) {
			t.Fatalf("subject %q missing %q", got.Subject, want)
		}
	}
}

func TestPlanBatchRefusesOverlappingPaths(t *testing.T) {
	got := PlanBatch([]Intent{
		{ID: "base", Paths: []string{"internal/gateway"}, Stamp: "gateway"},
		{ID: "overlap", Paths: []string{"internal/gateway/http.go"}, Stamp: "gateway"},
		{ID: "ok", Paths: []string{"internal/gateway/debug.go"}, Stamp: "gateway"},
	}, Config{})

	if !reflect.DeepEqual(got.IntentIDs, []string{"base"}) {
		t.Fatalf("intent ids = %v, want only base", got.IntentIDs)
	}
	if len(got.Refusals) != 2 {
		t.Fatalf("refusals = %+v, want two overlaps", got.Refusals)
	}
	for _, r := range got.Refusals {
		if r.Reason != ReasonOverlappingPath {
			t.Fatalf("reason = %s, want %s in %+v", r.Reason, ReasonOverlappingPath, got.Refusals)
		}
	}
}

func TestPlanBatchRefusesIncompatibleStamp(t *testing.T) {
	got := PlanBatch([]Intent{
		{ID: "gateway", Paths: []string{"internal/gateway/http.go"}, Stamp: "gateway"},
		{ID: "safecommit", Paths: []string{"internal/safecommit/safecommit.go"}, Stamp: "safecommit"},
	}, Config{})

	if !reflect.DeepEqual(got.IntentIDs, []string{"gateway"}) {
		t.Fatalf("intent ids = %v, want gateway only", got.IntentIDs)
	}
	if len(got.Refusals) != 1 {
		t.Fatalf("refusals = %+v, want one", got.Refusals)
	}
	r := got.Refusals[0]
	if r.IntentID != "safecommit" || r.Reason != ReasonIncompatibleStamp {
		t.Fatalf("refusal = %+v, want safecommit/%s", r, ReasonIncompatibleStamp)
	}
	if !strings.Contains(r.Detail, "(fak safecommit)") || !strings.Contains(r.Detail, "(fak gateway)") {
		t.Fatalf("refusal detail should name both stamps, got %q", r.Detail)
	}
}

func TestPlanBatchRefusesStaleAndRefusedInputs(t *testing.T) {
	got := PlanBatch([]Intent{
		{ID: "stale", Paths: []string{"internal/gateway/a.go"}, Stamp: "gateway", Stale: true},
		{ID: "refused", Paths: []string{"internal/gateway/b.go"}, Stamp: "gateway", Refused: true},
		{ID: "valid", Paths: []string{"internal/gateway/c.go"}, Stamp: "gateway"},
	}, Config{})

	if !reflect.DeepEqual(got.IntentIDs, []string{"valid"}) {
		t.Fatalf("intent ids = %v, want valid only", got.IntentIDs)
	}
	if len(got.Refusals) != 2 {
		t.Fatalf("refusals = %+v, want two", got.Refusals)
	}
	reasons := map[string]Reason{}
	for _, r := range got.Refusals {
		reasons[r.IntentID] = r.Reason
	}
	if reasons["stale"] != ReasonStaleInput {
		t.Fatalf("stale reason = %s", reasons["stale"])
	}
	if reasons["refused"] != ReasonRefusedInput {
		t.Fatalf("refused reason = %s", reasons["refused"])
	}
}

func TestPlanBatchDisabledKeepsOneIntentMode(t *testing.T) {
	got := PlanBatch([]Intent{
		{ID: "first", Paths: []string{"internal/gateway/a.go"}, Stamp: "gateway"},
		{ID: "second", Paths: []string{"internal/gateway/b.go"}, Stamp: "gateway"},
	}, Config{DisableRollup: true})

	if got.RollupEnabled {
		t.Fatal("rollup enabled = true, want disabled")
	}
	if !reflect.DeepEqual(got.IntentIDs, []string{"first"}) {
		t.Fatalf("intent ids = %v, want first only", got.IntentIDs)
	}
	if len(got.Refusals) != 1 || got.Refusals[0].Reason != ReasonRollupDisabled {
		t.Fatalf("refusals = %+v, want ROLLUP_DISABLED", got.Refusals)
	}
}

func TestPlanBatchRefusesMalformedInputs(t *testing.T) {
	got := PlanBatch([]Intent{
		{ID: "", Paths: []string{"internal/gateway/a.go"}, Stamp: "gateway"},
		{ID: "no-paths", Stamp: "gateway"},
		{ID: "no-stamp", Paths: []string{"internal/gateway/b.go"}},
		{ID: "bad-stamp", Paths: []string{"internal/gateway/c.go"}, Stamp: "(fak Gateway)"},
		{ID: "bad-path", Paths: []string{"../outside"}, Stamp: "gateway"},
	}, Config{})

	if got.OK {
		t.Fatalf("OK = true with no included intents: %+v", got)
	}
	want := []Reason{ReasonMissingID, ReasonMissingPathset, ReasonMissingStamp, ReasonInvalidStamp, ReasonInvalidPath}
	if len(got.Refusals) != len(want) {
		t.Fatalf("refusals = %+v, want %d", got.Refusals, len(want))
	}
	for i, reason := range want {
		if got.Refusals[i].Reason != reason {
			t.Fatalf("refusal[%d] = %s, want %s", i, got.Refusals[i].Reason, reason)
		}
	}
}

func TestAssertPathsetDetectsHiddenExpansion(t *testing.T) {
	plan := PlanBatch([]Intent{
		{ID: "a", Paths: []string{"internal/gateway/a.go"}, Stamp: "gateway"},
		{ID: "b", Paths: []string{"internal/gateway/b.go"}, Stamp: "gateway"},
	}, Config{})

	ok := plan.AssertPathset([]string{"./internal/gateway/b.go", `internal\gateway\a.go`})
	if !ok.OK {
		t.Fatalf("assertion should pass after normalization: %+v", ok)
	}

	bad := plan.AssertPathset([]string{"internal/gateway/a.go", "internal/gateway/extra.go"})
	if bad.OK || bad.Reason != ReasonPathsetMismatch {
		t.Fatalf("assertion = %+v, want PATHSET_MISMATCH", bad)
	}
	if !reflect.DeepEqual(bad.Missing, []string{"internal/gateway/b.go"}) {
		t.Fatalf("missing = %v", bad.Missing)
	}
	if !reflect.DeepEqual(bad.Extra, []string{"internal/gateway/extra.go"}) {
		t.Fatalf("extra = %v", bad.Extra)
	}
}

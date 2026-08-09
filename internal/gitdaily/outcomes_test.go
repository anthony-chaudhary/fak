package gitdaily

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// TestOutcomeClassifiesTheThreeInvocationResults pins the vocabulary each recorded tick
// is folded into. The precedence matters: a run that drifted posture AND deferred its
// fold tier must read as an ERROR, because a refusal is something to watch while an
// incident is something to repair.
func TestOutcomeClassifiesTheThreeInvocationResults(t *testing.T) {
	for _, tc := range []struct {
		name string
		row  Row
		want Outcome
	}{
		{"healthy fold", Row{Day: "2026-08-01", LooseBefore: 400, LooseAfter: 0}, OutcomeOK},
		{"fold tier held back", Row{Day: "2026-08-02", GraceRefused: "LOCKED"}, OutcomeRefused},
		{"opted-in prune held back", Row{Day: "2026-08-03", GracePruneRefused: "SESSION_LIVE"}, OutcomeRefused},
		{"incident", Row{Day: "2026-08-04", Incident: true}, OutcomeError},
		{"incident outranks refusal", Row{Day: "2026-08-05", GraceRefused: "POSTURE_DRIFT", Incident: true}, OutcomeError},
	} {
		if got := tc.row.Outcome(); got != tc.want {
			t.Errorf("%s: outcome = %q, want %q", tc.name, got, tc.want)
		}
	}
}

// TestPruneOffIsNotARefusal is the load-bearing classification call. The grace-prune tier
// is opt-in and default-OFF, so PRUNE_OFF is the CONFIGURED posture of a perfectly
// healthy run — every default run carries it. Counting it would report 100% refused and
// bury the LOCKED streak these counters exist to surface.
func TestPruneOffIsNotARefusal(t *testing.T) {
	healthy := Row{Day: "2026-08-04", LooseBefore: 4129, LooseAfter: 0, GracePruneRefused: "PRUNE_OFF"}
	if got := healthy.Outcome(); got != OutcomeOK {
		t.Fatalf("a default (never opted into prune) run classified %q, want %q", got, OutcomeOK)
	}
	if reason := healthy.RefusalReason(); reason != "" {
		t.Fatalf("PRUNE_OFF surfaced as a refusal reason %q", reason)
	}
	// The same field carrying any OTHER reason can only happen when the operator DID
	// opt in, so that one is a real refusal.
	optedIn := Row{Day: "2026-08-04", GracePruneRefused: "PRUNE_EXPIRE_UNSAFE"}
	if reason := optedIn.RefusalReason(); reason != "PRUNE_EXPIRE_UNSAFE" {
		t.Fatalf("opted-in prune refusal reason = %q", reason)
	}
}

// TestFoldOutcomesTalliesAHistory folds a mixed history end to end: the counts, the
// per-reason breakdown, the window bounds, and the folded volume.
func TestFoldOutcomesTalliesAHistory(t *testing.T) {
	got := FoldOutcomes([]Row{
		{Day: "2026-08-01", LooseBefore: 4200, LooseAfter: 12},
		{Day: "2026-08-02", GraceRefused: "LOCKED"},
		{Day: "2026-08-03", GraceRefused: "LOCKED"},
		{Day: "2026-08-04", GraceRefused: "POSTURE_DRIFT", Incident: true},
		{Day: "2026-08-05", LooseBefore: 90, LooseAfter: 40, GracePruneRefused: "PRUNE_OFF"},
	})
	want := Outcomes{
		Runs: 5, OK: 2, Refused: 2, Errors: 1,
		Reasons:  map[string]int{"LOCKED": 2},
		FirstDay: "2026-08-01", LastDay: "2026-08-05",
		LooseFolded: 4238,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("fold =\n  %+v\nwant\n  %+v", got, want)
	}
}

// TestFoldOutcomesIgnoresPeerLooseGrowth: a peer committing during the tick can leave
// MORE loose objects than the run folded. That is their write volume, not this job
// running backwards, so it must never be subtracted from the work it did.
func TestFoldOutcomesIgnoresPeerLooseGrowth(t *testing.T) {
	got := FoldOutcomes([]Row{
		{Day: "2026-08-01", LooseBefore: 100, LooseAfter: 0},
		{Day: "2026-08-02", LooseBefore: 10, LooseAfter: 900},
	})
	if got.LooseFolded != 100 {
		t.Fatalf("LooseFolded = %d, want 100 (the growth row must not net it out)", got.LooseFolded)
	}
	if got.OK != 2 {
		t.Fatalf("OK = %d, want 2 — growing loose objects is not an outcome failure", got.OK)
	}
}

// TestFoldOutcomesOnEmptyHistory keeps the first-run contract: a clone that has never
// ticked folds to zeros, not to a nil-map panic or a bogus window.
func TestFoldOutcomesOnEmptyHistory(t *testing.T) {
	got := FoldOutcomes(nil)
	if got.Runs != 0 || got.Reasons != nil || got.FirstDay != "" || len(got.ReasonsByCount()) != 0 {
		t.Fatalf("empty history folded to %+v", got)
	}
}

// TestReasonsByCountIsStable: the breakdown is rendered commonest-first with ties broken
// by name, so an operator diffing two days' readouts sees real change, not map churn.
func TestReasonsByCountIsStable(t *testing.T) {
	o := Outcomes{Reasons: map[string]int{"LOCKED": 9, "POSTURE_DRIFT": 1, "SESSION_LIVE": 9}}
	want := []string{"LOCKED", "SESSION_LIVE", "POSTURE_DRIFT"}
	for i := 0; i < 5; i++ {
		if got := o.ReasonsByCount(); !reflect.DeepEqual(got, want) {
			t.Fatalf("order = %v, want %v", got, want)
		}
	}
}

// TestFoldOutcomesReadsTheRealLedgerFormat closes the loop the issue actually asks for:
// the counts must come off the bytes the tick WRITES, not off hand-built structs. This
// replays real `fak-git-daily/1` rows captured from this clone's own ledger through
// Status (the surface `--status` reads) and folds them.
func TestFoldOutcomesReadsTheRealLedgerFormat(t *testing.T) {
	path := filepath.Join(t.TempDir(), LedgerName)
	captured := []string{
		`{"schema":"fak-git-daily/1","day":"2026-08-04","at":"2026-08-04T11:14:37-07:00","lease_locks_reaped":0,"index_locks_reaped":7,"lock_actions":1,"loose_before":4129,"loose_after":4129,"packs_before":4,"packs_after":4,"grace_prune_refused":"PRUNE_OFF"}`,
		`{"schema":"fak-git-daily/1","day":"2026-08-04","at":"2026-08-04T11:18:58-07:00","lease_locks_reaped":0,"index_locks_reaped":0,"lock_actions":0,"loose_before":4129,"loose_after":0,"packs_before":4,"packs_after":2,"grace_prune_refused":"PRUNE_OFF"}`,
		`{"schema":"fak-git-daily/1","day":"2026-08-04","at":"2026-08-04T11:59:46-07:00","lease_locks_reaped":0,"index_locks_reaped":0,"lock_actions":1,"loose_before":48,"loose_after":0,"packs_before":2,"packs_after":4,"grace_prune_refused":"PRUNE_OFF"}`,
		`{"schema":"fak-git-daily/1","day":"2026-08-04","at":"2026-08-04T12:15:16-07:00","lease_locks_reaped":0,"index_locks_reaped":0,"lock_actions":1,"loose_before":34,"loose_after":0,"packs_before":4,"packs_after":4,"grace_prune_refused":"PRUNE_OFF"}`,
	}
	var blob string
	for _, line := range captured {
		blob += line + "\n"
	}
	if err := os.WriteFile(path, []byte(blob), 0o644); err != nil {
		t.Fatal(err)
	}

	got := FoldOutcomes(Status(path, 0))
	if got.Runs != 4 || got.OK != 4 || got.Refused != 0 || got.Errors != 0 {
		t.Fatalf("real-ledger fold = %+v, want 4 runs all ok", got)
	}
	if got.LooseFolded != 4211 {
		t.Fatalf("LooseFolded = %d, want 4211 (4129 + 48 + 34)", got.LooseFolded)
	}
	if got.FirstDay != "2026-08-04" || got.LastDay != "2026-08-04" {
		t.Fatalf("window = %s..%s", got.FirstDay, got.LastDay)
	}

	// The counters must survive the JSON envelope the --status --json surface emits.
	b, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	var back map[string]any
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"runs", "ok", "refused", "errors", "loose_folded"} {
		if _, ok := back[key]; !ok {
			t.Fatalf("counter %q missing from the JSON envelope: %s", key, b)
		}
	}
}

func TestFoldOutcomesByWeekCountsRecordedInvocations(t *testing.T) {
	rows := []Row{
		{At: "2026-08-02T20:00:00Z"}, // Sunday of the 2026-07-27 UTC week.
		{At: "2026-08-03T12:00:00Z", GraceRefused: "LOCKED"},
		{At: "2026-08-09T23:59:59Z", Incident: true},
		{At: "2026-08-10T00:00:00Z"},
		{At: "not-a-timestamp"}, // Never fabricate a week for legacy/bad rows.
	}
	got := FoldOutcomesByWeek(rows)
	want := []WeekOutcome{
		{Week: "2026-07-27", Total: 1, OK: 1},
		{Week: "2026-08-03", Total: 2, Refused: 1, Errors: 1},
		{Week: "2026-08-10", Total: 1, OK: 1},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("weekly outcomes = %#v, want %#v", got, want)
	}
}

func TestFoldOutcomesByWeekEmptyIsStable(t *testing.T) {
	if got := FoldOutcomesByWeek(nil); len(got) != 0 {
		t.Fatalf("weekly outcomes = %#v, want empty", got)
	}
}

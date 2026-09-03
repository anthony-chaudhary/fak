package codexlifecycle

import (
	"strings"
	"testing"
)

// Real rollout lines, shape-matched to the live corpus (~/.codex/sessions):
//
//	{"timestamp":"…","type":"event_msg","payload":{"type":"task_started","turn_id":"…",…}}
//	{"timestamp":"…","type":"event_msg","payload":{"type":"task_complete","turn_id":"…","duration_ms":1382}}
//	{"timestamp":"…","type":"event_msg","payload":{"type":"turn_aborted","turn_id":"…","reason":"interrupted","duration_ms":1391321}}
func started(ts, id string) string {
	return `{"timestamp":"` + ts + `","type":"event_msg","payload":{"type":"task_started","turn_id":"` + id + `","model_context_window":258400}}`
}
func complete(ts, id string) string {
	return `{"timestamp":"` + ts + `","type":"event_msg","payload":{"type":"task_complete","turn_id":"` + id + `","last_agent_message":null,"duration_ms":1382}}`
}
func aborted(ts, id string) string {
	return `{"timestamp":"` + ts + `","type":"event_msg","payload":{"type":"turn_aborted","turn_id":"` + id + `","reason":"interrupted","duration_ms":1391321}}`
}

// noise is a real non-lifecycle record; the fold must step over it, not choke.
const noise = `{"timestamp":"2026-06-10T16:00:10.000Z","type":"response_item","payload":{"type":"function_call","name":"shell","call_id":"c1"}}`

func foldLines(t *testing.T, fresh bool, lines ...string) Report {
	t.Helper()
	ev, err := ParseRollout(strings.NewReader(strings.Join(lines, "\n") + "\n"))
	if err != nil {
		t.Fatalf("ParseRollout: %v", err)
	}
	return Fold(ev, fresh)
}

func onlyTask(t *testing.T, r Report) Task {
	t.Helper()
	if len(r.Tasks) != 1 {
		t.Fatalf("tasks = %d, want 1: %+v", len(r.Tasks), r.Tasks)
	}
	return r.Tasks[0]
}

// FIXTURE 1 — normal completion: an observed task_complete is the terminal, read not
// inferred.
func TestFold_NormalCompletion(t *testing.T) {
	r := foldLines(t, false,
		started("2026-06-10T16:00:09.533Z", "A"), noise, complete("2026-06-10T16:00:10.915Z", "A"))
	got := onlyTask(t, r)
	if got.Outcome != Complete || got.Provenance != Observed {
		t.Errorf("outcome/provenance = %q/%q, want complete/observed", got.Outcome, got.Provenance)
	}
	if !got.Outcome.Success() {
		t.Error("a completed turn must report Success()")
	}
	if got.EndedAt != "2026-06-10T16:00:10.915Z" || got.DurationMS != 1382 {
		t.Errorf("ended/duration = %q/%d, want the observed terminal's own values", got.EndedAt, got.DurationMS)
	}
}

// FIXTURE 2 — explicit abort: a typed non-success terminal, carrying the reason the
// producer already records.
func TestFold_ExplicitAbort(t *testing.T) {
	r := foldLines(t, false, started("2026-06-16T19:36:00.000Z", "A"), aborted("2026-06-16T19:59:08.325Z", "A"))
	got := onlyTask(t, r)
	if got.Outcome != Aborted || got.Provenance != Observed {
		t.Errorf("outcome/provenance = %q/%q, want aborted/observed", got.Outcome, got.Provenance)
	}
	if got.Outcome.Success() {
		t.Error("an aborted turn must NOT report Success()")
	}
	if got.Reason != "interrupted" {
		t.Errorf("reason = %q, want interrupted", got.Reason)
	}
}

// FIXTURE 3 — THE HEADLINE DEFECT (#4785): a mid-session start with no terminal,
// followed by a later turn that completes normally. The abandoned turn must be closed
// with a typed NON-success outcome at an evidence-backed boundary (the succeeding
// start's own timestamp), and must never be fabricated into a task_complete. The
// later turn keeps its own honest completion.
func TestFold_MidSessionMissingTerminalIsSupersededNotFabricated(t *testing.T) {
	r := foldLines(t, false,
		started("2026-06-10T16:00:09.000Z", "A"), noise, // A does real work, then never terminates
		started("2026-06-10T16:05:00.000Z", "B"), complete("2026-06-10T16:05:30.000Z", "B"))

	if len(r.Tasks) != 2 {
		t.Fatalf("tasks = %d, want 2 (both starts survive the fold): %+v", len(r.Tasks), r.Tasks)
	}
	a, b := r.Tasks[0], r.Tasks[1]

	if a.Outcome != Superseded {
		t.Errorf("abandoned turn outcome = %q, want superseded", a.Outcome)
	}
	if a.Outcome.Success() {
		t.Error("the abandoned turn must never be counted a success — that is the fabrication this fences")
	}
	if a.Provenance != Synthesized {
		t.Errorf("abandoned turn provenance = %q, want synthesized (fak inferred this end)", a.Provenance)
	}
	// The boundary is evidence-backed: B's start is where A demonstrably stopped
	// owning the session, so no later token/tool delta is attributable to A.
	if a.EndedAt != "2026-06-10T16:05:00.000Z" {
		t.Errorf("boundary = %q, want the succeeding start's timestamp", a.EndedAt)
	}
	if a.Reason != "superseded_by_turn:B" {
		t.Errorf("reason = %q, want it to name the superseding turn", a.Reason)
	}
	// The later turn is unaffected — reconciliation must not corrupt honest rows.
	if b.Outcome != Complete || b.Provenance != Observed {
		t.Errorf("later turn = %q/%q, want complete/observed", b.Outcome, b.Provenance)
	}
	if len(r.Unclassified()) != 0 {
		t.Errorf("unclassified = %d, want 0 (every start carries a typed terminal)", len(r.Unclassified()))
	}
}

// FIXTURE 4 — process death: the final start never terminates and the rollout is
// STALE. It is typed distinctly from a live turn, and no end is fabricated beyond the
// typed marker.
func TestFold_ProcessDeathOnStaleFinalStart(t *testing.T) {
	r := foldLines(t, false, started("2026-06-10T16:00:09.000Z", "A"), noise)
	got := onlyTask(t, r)
	if got.Outcome != ProcessDeath || got.Provenance != Synthesized {
		t.Errorf("outcome/provenance = %q/%q, want process_death/synthesized", got.Outcome, got.Provenance)
	}
	if got.Outcome.Success() {
		t.Error("process death must not report Success()")
	}
}

// FIXTURE 4b — the SAME unmatched final start on a FRESH rollout is a genuinely live
// turn, not process death. Freshness is the whole discriminator, and a live turn has
// no EndedAt — inventing one would be the fabrication.
func TestFold_FreshFinalStartIsLiveNotProcessDeath(t *testing.T) {
	r := foldLines(t, true, started("2026-06-10T16:00:09.000Z", "A"), noise)
	got := onlyTask(t, r)
	if got.Outcome != Live {
		t.Errorf("outcome = %q, want live (fresh rollout)", got.Outcome)
	}
	if got.EndedAt != "" {
		t.Errorf("live turn EndedAt = %q, want empty — a running turn has not ended", got.EndedAt)
	}
}

// FIXTURE 5 — orphan terminal: a terminal whose turn_id was never started here.
// Reported separately, and it must not invent a task row.
func TestFold_OrphanTerminal(t *testing.T) {
	r := foldLines(t, false, complete("2026-06-10T16:00:10.915Z", "GHOST"))
	if len(r.Tasks) != 0 {
		t.Errorf("tasks = %d, want 0 (an orphan terminal must not fabricate a start)", len(r.Tasks))
	}
	if len(r.Orphans) != 1 || r.Orphans[0].TurnID != "GHOST" {
		t.Errorf("orphans = %+v, want exactly the GHOST terminal", r.Orphans)
	}
}

// FIXTURE 6 — duplicate terminal: a second OBSERVED terminal for an already-observed
// turn is reported in its own class, and does not overwrite the first.
func TestFold_DuplicateTerminal(t *testing.T) {
	r := foldLines(t, false,
		started("2026-06-10T16:00:09.000Z", "A"),
		complete("2026-06-10T16:00:10.915Z", "A"),
		complete("2026-06-10T16:00:11.915Z", "A"))
	got := onlyTask(t, r)
	if got.EndedAt != "2026-06-10T16:00:10.915Z" {
		t.Errorf("ended = %q, want the FIRST observed terminal (a duplicate must not overwrite)", got.EndedAt)
	}
	if len(r.MultiplyTerminated) != 1 || r.MultiplyTerminated[0] != "A" {
		t.Errorf("multiply_terminated = %v, want [A]", r.MultiplyTerminated)
	}
}

// The three integrity classes are reported SEPARATELY, not merged into one count —
// the acceptance criterion that a consumer can tell them apart.
func TestFold_ReusedTurnIDIsItsOwnClass(t *testing.T) {
	r := foldLines(t, false,
		started("2026-06-10T16:00:09.000Z", "A"), complete("2026-06-10T16:00:10.000Z", "A"),
		started("2026-06-10T16:01:00.000Z", "A"))
	if len(r.Reused) != 1 || r.Reused[0] != "A" {
		t.Errorf("reused = %v, want [A]", r.Reused)
	}
	if len(r.MultiplyTerminated) != 0 || len(r.Orphans) != 0 {
		t.Errorf("a reused id must not also count as terminated/orphan: %+v", r)
	}
}

// A late REAL terminal for a turn fak had synthesized closed must repair the row:
// observed evidence outranks an inference, and this is not a duplicate.
func TestFold_ObservedTerminalRepairsSynthesizedClose(t *testing.T) {
	r := foldLines(t, false,
		started("2026-06-10T16:00:09.000Z", "A"),
		started("2026-06-10T16:05:00.000Z", "B"), // closes A as superseded (synthesized)
		aborted("2026-06-10T16:06:00.000Z", "A")) // …but A really did abort
	a := r.Tasks[0]
	if a.Outcome != Aborted || a.Provenance != Observed {
		t.Errorf("A = %q/%q, want aborted/observed (real evidence must replace the guess)", a.Outcome, a.Provenance)
	}
	if len(r.MultiplyTerminated) != 0 {
		t.Errorf("repairing a synthesized close is not a duplicate terminal: %v", r.MultiplyTerminated)
	}
}

// Legacy rollouts emit turn_aborted with NO turn_id (corpus: 2025-09-16 files). The
// unkeyed terminal binds to the open turn rather than being dropped on the floor.
func TestFold_LegacyUnkeyedAbortBindsToActiveTurn(t *testing.T) {
	line := `{"timestamp":"2025-09-16T01:57:29.280Z","type":"event_msg","payload":{"type":"turn_aborted","reason":"interrupted"}}`
	r := foldLines(t, false, started("2025-09-16T01:50:00.000Z", "A"), line)
	got := onlyTask(t, r)
	if got.Outcome != Aborted || got.Provenance != Observed {
		t.Errorf("outcome/provenance = %q/%q, want aborted/observed", got.Outcome, got.Provenance)
	}
	if len(r.Orphans) != 0 {
		t.Errorf("the unkeyed abort must bind to the open turn, not orphan: %+v", r.Orphans)
	}
}

// A torn final line (a crashed writer mid-record) must not fail the read — the
// truncated tail IS the process-death evidence the fold has to survive to classify.
func TestParseRollout_SurvivesTornFinalLine(t *testing.T) {
	body := started("2026-06-10T16:00:09.000Z", "A") + "\n" + `{"timestamp":"2026-06-10T16:0`
	ev, err := ParseRollout(strings.NewReader(body))
	if err != nil {
		t.Fatalf("ParseRollout on a torn tail: %v", err)
	}
	if len(ev) != 1 || ev[0].Kind != KindStarted {
		t.Fatalf("events = %+v, want just the intact start", ev)
	}
	if got := Fold(ev, false); onlyTask(t, got).Outcome != ProcessDeath {
		t.Error("a torn tail with an open start is process death")
	}
}

func TestFold_CompletedWithTrailingEmptyAbort(t *testing.T) {
	ev := []Event{
		{Kind: KindStarted, TurnID: "turn-1", Timestamp: "2026-09-03T01:00:00Z"},
		{Kind: KindComplete, TurnID: "turn-1", Timestamp: "2026-09-03T01:01:00Z", DurationMS: 60000},
		{Kind: KindStarted, TurnID: "turn-2", Timestamp: "2026-09-03T01:01:01Z"},
		{Kind: KindAborted, TurnID: "turn-2", Timestamp: "2026-09-03T01:01:02Z", DurationMS: 50, Reason: "interrupted"},
	}
	r := Fold(ev, false)
	if len(r.Tasks) != 2 {
		t.Fatalf("tasks count = %d, want 2", len(r.Tasks))
	}
	if r.Tasks[0].Outcome != Complete {
		t.Errorf("turn-1 outcome = %q, want complete", r.Tasks[0].Outcome)
	}
	if r.Tasks[1].Outcome != Aborted || !r.Tasks[1].TrailingEmptyAbort {
		t.Errorf("turn-2 = %+v, want aborted with trailing empty abort", r.Tasks[1])
	}
	if !r.SubstantiveCompleted {
		t.Error("substantive_completed must be true")
	}
	if !r.CompletedWithTrailingAbort {
		t.Error("completed_with_trailing_abort must be true")
	}
}

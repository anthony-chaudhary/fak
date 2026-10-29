package supervisoragent

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// recordingVerbs is the closed admission seam bound by the loop tests: it
// counts every call so a test can prove exactly which verbs ran (or that
// ZERO ran). Admit/Arbitrate/EmitEscalation each refuse unless told to, so a
// test that forgets to grant a verb fails loudly instead of silently passing.
type recordingVerbs struct {
	arbitrateCalls int
	admitCalls     int
	emitCalls      int
	lastAdmit      [3]string // issue, lane, supersedes
	granted        bool
}

func (v *recordingVerbs) Arbitrate(lane string, tree []string) (Lease, error) {
	v.arbitrateCalls++
	if !v.granted {
		return Lease{}, errors.New("arbiter refused")
	}
	return Lease{Lane: lane, Kind: "cluster", Tree: tree}, nil
}

func (v *recordingVerbs) Admit(issue, lane, supersedes string) (AdmitReceipt, error) {
	v.admitCalls++
	if !v.granted {
		return AdmitReceipt{}, errors.New("admit path refused")
	}
	v.lastAdmit = [3]string{issue, lane, supersedes}
	return AdmitReceipt{RunID: "RID-NEW", Issue: issue, Lane: lane, Supersedes: supersedes}, nil
}

func (v *recordingVerbs) EmitEscalation(head Escalation) (Escalation, error) {
	v.emitCalls++
	head.ID = "ESC-1"
	return head, nil
}

func (v *recordingVerbs) totalCalls() int { return v.arbitrateCalls + v.admitCalls + v.emitCalls }

var seatNow = time.Date(2026, 10, 28, 9, 0, 0, 0, time.UTC)

// greenSources is the fully-witnessed green baseline as Sources (the shape
// RunWake takes): one healthy worker, moving liveness, no open packets, no
// leases.
func greenSources() Sources {
	return Sources{
		Liveness:    Seen(Liveness{RunID: "RID-1", Class: "moving", Commits: 3}),
		Workers:     Seen([]WorkerCensus{{RunID: "RID-1", Issue: "4480", Lane: "supervisoragent", Class: "healthy"}}),
		Escalations: Seen([]Escalation{}),
		Leases:      Seen([]leaseref.ArbiterLease{}),
	}
}

// deadWorkerSources is greenSources with the worker verdict flipped dead —
// the typed witness whose strongest rung proposes replace.
func deadWorkerSources() Sources {
	src := greenSources()
	w := src.Workers.Value
	w[0].Class = "dead"
	src.Workers = Seen(w)
	return src
}

// TestOutcomeVocabularyClosed pins the closed outcome token set: exactly these
// six tokens, distinct, non-empty, with their wire spellings. Adding or
// renaming a token is a deliberate, test-breaking act — like widening the
// reason vocabulary.
func TestOutcomeVocabularyClosed(t *testing.T) {
	wantTokens := map[Outcome]string{
		OutcomeLowered:     "lowered",
		OutcomeRefused:     "refused",
		OutcomeEscalated:   "escalated",
		OutcomeHold:        "hold",
		OutcomeDryRun:      "dry_run",
		OutcomeVerbRefused: "verb_refused",
	}
	if len(wantTokens) != 6 {
		t.Fatalf("vocabulary has %d tokens; want 6", len(wantTokens))
	}
	for o, tok := range wantTokens {
		if string(o) == "" {
			t.Error("empty outcome token in the closed vocabulary")
		}
		if string(o) != tok {
			t.Errorf("outcome token spells %q; want %q", string(o), tok)
		}
	}
}

// TestSupervisorSeatLoop is the S3 loop-level witness: the full
// wake -> assemble -> decide -> lower -> journal cycle over fake witnesses,
// covering the four mandated shapes — green fleet holds, a missing witness
// escalates without lowering any widening verb, an envelope-gated proposal
// lowers through the existing admission-verb seam, and journal records
// survive replay — plus the dry-run zero-admission-call proof and the
// never-retry refusal record.
func TestSupervisorSeatLoop(t *testing.T) {
	full := Envelope{Widths: fullWidths(t)}
	zero := Envelope{Widths: map[ActionKind]int{}}

	t.Run("green_fleet_holds", func(t *testing.T) {
		j := NewMemoryJournal()
		v := &recordingVerbs{granted: true}
		seat := &Seat{Journal: j, Verbs: v, Envelope: full, Mode: ModeEnforce}

		rec, err := seat.RunWake("wake://fleet/cmd", greenSources(), seatNow, false)
		if err != nil {
			t.Fatalf("RunWake: %v", err)
		}
		if rec.ProposedVerb != ActionHold || rec.ReasonToken != ReasonHoldGreen {
			t.Errorf("green fleet proposed (%v, %v); want (hold, hold_green)", rec.ProposedVerb, rec.ReasonToken)
		}
		if rec.Outcome != OutcomeHold {
			t.Errorf("outcome = %v; want hold", rec.Outcome)
		}
		if v.totalCalls() != 0 {
			t.Errorf("hold made %d admission calls; want zero", v.totalCalls())
		}
		if rec.ArtifactRef != "" {
			t.Errorf("hold left artifact %q; want none", rec.ArtifactRef)
		}
		if rec.Seq != 1 || rec.Topic != "wake://fleet/cmd" {
			t.Errorf("record header = seq %d topic %q; want seq 1 addressed", rec.Seq, rec.Topic)
		}
	})

	t.Run("missing_witness_escalates_no_lower", func(t *testing.T) {
		j := NewMemoryJournal()
		v := &recordingVerbs{granted: true}
		seat := &Seat{Journal: j, Verbs: v, Envelope: full, Mode: ModeEnforce}

		src := greenSources()
		src.Workers = Absent[[]WorkerCensus]() // the census could not be read
		rec, err := seat.RunWake("wake://fleet/cmd", src, seatNow, false)
		if err != nil {
			t.Fatalf("RunWake: %v", err)
		}
		if rec.ProposedVerb != ActionEscalate || rec.ReasonToken != ReasonWitnessAbsent {
			t.Errorf("absent witness proposed (%v, %v); want (escalate, witness_absent)", rec.ProposedVerb, rec.ReasonToken)
		}
		if rec.Outcome != OutcomeEscalated {
			t.Errorf("outcome = %v; want escalated", rec.Outcome)
		}
		// No widening verb lowered: only the fail-closed packet emit ran.
		if v.admitCalls != 0 || v.arbitrateCalls != 0 {
			t.Errorf("absent witness lowered a widening verb (admit=%d arbitrate=%d); want none", v.admitCalls, v.arbitrateCalls)
		}
		if v.emitCalls != 1 {
			t.Errorf("escalation emit called %d times; want exactly 1", v.emitCalls)
		}
		if rec.ArtifactRef != "escalation:ESC-1" {
			t.Errorf("artifact = %q; want escalation:ESC-1", rec.ArtifactRef)
		}
	})

	t.Run("envelope_gated_proposal_lowers_through_seam", func(t *testing.T) {
		j := NewMemoryJournal()
		v := &recordingVerbs{granted: true}
		seat := &Seat{Journal: j, Verbs: v, Envelope: full, Mode: ModeEnforce}

		rec, err := seat.RunWake("wake://fleet/cmd", deadWorkerSources(), seatNow, false)
		if err != nil {
			t.Fatalf("RunWake: %v", err)
		}
		if rec.ProposedVerb != ActionReplace || rec.ReasonToken != ReasonReplaceStaleLease {
			t.Errorf("dead worker proposed (%v, %v); want (replace, replace_stale_lease)", rec.ProposedVerb, rec.ReasonToken)
		}
		if rec.EnvelopeVerdict.Auth != AuthUnattended {
			t.Errorf("verdict auth = %v; want unattended at full width", rec.EnvelopeVerdict.Auth)
		}
		if rec.Outcome != OutcomeLowered {
			t.Errorf("outcome = %v; want lowered", rec.Outcome)
		}
		// The proposal lowered through the EXISTING admission-verb seam:
		// exactly one admit, naming the superseded run.
		if v.admitCalls != 1 || v.arbitrateCalls != 0 || v.emitCalls != 0 {
			t.Errorf("admission calls = admit %d / arbitrate %d / emit %d; want 1/0/0", v.admitCalls, v.arbitrateCalls, v.emitCalls)
		}
		want := [3]string{"4480", "supervisoragent", "RID-1"}
		if v.lastAdmit != want {
			t.Errorf("admit args = %v; want %v", v.lastAdmit, want)
		}
		if rec.ArtifactRef != "admit:RID-NEW#4480" {
			t.Errorf("artifact = %q; want admit:RID-NEW#4480", rec.ArtifactRef)
		}
	})

	t.Run("enforce_sub_envelope_refuses_without_lowering", func(t *testing.T) {
		j := NewMemoryJournal()
		v := &recordingVerbs{granted: true}
		seat := &Seat{Journal: j, Verbs: v, Envelope: zero, Mode: ModeEnforce}

		rec, err := seat.RunWake("wake://fleet/cmd", deadWorkerSources(), seatNow, false)
		if err != nil {
			t.Fatalf("RunWake: %v", err)
		}
		if rec.ProposedVerb != ActionReplace || rec.ReasonToken != ReasonEnvelopeRefused {
			t.Errorf("sub-envelope proposal = (%v, %v); want (replace, envelope_refused)", rec.ProposedVerb, rec.ReasonToken)
		}
		if rec.Outcome != OutcomeRefused {
			t.Errorf("outcome = %v; want refused", rec.Outcome)
		}
		if v.totalCalls() != 0 {
			t.Errorf("refused proposal made %d admission calls; want zero", v.totalCalls())
		}
	})

	t.Run("collapsed_envelope_escalates_instead_of_replacing", func(t *testing.T) {
		j := NewMemoryJournal()
		v := &recordingVerbs{granted: true}
		collapsed := Envelope{Widths: fullWidths(t), WitnessRefused: true}
		seat := &Seat{Journal: j, Verbs: v, Envelope: collapsed, Mode: ModeEnforce}

		rec, err := seat.RunWake("wake://fleet/cmd", deadWorkerSources(), seatNow, false)
		if err != nil {
			t.Fatalf("RunWake: %v", err)
		}
		if rec.ProposedVerb != ActionEscalate || rec.Outcome != OutcomeEscalated {
			t.Errorf("collapsed envelope gave (%v, %v); want (escalate, escalated)", rec.ProposedVerb, rec.Outcome)
		}
		if v.admitCalls != 0 {
			t.Errorf("collapsed envelope admitted a replace (%d admits); want none", v.admitCalls)
		}
	})

	t.Run("dry_run_makes_zero_admission_calls", func(t *testing.T) {
		j := NewMemoryJournal()
		v := &recordingVerbs{granted: true}
		seat := &Seat{Journal: j, Verbs: v, Envelope: full, Mode: ModeEnforce}

		// Live mode would lower this replace through Admit; dry-run must not.
		rec, err := seat.RunWake("wake://fleet/cmd", deadWorkerSources(), seatNow, true)
		if err != nil {
			t.Fatalf("RunWake(dry): %v", err)
		}
		if rec.ProposedVerb != ActionReplace || rec.ReasonToken != ReasonReplaceStaleLease {
			t.Errorf("dry-run still decides: got (%v, %v); want (replace, replace_stale_lease)", rec.ProposedVerb, rec.ReasonToken)
		}
		if rec.EnvelopeVerdict.Auth != AuthUnattended {
			t.Errorf("dry-run verdict auth = %v; want unattended (the live verdict rides the record)", rec.EnvelopeVerdict.Auth)
		}
		if rec.Outcome != OutcomeDryRun {
			t.Errorf("outcome = %v; want dry_run", rec.Outcome)
		}
		if v.totalCalls() != 0 {
			t.Errorf("dry-run made %d admission calls; want ZERO", v.totalCalls())
		}
		if rec.ArtifactRef != "" {
			t.Errorf("dry-run left artifact %q; want none", rec.ArtifactRef)
		}
		// Exactly one record despite the would-be lowering.
		all, _ := j.Replay(0)
		if len(all) != 1 {
			t.Fatalf("journal has %d records; want exactly one per wake", len(all))
		}
	})

	t.Run("dry_run_needs_no_bound_verbs", func(t *testing.T) {
		// Nil verbs + dry-run is the structural zero-side-effect shape: the
		// preview path cannot touch a seam that does not exist.
		seat := &Seat{Journal: NewMemoryJournal(), Verbs: nil, Envelope: full, Mode: ModeEnforce}
		if _, err := seat.RunWake("wake://fleet/cmd", deadWorkerSources(), seatNow, true); err != nil {
			t.Fatalf("dry-run with unbound verbs: %v", err)
		}
		// ...while the same seat live must fail closed before touching anything.
		if _, err := seat.RunWake("wake://fleet/cmd", deadWorkerSources(), seatNow, false); !errors.Is(err, ErrNoAdmissionVerbs) {
			t.Errorf("live wake with unbound verbs = %v; want ErrNoAdmissionVerbs", err)
		}
	})

	t.Run("verb_refusal_is_recorded_not_retried", func(t *testing.T) {
		j := NewMemoryJournal()
		v := &recordingVerbs{granted: false} // the admit path refuses
		seat := &Seat{Journal: j, Verbs: v, Envelope: full, Mode: ModeEnforce}

		rec, err := seat.RunWake("wake://fleet/cmd", deadWorkerSources(), seatNow, false)
		if err != nil {
			t.Fatalf("RunWake: %v", err)
		}
		if rec.Outcome != OutcomeVerbRefused {
			t.Errorf("outcome = %v; want verb_refused", rec.Outcome)
		}
		if v.admitCalls != 1 {
			t.Errorf("admit called %d times; want exactly 1 (never retried)", v.admitCalls)
		}
		if rec.ArtifactRef != "" {
			t.Errorf("refused verb left artifact %q; want none", rec.ArtifactRef)
		}
	})

	t.Run("journal_records_survive_replay", func(t *testing.T) {
		j := NewMemoryJournal()
		v := &recordingVerbs{granted: true}
		seat := &Seat{Journal: j, Verbs: v, Envelope: full, Mode: ModeEnforce}

		if _, err := seat.RunWake("wake://fleet/cmd", greenSources(), seatNow, false); err != nil {
			t.Fatalf("wake 1: %v", err)
		}
		if _, err := seat.RunWake("wake://fleet/cmd", deadWorkerSources(), seatNow.Add(time.Minute), false); err != nil {
			t.Fatalf("wake 2: %v", err)
		}
		if _, err := seat.RunWake("wake://fleet/cmd", greenSources(), seatNow.Add(2*time.Minute), true); err != nil {
			t.Fatalf("wake 3 (dry): %v", err)
		}

		all, err := j.Replay(0)
		if err != nil {
			t.Fatalf("replay: %v", err)
		}
		if len(all) != 3 {
			t.Fatalf("replay returned %d records; want one per wake (3)", len(all))
		}
		wantOutcomes := []Outcome{OutcomeHold, OutcomeLowered, OutcomeDryRun}
		for i, r := range all {
			if r.Seq != uint64(i+1) {
				t.Errorf("record %d has Seq %d; want %d", i, r.Seq, i+1)
			}
			if r.Outcome != wantOutcomes[i] {
				t.Errorf("record %d outcome = %v; want %v", i, r.Outcome, wantOutcomes[i])
			}
		}

		// Replay is stable across calls and resumable from a cursor.
		again, _ := j.Replay(0)
		if !reflect.DeepEqual(all, again) {
			t.Error("Replay(0) is not stable across calls")
		}
		tail, _ := j.Replay(3)
		if len(tail) != 1 || tail[0].Seq != 3 {
			t.Errorf("Replay(3) = %+v; want the third record only", tail)
		}
		past, _ := j.Replay(4)
		if past == nil || len(past) != 0 {
			t.Errorf("Replay past the tail = %v; want empty non-nil", past)
		}
		last, ok := j.Last()
		if !ok || last.Seq != 3 {
			t.Errorf("Last() = %+v, %v; want the third record", last, ok)
		}
	})
}

// TestFileJournalGoldenBehavior pins the durable backend against the golden
// behavior MemoryJournal owns: seq monotonic from 1, replay stable and
// resumable from a cursor, last after append — PLUS the restart property the
// file form exists for: a NEW instance on the same file continues the
// sequence, and a malformed line is a hard error, never a guess.
func TestFileJournalGoldenBehavior(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "decisions.jsonl")

	j, err := NewFileJournal(path)
	if err != nil {
		t.Fatalf("NewFileJournal: %v", err)
	}
	if _, ok := j.Last(); ok {
		t.Fatal("Last() on a fresh journal must report false")
	}

	digest := DigestInput(greenInput())
	for i := 0; i < 3; i++ {
		err := j.Append(DecisionRecord{
			Topic:        "wake://fleet/cmd",
			InputDigest:  digest,
			ProposedVerb: ActionHold,
			ReasonToken:  ReasonHoldGreen,
			Outcome:      OutcomeHold,
			DecidedAt:    seatNow.Add(time.Duration(i) * time.Minute),
		})
		if err != nil {
			t.Fatalf("append %d: %v", i, err)
		}
	}

	all, err := j.Replay(0)
	if err != nil {
		t.Fatalf("replay: %v", err)
	}
	if len(all) != 3 {
		t.Fatalf("Replay(0) returned %d records; want 3", len(all))
	}
	for i, r := range all {
		if r.Seq != uint64(i+1) {
			t.Errorf("record %d has Seq %d; want %d", i, r.Seq, i+1)
		}
	}
	tail, _ := j.Replay(3)
	if len(tail) != 1 || tail[0].Seq != 3 {
		t.Errorf("Replay(3) = %+v; want the tail from Seq 3", tail)
	}
	last, ok := j.Last()
	if !ok || last.Seq != 3 {
		t.Errorf("Last() = %+v, %v; want the third record", last, ok)
	}

	// Restart: a fresh instance on the same file resumes the series.
	j2, err := NewFileJournal(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	if err := j2.Append(DecisionRecord{
		Topic: "wake://fleet/cmd", InputDigest: digest, ProposedVerb: ActionHold,
		ReasonToken: ReasonHoldGreen, Outcome: OutcomeHold, DecidedAt: seatNow.Add(3 * time.Minute),
	}); err != nil {
		t.Fatalf("append after restart: %v", err)
	}
	resumed, err := j2.Replay(0)
	if err != nil {
		t.Fatalf("replay after restart: %v", err)
	}
	if len(resumed) != 4 || resumed[3].Seq != 4 {
		t.Errorf("restart did not resume the sequence: %d records, last Seq %d; want 4 records ending at Seq 4", len(resumed), resumed[len(resumed)-1].Seq)
	}

	// A torn/malformed line is a hard error, never guessed over.
	corrupt := path + ".corrupt"
	if err := os.WriteFile(corrupt, []byte("{not json\n"), 0o644); err != nil {
		t.Fatalf("write corrupt journal: %v", err)
	}
	jc, _ := NewFileJournal(corrupt)
	if _, err := jc.Replay(0); err == nil {
		t.Error("Replay over a malformed line must fail closed")
	}
	if err := jc.Append(DecisionRecord{Topic: "x"}); err == nil {
		t.Error("Append over a malformed journal must fail closed")
	}
}

// greenInput is the SupervisorInput twin of greenSources (for digests).
func greenInput() SupervisorInput {
	return SupervisorInput{
		Liveness:    Seen(Liveness{RunID: "RID-1", Class: "moving", Commits: 3}),
		Workers:     Seen([]WorkerVerdict{{RunID: "RID-1", Issue: "4480", Lane: "supervisoragent", State: WorkerHealthy}}),
		Escalations: Seen([]Escalation{}),
		Leases:      Seen([]Lease{}),
	}
}

package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/supervisoragent"
)

// TestSupervisorSeatLoop is the S3 loop-level witness at the dispatch
// touchpoint (#13464): fake typed witnesses drive one dry-run wake each
// through the real verb handler -> Seat.RunWake -> FileJournal cycle, proving
// the four mandated shapes end to end — green fleet holds, a missing witness
// escalates with no lowering, an envelope-gated proposal carries its verdict
// through the record (the live lower stage binds the same admission-verb seam
// the package-level TestSupervisorSeatLoop exercises with a recording double),
// and journal records survive replay across journal instances.
func TestSupervisorSeatLoop(t *testing.T) {
	dir := t.TempDir()
	journalPath := filepath.Join(dir, "decisions.jsonl")

	runWake := func(t *testing.T, payload string) {
		t.Helper()
		var errBuf bytes.Buffer
		code := runSupervisorSeat([]string{
			"wake",
			"--payload", "-",
			"--journal", journalPath,
		}, strings.NewReader(payload), io.Discard, &errBuf)
		if code != 0 {
			t.Fatalf("wake exit %d; stderr: %s", code, errBuf.String())
		}
	}

	replay := func(t *testing.T) []supervisoragent.DecisionRecord {
		t.Helper()
		j, err := supervisoragent.NewFileJournal(journalPath)
		if err != nil {
			t.Fatalf("open journal: %v", err)
		}
		recs, err := j.Replay(0)
		if err != nil {
			t.Fatalf("replay: %v", err)
		}
		return recs
	}

	const fullEnvelope = `{"widths":{"replace":3,"replan":1,"spawn":2,"widen":4},"witness_refused":false}`
	envPath := filepath.Join(dir, "envelope.json")
	if err := os.WriteFile(envPath, []byte(fullEnvelope), 0o644); err != nil {
		t.Fatal(err)
	}

	t.Run("green_fleet_holds", func(t *testing.T) {
		payload := `{
			"topic": "wake://fleet/cmd",
			"liveness": {"present": true, "value": {"run_id": "RID-1", "class": "moving", "commits": 3}},
			"workers": {"present": true, "value": [{"run_id": "RID-1", "issue": "4480", "lane": "supervisoragent", "class": "healthy"}]},
			"escalations": {"present": true, "value": []},
			"leases": {"present": true, "value": []}
		}`
		runWake(t, payload)
		recs := replay(t)
		if len(recs) != 1 {
			t.Fatalf("journal has %d records; want 1", len(recs))
		}
		r := recs[0]
		if r.ProposedVerb != supervisoragent.ActionHold || r.ReasonToken != supervisoragent.ReasonHoldGreen {
			t.Errorf("green fleet recorded (%v, %v); want (hold, hold_green)", r.ProposedVerb, r.ReasonToken)
		}
		if r.Outcome != supervisoragent.OutcomeDryRun {
			t.Errorf("outcome = %v; want dry_run (S3 lands the preview cycle)", r.Outcome)
		}
		if r.ArtifactRef != "" {
			t.Errorf("dry-run hold left artifact %q; want none", r.ArtifactRef)
		}
	})

	t.Run("missing_witness_escalates_no_lower", func(t *testing.T) {
		// The census surface could not be read: the workers key is absent, so
		// the witness stays ABSENT — never a silently-empty present list.
		payload := `{
			"topic": "wake://fleet/cmd",
			"liveness": {"present": true, "value": {"run_id": "RID-1", "class": "moving", "commits": 3}},
			"escalations": {"present": true, "value": []},
			"leases": {"present": true, "value": []}
		}`
		runWake(t, payload)
		recs := replay(t)
		if len(recs) != 2 {
			t.Fatalf("journal has %d records; want 2", len(recs))
		}
		r := recs[1]
		if r.ProposedVerb != supervisoragent.ActionEscalate || r.ReasonToken != supervisoragent.ReasonWitnessAbsent {
			t.Errorf("absent witness recorded (%v, %v); want (escalate, witness_absent)", r.ProposedVerb, r.ReasonToken)
		}
		if r.Outcome != supervisoragent.OutcomeDryRun {
			t.Errorf("outcome = %v; want dry_run (the packet emit is the live-stage verb; nothing ran here)", r.Outcome)
		}
		if r.ArtifactRef != "" {
			t.Errorf("no lowering happened; artifact must be empty, got %q", r.ArtifactRef)
		}
	})

	t.Run("envelope_gated_proposal_carries_its_verdict", func(t *testing.T) {
		// A dead worker proposes replace. With the earned envelope bound, the
		// gate says unattended — the exact verdict the live lower stage would
		// consume to lower through the dispatch-admit seam.
		payload := `{
			"topic": "wake://fleet/cmd",
			"liveness": {"present": true, "value": {"run_id": "RID-1", "class": "moving", "commits": 3}},
			"workers": {"present": true, "value": [{"run_id": "RID-1", "issue": "4480", "lane": "supervisoragent", "class": "dead"}]},
			"escalations": {"present": true, "value": []},
			"leases": {"present": true, "value": []}
		}`
		var out, errBuf bytes.Buffer
		code := runSupervisorSeat([]string{
			"wake", "--payload", "-", "--journal", journalPath, "--envelope", envPath,
		}, strings.NewReader(payload), &out, &errBuf)
		if code != 0 {
			t.Fatalf("wake exit %d; stderr: %s", code, errBuf.String())
		}
		recs := replay(t)
		if len(recs) != 3 {
			t.Fatalf("journal has %d records; want 3", len(recs))
		}
		r := recs[2]
		if r.ProposedVerb != supervisoragent.ActionReplace || r.ReasonToken != supervisoragent.ReasonReplaceStaleLease {
			t.Errorf("dead worker recorded (%v, %v); want (replace, replace_stale_lease)", r.ProposedVerb, r.ReasonToken)
		}
		if r.EnvelopeVerdict.Auth != supervisoragent.AuthUnattended {
			t.Errorf("earned-envelope verdict = %v; want unattended", r.EnvelopeVerdict.Auth)
		}
		if r.Outcome != supervisoragent.OutcomeDryRun {
			t.Errorf("outcome = %v; want dry_run", r.Outcome)
		}

		// The same wake WITHOUT the earned envelope: zero widths refuse the
		// widening proposal under enforce — proposed for confirmation, nothing
		// runs, and the refusal is journaled. The soak switch comes from the
		// envelope's own environment variable; unset soaks (warn), so pin
		// enforce here to exercise the refusal.
		t.Setenv(supervisoragent.EnvelopeModeEnv, "enforce")
		var out2, err2 bytes.Buffer
		code = runSupervisorSeat([]string{
			"wake", "--payload", "-", "--journal", journalPath,
		}, strings.NewReader(payload), &out2, &err2)
		if code != 0 {
			t.Fatalf("zero-width wake exit %d; stderr: %s", code, err2.String())
		}
		recs = replay(t)
		r = recs[3]
		if r.ReasonToken != supervisoragent.ReasonEnvelopeRefused {
			t.Errorf("zero-width verdict reason = %v; want envelope_refused", r.ReasonToken)
		}
		if r.EnvelopeVerdict.Auth != supervisoragent.AuthConfirm {
			t.Errorf("zero-width verdict auth = %v; want confirm (proposed, nothing ran)", r.EnvelopeVerdict.Auth)
		}
	})

	t.Run("journal_records_survive_replay", func(t *testing.T) {
		recs := replay(t)
		if len(recs) != 4 {
			t.Fatalf("journal has %d records; want 4", len(recs))
		}
		for i, r := range recs {
			if r.Seq != uint64(i+1) {
				t.Errorf("record %d has Seq %d; want %d (monotonic from 1)", i, r.Seq, i+1)
			}
			if r.Topic != "wake://fleet/cmd" {
				t.Errorf("record %d topic = %q; want the wake address", i, r.Topic)
			}
			if len(r.InputDigest) != 64 {
				t.Errorf("record %d input digest = %q; want a 64-hex-char digest, never contents", i, r.InputDigest)
			}
		}
		// Cursor resume from a fresh journal instance on the same file.
		j, err := supervisoragent.NewFileJournal(journalPath)
		if err != nil {
			t.Fatalf("reopen: %v", err)
		}
		tail, err := j.Replay(3)
		if err != nil {
			t.Fatalf("replay cursor: %v", err)
		}
		if len(tail) != 2 || tail[0].Seq != 3 || tail[1].Seq != 4 {
			t.Errorf("Replay(3) = %+v; want the tail from Seq 3", tail)
		}
		last, ok := j.Last()
		if !ok || last.Seq != 4 {
			t.Errorf("Last() = %+v, %v; want the fourth record", last, ok)
		}
	})

	t.Run("apply_is_refused_with_a_named_token", func(t *testing.T) {
		var out, errBuf bytes.Buffer
		code := runSupervisorSeat([]string{"wake", "--payload", "-", "--apply"}, strings.NewReader("{}"), &out, &errBuf)
		if code != 2 {
			t.Fatalf("--apply exit %d; want 2 (refused until S4)", code)
		}
		if !strings.Contains(errBuf.String(), "SUPERVISOR_SEAT_LIVE_LOWER_NOT_SHIPPED") {
			t.Errorf("refusal did not name its token: %s", errBuf.String())
		}
	})

	t.Run("unknown_payload_key_is_rejected", func(t *testing.T) {
		// The closed contract cannot acquire a payload field by drift.
		payload := `{"liveness": {"present": true, "value": {}}, "transcript": "hi"}`
		var out, errBuf bytes.Buffer
		code := runSupervisorSeat([]string{"wake", "--payload", "-", "--journal", journalPath}, strings.NewReader(payload), &out, &errBuf)
		if code != 1 {
			t.Fatalf("unknown-key payload exit %d; want 1 (invalid)", code)
		}
		if !strings.Contains(errBuf.String(), "SUPERVISOR_SEAT_PAYLOAD_INVALID") {
			t.Errorf("rejection did not name its token: %s", errBuf.String())
		}
	})

	t.Run("json_output_is_the_journaled_record", func(t *testing.T) {
		payload := `{
			"liveness": {"present": true, "value": {"run_id": "RID-1", "class": "moving"}},
			"workers": {"present": true, "value": []},
			"escalations": {"present": true, "value": []},
			"leases": {"present": true, "value": []}
		}`
		var out, errBuf bytes.Buffer
		code := runSupervisorSeat([]string{"wake", "--payload", "-", "--journal", journalPath, "--json"}, strings.NewReader(payload), &out, &errBuf)
		if code != 0 {
			t.Fatalf("json wake exit %d; stderr: %s", code, errBuf.String())
		}
		var rec supervisoragent.DecisionRecord
		if err := json.Unmarshal(bytes.TrimSpace(out.Bytes()), &rec); err != nil {
			t.Fatalf("stdout is not a DecisionRecord JSON: %v\n%s", err, out.String())
		}
		if rec.Seq != 5 || rec.Outcome != supervisoragent.OutcomeDryRun {
			t.Errorf("json record = seq %d outcome %v; want seq 5 dry_run", rec.Seq, rec.Outcome)
		}
	})
}

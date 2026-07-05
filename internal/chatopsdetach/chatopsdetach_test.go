package chatopsdetach

import (
	"strings"
	"testing"
	"time"
)

// spool is a tiny in-test stand-in for the shell's durable per-nonce Record
// store. The real shell backs this with a JSONL done-ledger (the fakrpc
// `done/<nonce> ⇒ skip` discipline); the kernel only needs the prior row.
type spool map[string]Record

func (s spool) prior(nonce string) Record { return s[nonce] }
func (s spool) commit(d Decision)         { s[d.Record.Nonce] = d.Record }

// deliver runs one command delivery through the kernel against the spool, then
// persists the decision's Record exactly as the shell would.
func (s spool) deliver(cmd Command, adm Admission) Decision {
	d := Decide(cmd, adm, s.prior(cmd.Nonce))
	s.commit(d)
	return d
}

// DoD #2 (idempotency): a command delivered twice under the same nonce dispatches
// exactly ONE run and produces two byte-identical acks — a dropped ack or a
// retried Slack event can never double-dispatch.
func TestDoubleDeliveryDispatchesOnce(t *testing.T) {
	s := spool{}
	cmd := Command{Nonce: "ts-100", Verb: VerbDispatch, Target: "#2265", Channel: "C1", User: "u1"}
	adm := Admission{Admitted: true, RunID: "run-abc", Lane: "chatops"}

	first := s.deliver(cmd, adm)
	if first.Action != Dispatch {
		t.Fatalf("first delivery: action = %s, want DISPATCH", first.Action)
	}
	if first.RunID != "run-abc" {
		t.Fatalf("first delivery: run = %q, want run-abc", first.RunID)
	}

	// The retry commonly re-runs admission and mints a DIFFERENT run id. The
	// kernel must bind the ORIGINAL run from the spool, never the fresh one — the
	// double-delivery guard wins over a stale re-admit.
	second := s.deliver(cmd, Admission{Admitted: true, RunID: "run-XYZ-different", Lane: "other"})
	if second.Action != ReAck {
		t.Fatalf("second delivery: action = %s, want RE_ACK", second.Action)
	}
	if second.RunID != "run-abc" {
		t.Fatalf("second delivery: run = %q, want the original run-abc (no double-dispatch)", second.RunID)
	}
	if second.Ack != first.Ack {
		t.Fatalf("re-ack not identical to first ack:\n first  = %q\n second = %q", first.Ack, second.Ack)
	}

	// A third delivery keeps re-acking the same run: idempotency is stable.
	third := s.deliver(cmd, adm)
	if third.Action != ReAck || third.RunID != "run-abc" || third.Ack != first.Ack {
		t.Fatalf("third delivery not a stable re-ack: %+v", third)
	}
}

// DoD #3 (refusal at admission): when the front door refuses, the kernel routes a
// structured refusal carrying the closed token and starts nothing — a refused
// command cannot silently queue-jump the cap.
func TestRefusalAtAdmissionStartsNothing(t *testing.T) {
	s := spool{}
	cmd := Command{Nonce: "ts-200", Verb: VerbDispatch, Target: "#2265", Channel: "C1"}

	d := s.deliver(cmd, Admission{Admitted: false, Reason: "REFUSE_AT_CAP"})
	if d.Action != Refuse {
		t.Fatalf("action = %s, want REFUSE", d.Action)
	}
	if d.RunID != "" {
		t.Fatalf("a refusal minted a run %q — that is a queue-jump", d.RunID)
	}
	if !strings.Contains(d.Reason, "REFUSE_AT_CAP") {
		t.Fatalf("refusal reason %q does not carry the closed token", d.Reason)
	}
	if !strings.Contains(d.Reason, "no seat taken") {
		t.Fatalf("refusal reason %q does not state that no seat was taken", d.Reason)
	}
	if d.Record.Dispatched() {
		t.Fatalf("a refusal recorded a dispatched run: %+v", d.Record)
	}
}

// A refused nonce is re-admittable: once capacity frees, a later delivery of the
// SAME nonce dispatches honestly through the admission gate rather than being
// frozen on the first "no". The only path to Dispatch is an Admitted verdict, so
// this is not a queue-jump — it is a fair admission.
func TestRefusedCommandIsReadmittedWhenCapacityFrees(t *testing.T) {
	s := spool{}
	cmd := Command{Nonce: "ts-300", Verb: VerbDispatch, Target: "#2265"}

	if d := s.deliver(cmd, Admission{Reason: "REFUSE_NO_SEAT"}); d.Action != Refuse {
		t.Fatalf("first delivery: action = %s, want REFUSE", d.Action)
	}
	d := s.deliver(cmd, Admission{Admitted: true, RunID: "run-late", Lane: "chatops"})
	if d.Action != Dispatch || d.RunID != "run-late" {
		t.Fatalf("re-admitted delivery: %+v, want DISPATCH run-late", d)
	}
	// And now it is idempotent like any dispatched command.
	if again := s.deliver(cmd, Admission{Admitted: true, RunID: "run-late"}); again.Action != ReAck {
		t.Fatalf("post-dispatch delivery: action = %s, want RE_ACK", again.Action)
	}
}

// An empty refusal token is normalized so a refusal line never reads as a
// dangling "refused: ".
func TestRefusalEmptyTokenNormalized(t *testing.T) {
	d := Decide(Command{Nonce: "n", Verb: VerbDispatch, Target: "#1"}, Admission{}, Record{})
	if d.Action != Refuse || !strings.Contains(d.Reason, "REFUSE_UNSPECIFIED") {
		t.Fatalf("empty-token refusal not normalized: %+v", d)
	}
}

// Decide is a pure fold: identical inputs yield an identical decision, byte for
// byte, so a replay (or a re-delivery on a warm spool) is deterministic.
func TestDecideIsDeterministic(t *testing.T) {
	cmd := Command{Nonce: "ts-400", Verb: VerbResume, Target: "run-7", Channel: "C9"}
	adm := Admission{Admitted: true, RunID: "run-7", Lane: "loops"}
	a := Decide(cmd, adm, Record{})
	b := Decide(cmd, adm, Record{})
	if a != b {
		t.Fatalf("Decide not deterministic:\n a = %+v\n b = %+v", a, b)
	}
}

// DoD #4 (stall escalation): a run silent past its budget drives the
// blockers-escalation path; the severity climbs from a background note to an
// operator page as the silence deepens; a run within budget (or with no budget)
// does not escalate.
func TestJudgeStallEscalation(t *testing.T) {
	budget := 10 * time.Minute
	cases := []struct {
		name         string
		silent       time.Duration
		budget       time.Duration
		wantEscalate bool
		wantSeverity string
	}{
		{"within budget", 5 * time.Minute, budget, false, ""},
		{"exactly at budget does not stall", budget, budget, false, ""},
		{"past budget escalates as status", 15 * time.Minute, budget, true, SeverityStatus},
		{"double budget pages the operator", 20 * time.Minute, budget, true, SeverityOperator},
		{"well past budget pages the operator", 90 * time.Minute, budget, true, SeverityOperator},
		{"no budget cannot stall", 90 * time.Minute, 0, false, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := JudgeStall(Stall{RunID: "run-s", Issue: "#2265", SilentFor: tc.silent, Budget: tc.budget})
			if e.Escalate != tc.wantEscalate {
				t.Fatalf("escalate = %v, want %v (%+v)", e.Escalate, tc.wantEscalate, e)
			}
			if !tc.wantEscalate {
				return
			}
			if e.Severity != tc.wantSeverity {
				t.Fatalf("severity = %q, want %q", e.Severity, tc.wantSeverity)
			}
			if !strings.Contains(e.Text, "run-s") || !strings.Contains(e.Text, "#2265") {
				t.Fatalf("stall text %q missing run/issue identity", e.Text)
			}
			if tc.wantSeverity == SeverityOperator && !strings.Contains(e.Text, "<!here>") {
				t.Fatalf("operator page %q missing the <!here> page prefix", e.Text)
			}
		})
	}
}

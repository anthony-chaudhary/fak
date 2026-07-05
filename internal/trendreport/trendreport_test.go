package trendreport

import (
	"encoding/json"
	"strings"
	"testing"
)

// row is a minimal concrete ledger row standing in for the consumers' richer
// rows, so the generic append-line marshaller can be exercised without importing
// a real consumer.
type row struct {
	Date        string `json:"date"`
	GeneratedAt string `json:"generated_at"`
	Debt        int    `json:"debt"`
}

func TestAppendLedgerLineRoundTrips(t *testing.T) {
	r := row{Date: "2026-06-29", GeneratedAt: "2026-06-29T12:00:00Z", Debt: 7}
	line, err := AppendLedgerLine(r)
	if err != nil {
		t.Fatalf("AppendLedgerLine err: %v", err)
	}
	if strings.Contains(line, "\n") {
		t.Errorf("rendered line must carry no trailing newline: %q", line)
	}
	var back row
	if err := json.Unmarshal([]byte(line), &back); err != nil {
		t.Fatalf("rendered line is not valid JSON: %v (%q)", err, line)
	}
	if back != r {
		t.Errorf("round-trip mismatch: got %+v want %+v", back, r)
	}
}

func TestDirectionWord(t *testing.T) {
	cases := []struct {
		delta int
		want  string
	}{
		{5, "up"},
		{-3, "down"},
		{0, "flat"},
	}
	for _, c := range cases {
		if got := DirectionWord(c.delta); got != c.want {
			t.Errorf("DirectionWord(%d) = %q, want %q", c.delta, got, c.want)
		}
	}
}

func TestStampSeedsEnvelope(t *testing.T) {
	e := Stamp("fak-test-report/1", Opts{
		Workspace:   "/repo",
		Commit:      "abc123",
		GeneratedAt: "2026-06-29T12:00:00Z",
		Date:        "2026-06-29",
	})
	if e.Schema != "fak-test-report/1" {
		t.Errorf("schema = %q", e.Schema)
	}
	if e.Workspace != "/repo" || e.Commit != "abc123" || e.Date != "2026-06-29" || e.GeneratedAt != "2026-06-29T12:00:00Z" {
		t.Errorf("ambient context not stamped: %+v", e)
	}
	// Stamp leaves the verdict fields zero for the consumer's fold to set.
	if e.OK || e.Verdict != "" || e.Finding != "" || e.GateExit != nil {
		t.Errorf("Stamp should not set verdict/gate fields: %+v", e)
	}
}

func TestAdvisoryGate(t *testing.T) {
	const unmeasured = "cadence_unmeasured"
	tests := []struct {
		name        string
		finding     string
		reason      string
		wantExit    int
		wantContain string
	}{
		{
			name:        "unmeasured finding fails the gate",
			finding:     unmeasured,
			reason:      "could not measure scores",
			wantExit:    1,
			wantContain: "CADENCE INCOMPLETE: could not measure scores",
		},
		{
			name:        "recorded finding passes",
			finding:     "cadence_recorded",
			reason:      "cadence recorded; all green",
			wantExit:    0,
			wantContain: "CADENCE OK: cadence recorded; all green",
		},
		{
			name:        "advisory (regression) finding still passes — mirror not gate",
			finding:     "cadence_advisory",
			reason:      "cadence recorded; score debt regressed",
			wantExit:    0,
			wantContain: "CADENCE OK:",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			v := AdvisoryGate("CADENCE", tc.finding, tc.reason, unmeasured)
			if v.Exit != tc.wantExit {
				t.Errorf("exit = %d, want %d", v.Exit, tc.wantExit)
			}
			if !strings.Contains(v.Message, tc.wantContain) {
				t.Errorf("message %q does not contain %q", v.Message, tc.wantContain)
			}
		})
	}
}

func TestWithGate(t *testing.T) {
	base := Stamp("fak-test-report/1", Opts{Date: "2026-06-29"})
	base.Finding = "test_recorded"

	t.Run("exit 0 reconciles to OK", func(t *testing.T) {
		v := GateVerdict{Exit: 0, Message: "TEST OK: recorded"}
		got := base.WithGate(v)
		if !got.OK || got.Verdict != VerdictOK {
			t.Errorf("want OK/%s, got ok=%v verdict=%q", VerdictOK, got.OK, got.Verdict)
		}
		if got.GateExit == nil || *got.GateExit != 0 || got.GateMessage != "TEST OK: recorded" {
			t.Errorf("gate fields wrong: exit=%v msg=%q", got.GateExit, got.GateMessage)
		}
		// WithGate must not mutate the receiver.
		if base.GateExit != nil {
			t.Errorf("WithGate mutated the receiver: %+v", base)
		}
	})

	t.Run("exit 1 reconciles to ACTION", func(t *testing.T) {
		v := GateVerdict{Exit: 1, Message: "TEST INCOMPLETE: unmeasured"}
		got := base.WithGate(v)
		if got.OK || got.Verdict != VerdictAction {
			t.Errorf("want ACTION, got ok=%v verdict=%q", got.OK, got.Verdict)
		}
		if got.GateExit == nil || *got.GateExit != 1 {
			t.Errorf("gate exit wrong: %v", got.GateExit)
		}
	})
}

// TestEnvelopeJSONTags pins the envelope's wire shape: a consumer embedding
// Envelope must emit exactly the field names the existing two reports already do,
// so the switch-over is byte-preserving.
func TestEnvelopeJSONTags(t *testing.T) {
	exit := 0
	e := Envelope{
		Schema: "s", OK: true, Verdict: "OK", Finding: "f", Reason: "r",
		NextAction: "n", Workspace: "w", Commit: "c", GeneratedAt: "g", Date: "d",
		GateExit: &exit, GateMessage: "m",
	}
	b, err := json.Marshal(e)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for _, want := range []string{
		`"schema":"s"`, `"ok":true`, `"verdict":"OK"`, `"finding":"f"`,
		`"reason":"r"`, `"next_action":"n"`, `"workspace":"w"`, `"commit":"c"`,
		`"generated_at":"g"`, `"date":"d"`, `"gate_exit":0`, `"gate_message":"m"`,
	} {
		if !strings.Contains(string(b), want) {
			t.Errorf("envelope JSON missing %s\n  got %s", want, b)
		}
	}

	// The two gate fields are omitempty: a zero-value envelope omits them.
	z, _ := json.Marshal(Envelope{})
	if strings.Contains(string(z), "gate_exit") || strings.Contains(string(z), "gate_message") {
		t.Errorf("zero envelope should omit gate fields, got %s", z)
	}
}

package trajectoryassurance

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

func TestAnalyzeEventIntegrityFaultInjection(t *testing.T) {
	base := time.Date(2026, 8, 30, 12, 0, 0, 0, time.UTC)
	events := []trajectory.Event{
		integrityEvent("start-ok", trajectory.EventRunLifecycle, "started", 1, base, nil, `{"run_id":"ok"}`),
		integrityEvent("complete-ok", trajectory.EventRunLifecycle, "completed", 2, base.Add(time.Second), []string{"start-ok"}, `{"run_id":"ok"}`),
		integrityEvent("start-lost", trajectory.EventRunLifecycle, "started", 4, base.Add(2*time.Second), nil, `{"run_id":"lost-start"}`),
		integrityEvent("complete-orphan", trajectory.EventRunLifecycle, "completed", 5, base.Add(3*time.Second), nil, `{"run_id":"lost-complete"}`),
		integrityEvent("call-ok", trajectory.EventTool, "proposed", 6, base.Add(4*time.Second), nil, `{"tool_call_id":"ok"}`),
		integrityEvent("result-ok", trajectory.EventTool, "completed", 7, base.Add(5*time.Second), []string{"call-ok"}, `{"tool_call_id":"ok"}`),
		integrityEvent("call-orphan", trajectory.EventTool, "proposed", 8, base.Add(6*time.Second), nil, `{"tool_call_id":"lost-call"}`),
		integrityEvent("result-orphan", trajectory.EventTool, "completed", 9, base.Add(time.Second), nil, `{"tool_call_id":"lost-result"}`),
	}
	input := encodeIntegrityEvents(t, events)
	input = bytes.Replace(input, []byte("\n"), []byte("\nnot-json\n"), 1)

	report, err := AnalyzeEventIntegrity(bytes.NewReader(input), EventIntegrityConfig{Strict: true, SampleLimit: 1})
	if err != nil {
		t.Fatal(err)
	}
	want := map[EventIntegrityLoss]int{
		LossMalformedJSONL:      1,
		LossTruncatedFinalRow:   0,
		LossMissingSequence:     1,
		LossUnmatchedStart:      1,
		LossUnmatchedCompletion: 1,
		LossOrphanedToolCall:    1,
		LossOrphanedToolResult:  1,
		LossTimestampRegression: 1,
	}
	for loss, count := range want {
		if got := report.Counts[loss]; got != count {
			t.Errorf("%s count = %d, want %d", loss, got, count)
		}
		if count > 0 && len(report.Samples[loss]) != 1 {
			t.Errorf("%s samples = %d, want bounded sample", loss, len(report.Samples[loss]))
		}
	}
	if report.Pass {
		t.Fatal("strict zero-budget report passed")
	}
	if len(report.Violations) != 7 {
		t.Fatalf("violations = %d, want 7: %#v", len(report.Violations), report.Violations)
	}
}

func TestAnalyzeEventIntegrityDetectsAbsentSequenceField(t *testing.T) {
	event := integrityEvent("no-sequence", trajectory.EventMessage, "observed", 0, time.Date(2026, 8, 30, 12, 0, 0, 0, time.UTC), nil, `{}`)
	report, err := AnalyzeEventIntegrity(bytes.NewReader(encodeIntegrityEvents(t, []trajectory.Event{event})), EventIntegrityConfig{})
	if err != nil {
		t.Fatal(err)
	}
	if report.Counts[LossMissingSequence] != 1 || len(report.Samples[LossMissingSequence]) != 1 {
		t.Fatalf("missing sequence diagnostics = %#v", report)
	}
}
func TestAnalyzeEventIntegrityLiveAndClosedTruncation(t *testing.T) {
	complete := encodeIntegrityEvents(t, []trajectory.Event{
		integrityEvent("one", trajectory.EventMessage, "observed", 1, time.Date(2026, 8, 30, 12, 0, 0, 0, time.UTC), nil, `{}`),
	})
	partial := append(complete, []byte(`{"schema":"fak-trajectory-event/1","id":"in-flight"`)...)

	live, err := AnalyzeEventIntegrity(bytes.NewReader(partial), EventIntegrityConfig{Live: true, Strict: true})
	if err != nil {
		t.Fatal(err)
	}
	if live.Counts[LossTruncatedFinalRow] != 1 || live.ExpectedPartialFinalRows != 1 || !live.Pass {
		t.Fatalf("live partial verdict = %#v", live)
	}

	closed, err := AnalyzeEventIntegrity(bytes.NewReader(partial), EventIntegrityConfig{Strict: true})
	if err != nil {
		t.Fatal(err)
	}
	if closed.Counts[LossTruncatedFinalRow] != 1 || closed.ExpectedPartialFinalRows != 0 || closed.Pass {
		t.Fatalf("closed partial verdict = %#v", closed)
	}
	if len(closed.Violations) != 1 || closed.Violations[0].Loss != LossTruncatedFinalRow {
		t.Fatalf("closed violations = %#v", closed.Violations)
	}
}

func TestAnalyzeEventIntegrityConfigurableBudgetsAndBoundedSamples(t *testing.T) {
	input := strings.NewReader("bad\nalso-bad\nthird-bad\n")
	report, err := AnalyzeEventIntegrity(input, EventIntegrityConfig{
		Strict:      true,
		SampleLimit: 2,
		Budgets:     map[EventIntegrityLoss]int{LossMalformedJSONL: 3},
	})
	if err != nil {
		t.Fatal(err)
	}
	if report.Counts[LossMalformedJSONL] != 3 || len(report.Samples[LossMalformedJSONL]) != 2 {
		t.Fatalf("malformed diagnostics = count %d, samples %#v", report.Counts[LossMalformedJSONL], report.Samples[LossMalformedJSONL])
	}
	if !report.Pass || len(report.Violations) != 0 {
		t.Fatalf("budgeted report failed: %#v", report.Violations)
	}
	for _, loss := range allEventIntegrityLosses {
		if _, ok := report.Counts[loss]; !ok {
			t.Errorf("missing zero count for %s", loss)
		}
		if _, ok := report.Samples[loss]; !ok {
			t.Errorf("missing sample bucket for %s", loss)
		}
	}
}

func TestAnalyzeEventIntegrityRejectsInvalidConfig(t *testing.T) {
	if _, err := AnalyzeEventIntegrity(nil, EventIntegrityConfig{}); err == nil {
		t.Fatal("nil reader accepted")
	}
	if _, err := AnalyzeEventIntegrity(strings.NewReader(""), EventIntegrityConfig{SampleLimit: -1}); err == nil {
		t.Fatal("negative sample limit accepted")
	}
}

func integrityEvent(id string, kind trajectory.EventKind, action string, sequence uint64, timestamp time.Time, parents []string, payload string) trajectory.Event {
	return trajectory.Event{
		Schema:         trajectory.EventSchema,
		ID:             id,
		ConversationID: "conversation-1",
		Kind:           kind,
		Action:         action,
		Timestamp:      timestamp,
		Sequence:       sequence,
		ParentIDs:      parents,
		Visibility:     trajectory.VisibilityPublic,
		Source: trajectory.EventSource{
			Type: "fault-injection", Adapter: "event-integrity-test", AdapterVersion: "1",
		},
		Payload: json.RawMessage(payload),
	}
}

func encodeIntegrityEvents(t *testing.T, events []trajectory.Event) []byte {
	t.Helper()
	encoded, err := trajectory.EncodeEvents(events)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}

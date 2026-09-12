package agentbench

import (
	"fmt"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentbench/profileplan"
)

func TestNormalQualificationRequiresCompleteKnownPassingSLOs(t *testing.T) {
	plan, err := profileplan.Build(profileplan.Options{Profile: "normal", PreferredConcurrency: 2})
	if err != nil {
		t.Fatal(err)
	}
	cells, err := compileNormalSchedule(plan)
	if err != nil {
		t.Fatal(err)
	}
	events := qualifyingNormalEvents(cells)
	got := summarizeNormalQualification(plan, cells, events, 2)
	if !got.Qualified || len(got.Unknowns) != 0 {
		t.Fatalf("complete passing normal evidence did not qualify: %+v", got)
	}
	for name, metric := range map[string]normalMetric{
		"cold first output": got.SLOs.ColdFirstOutput,
		"cold complete":     got.SLOs.ColdComplete,
		"new area":          got.SLOs.NewAreaFirstOutput,
		"warm p50":          got.SLOs.WarmTTFTP50,
		"warm p90":          got.SLOs.WarmTTFTP90,
		"short decision":    got.SLOs.ShortDecisionP90,
		"medium complete":   got.SLOs.MediumCompleteP90,
		"long complete":     got.SLOs.LongCompleteP90,
		"progress gap":      got.SLOs.ProgressGapMaximum,
		"warm complete":     got.SLOs.WarmComplete,
	} {
		if !metric.Known || !metric.Passed || metric.LimitMillis <= 0 {
			t.Fatalf("%s metric = %+v, want known passing exact limit", name, metric)
		}
	}

	missing := append([]lifecycleEvent(nil), events[:len(events)-1]...)
	if incomplete := summarizeNormalQualification(plan, cells, missing, 2); incomplete.Qualified || incomplete.Reason == "" {
		t.Fatalf("missing required cell event qualified: %+v", incomplete)
	}
	unknown := withoutNormalEvent(events, "first_output")
	if result := summarizeNormalQualification(plan, cells, unknown, 2); result.Qualified || len(result.Unknowns) == 0 {
		t.Fatalf("unknown TTFT qualified: %+v", result)
	}
	exceeded := append([]lifecycleEvent(nil), events...)
	for i := range exceeded {
		if exceeded[i].Phase == "steady" && exceeded[i].Turn > 1 && exceeded[i].ScoredRequest && (exceeded[i].Event == "end" || exceeded[i].EventType == "terminal") {
			exceeded[i].DurationMilliseconds = plan.SLOs.WarmRequestCompleteWithin.Milliseconds() + 1
			break
		}
	}
	if result := summarizeNormalQualification(plan, cells, exceeded, 2); result.Qualified || result.Reason == "" {
		t.Fatalf("exceeded SLO qualified: %+v", result)
	}
}

func TestNormalSLOEvidenceCannotDropMissingOrSubstituteFirstByte(t *testing.T) {
	plan, _ := profileplan.Build(profileplan.Options{Profile: "normal", PreferredConcurrency: 2})
	cells, _ := compileNormalSchedule(plan)
	complete := qualifyingNormalEvents(cells)
	missingOne := append([]lifecycleEvent(nil), complete...)
	for i, event := range missingOne {
		if event.Rung == "steady" && event.Turn > 1 && event.Event == "first_output" {
			missingOne = append(missingOne[:i], missingOne[i+1:]...)
			break
		}
	}
	if got := deriveNormalSLOs(missingOne).WarmTTFTP90; got.Known {
		t.Fatalf("warm TTFT silently dropped a required request with no first-output event: %+v", got)
	}

	lateDecision := append([]lifecycleEvent(nil), complete...)
	for i := range lateDecision {
		if lateDecision[i].Rung == "steady" && lateDecision[i].Turn >= 2 && lateDecision[i].Turn <= 8 {
			start, _ := time.Parse(time.RFC3339Nano, lateDecision[i].StartedAt)
			if lateDecision[i].Event == "tool" {
				start = start.Add(-400 * time.Millisecond)
				lateDecision[i].StartedAt = start.Add(11 * time.Second).Format(time.RFC3339Nano)
			}
			if lateDecision[i].Event == "end" {
				lateDecision[i].CompletedAt = start.Add(11 * time.Second).Format(time.RFC3339Nano)
				lateDecision[i].DurationMilliseconds = 11000
			}
		}
	}
	if got := deriveNormalSLOs(lateDecision).ShortDecisionP90; !got.Known || got.Passed || got.ObservedMillis != 11000 {
		t.Fatalf("short decision used first byte instead of completed parseable tool call: %+v", got)
	}

	if got := deriveNormalSLOs(complete).ProgressGapMaximum; !got.Known || !got.Passed || got.ObservedMillis > 10000 {
		t.Fatalf("continuous streamed progress failed gap metric: %+v", got)
	}
	stalled := append([]lifecycleEvent(nil), complete...)
	stallSequence := 0
	for _, event := range stalled {
		if event.Rung == "steady" && event.Turn > 1 {
			stallSequence = event.Sequence
			break
		}
	}
	filtered := stalled[:0]
	for _, event := range stalled {
		if event.Sequence == stallSequence && event.Event == "progress" {
			continue
		}
		if event.Sequence == stallSequence && event.Event == "end" {
			start, _ := time.Parse(time.RFC3339Nano, event.StartedAt)
			event.CompletedAt, event.DurationMilliseconds = start.Add(12*time.Second).Format(time.RFC3339Nano), 12000
		}
		filtered = append(filtered, event)
	}
	if got := deriveNormalSLOs(filtered).ProgressGapMaximum; !got.Known || got.Passed || got.ObservedMillis <= 10000 {
		t.Fatalf("observed single-chunk stream with an unexplained >10s first-to-terminal gap passed: %+v", got)
	}
}

func qualifyingNormalEvents(cells []normalCell) []lifecycleEvent {
	events := normalPreconditionEvents()
	sequence := 0
	base := time.Date(2026, 9, 10, 12, 0, 0, 0, time.UTC)
	known := true
	for _, cell := range cells {
		for session := 1; session <= cell.Sessions; session++ {
			for turn := 1; turn <= cell.TurnsPerSession; turn++ {
				sequence++
				id := fmt.Sprintf("%s-S%d-T%d", cell.Name, session, turn)
				start := base.Add(time.Duration(sequence) * time.Second)
				common := lifecycleEvent{Sequence: sequence, RequestID: id, Rung: cell.Name, ConditionID: cell.Name, Phase: cell.Phase, Session: fmt.Sprintf("%s-S%d", cell.Name, session), Turn: turn, ScoredRequest: true, RequestedModel: "fixture-model", ObservedModel: "fixture-model", ModelIdentityStatus: "observed"}
				first := common
				first.Event, first.EventType, first.Status = "first_output", "first_output", "in_progress"
				first.StartedAt, first.CompletedAt = start.Format(time.RFC3339Nano), start.Add(100*time.Millisecond).Format(time.RFC3339Nano)
				events = append(events, first)
				progress := common
				progress.Event, progress.EventType, progress.Status = "progress", "stream_progress", "in_progress"
				progress.StartedAt = start.Add(250 * time.Millisecond).Format(time.RFC3339Nano)
				events = append(events, progress)
				if cell.Phase == "steady" && turn >= 2 && turn <= 8 {
					tool := common
					tool.Event, tool.EventType, tool.Status = "tool", "tool_call", "in_progress"
					tool.StartedAt = start.Add(400 * time.Millisecond).Format(time.RFC3339Nano)
					events = append(events, tool)
				}
				terminal := common
				terminal.Event, terminal.EventType, terminal.Status, terminal.ServiceVerdict = "end", "terminal", "completed", "OBSERVED_PASS"
				terminal.StartedAt, terminal.CompletedAt, terminal.DurationMilliseconds = start.Format(time.RFC3339Nano), start.Add(500*time.Millisecond).Format(time.RFC3339Nano), 500
				terminal.UsageKnown, terminal.Usage = &known, &observedUsage{PromptTokens: 14000, CompletionTokens: 32, TotalTokens: 14032}
				events = append(events, terminal)
			}
		}
	}
	return events
}

func withoutNormalEvent(events []lifecycleEvent, event string) []lifecycleEvent {
	out := make([]lifecycleEvent, 0, len(events))
	for _, candidate := range events {
		if candidate.Event != event && candidate.EventType != event {
			out = append(out, candidate)
		}
	}
	return out
}

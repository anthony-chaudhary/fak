package trajectoryassurance

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

// EventIntegrityLoss identifies one independently budgetable transcript loss class.
type EventIntegrityLoss string

const (
	LossMalformedJSONL      EventIntegrityLoss = "malformed_jsonl"
	LossTruncatedFinalRow   EventIntegrityLoss = "truncated_final_row"
	LossMissingSequence     EventIntegrityLoss = "missing_sequence"
	LossUnmatchedStart      EventIntegrityLoss = "unmatched_start"
	LossUnmatchedCompletion EventIntegrityLoss = "unmatched_completion"
	LossOrphanedToolCall    EventIntegrityLoss = "orphaned_tool_call"
	LossOrphanedToolResult  EventIntegrityLoss = "orphaned_tool_result"
	LossTimestampRegression EventIntegrityLoss = "timestamp_regression"
)

// EventIntegrityConfig controls sampling and strict budget evaluation. A nil
// Budgets map means every non-live loss has a zero budget when Strict is true.
type EventIntegrityConfig struct {
	Live        bool
	Strict      bool
	SampleLimit int
	Budgets     map[EventIntegrityLoss]int
}

// EventIntegritySample is a bounded, line-addressable example of a loss.
type EventIntegritySample struct {
	Line           int    `json:"line"`
	EventID        string `json:"event_id,omitempty"`
	ConversationID string `json:"conversation_id,omitempty"`
	Detail         string `json:"detail"`
}

// EventIntegrityViolation records a strict budget excess.
type EventIntegrityViolation struct {
	Loss   EventIntegrityLoss `json:"loss"`
	Count  int                `json:"count"`
	Budget int                `json:"budget"`
}

// EventIntegrityReport contains stable counts for every loss class, bounded
// examples, and a strict verdict. ExpectedPartialFinalRows distinguishes an
// ordinary unterminated row in a live file from corruption in a closed file.
type EventIntegrityReport struct {
	Events                   int                                           `json:"events"`
	Counts                   map[EventIntegrityLoss]int                    `json:"counts"`
	Samples                  map[EventIntegrityLoss][]EventIntegritySample `json:"samples"`
	ExpectedPartialFinalRows int                                           `json:"expected_partial_final_rows"`
	Pass                     bool                                          `json:"pass"`
	Violations               []EventIntegrityViolation                     `json:"violations,omitempty"`
}

type observedEvent struct {
	event trajectory.Event
	line  int
}

// AnalyzeEventIntegrity reads canonical trajectory.Event JSONL without losing
// diagnostics when individual rows are damaged. It does not require the input
// to fit in memory as one byte slice, though valid events are retained for
// cross-row correlation.
func AnalyzeEventIntegrity(r io.Reader, cfg EventIntegrityConfig) (EventIntegrityReport, error) {
	if r == nil {
		return EventIntegrityReport{}, fmt.Errorf("event integrity: nil reader")
	}
	if cfg.SampleLimit < 0 {
		return EventIntegrityReport{}, fmt.Errorf("event integrity: negative sample limit")
	}
	limit := cfg.SampleLimit
	if limit == 0 {
		limit = 5
	}
	report := newEventIntegrityReport()
	var events []observedEvent
	reader := bufio.NewReader(r)
	for line := 1; ; line++ {
		row, err := reader.ReadBytes('\n')
		if len(row) == 0 && err == io.EOF {
			break
		}
		terminated := len(row) > 0 && row[len(row)-1] == '\n'
		trimmed := bytes.TrimSpace(row)
		if len(trimmed) > 0 {
			var event trajectory.Event
			decodeErr := json.Unmarshal(trimmed, &event)
			if decodeErr == nil {
				decodeErr = event.Validate()
			}
			if decodeErr != nil {
				if err == io.EOF && !terminated && looksTruncatedJSON(trimmed) {
					report.add(LossTruncatedFinalRow, limit, EventIntegritySample{Line: line, Detail: decodeErr.Error()})
					if cfg.Live {
						report.ExpectedPartialFinalRows++
					}
				} else {
					report.add(LossMalformedJSONL, limit, EventIntegritySample{Line: line, Detail: decodeErr.Error()})
				}
			} else {
				events = append(events, observedEvent{event: event, line: line})
				report.Events++
			}
		}
		if err != nil {
			if err != io.EOF {
				return report, fmt.Errorf("event integrity: read line %d: %w", line, err)
			}
			break
		}
	}

	analyzeSequences(events, &report, limit)
	analyzeTimestamps(events, &report, limit)
	analyzePairs(events, &report, limit)
	report.applyVerdict(cfg)
	return report, nil
}

func newEventIntegrityReport() EventIntegrityReport {
	counts := make(map[EventIntegrityLoss]int, len(allEventIntegrityLosses))
	samples := make(map[EventIntegrityLoss][]EventIntegritySample, len(allEventIntegrityLosses))
	for _, loss := range allEventIntegrityLosses {
		counts[loss] = 0
		samples[loss] = []EventIntegritySample{}
	}
	return EventIntegrityReport{Counts: counts, Samples: samples, Pass: true}
}

var allEventIntegrityLosses = []EventIntegrityLoss{
	LossMalformedJSONL, LossTruncatedFinalRow, LossMissingSequence,
	LossUnmatchedStart, LossUnmatchedCompletion, LossOrphanedToolCall,
	LossOrphanedToolResult, LossTimestampRegression,
}

func (r *EventIntegrityReport) add(loss EventIntegrityLoss, limit int, sample EventIntegritySample) {
	r.Counts[loss]++
	if len(r.Samples[loss]) < limit {
		r.Samples[loss] = append(r.Samples[loss], sample)
	}
}

func (r *EventIntegrityReport) applyVerdict(cfg EventIntegrityConfig) {
	if !cfg.Strict {
		return
	}
	for _, loss := range allEventIntegrityLosses {
		count := r.Counts[loss]
		if loss == LossTruncatedFinalRow && cfg.Live {
			count -= r.ExpectedPartialFinalRows
		}
		budget := cfg.Budgets[loss]
		if count > budget {
			r.Pass = false
			r.Violations = append(r.Violations, EventIntegrityViolation{Loss: loss, Count: count, Budget: budget})
		}
	}
}

func looksTruncatedJSON(row []byte) bool {
	if json.Valid(row) {
		return false
	}
	var value any
	err := json.Unmarshal(row, &value)
	return err != nil && strings.Contains(err.Error(), "unexpected end of JSON input")
}

func analyzeSequences(events []observedEvent, report *EventIntegrityReport, limit int) {
	byConversation := make(map[string][]observedEvent)
	for _, event := range events {
		byConversation[event.event.ConversationID] = append(byConversation[event.event.ConversationID], event)
	}
	for conversation, group := range byConversation {
		sort.Slice(group, func(i, j int) bool { return group[i].event.Sequence < group[j].event.Sequence })
		expected := uint64(1)
		for _, item := range group {
			sequence := item.event.Sequence
			if sequence == 0 {
				report.add(LossMissingSequence, limit, sampleFor(item, "sequence is absent or zero"))
				continue
			}
			if sequence > expected {
				missing := sequence - expected
				report.Counts[LossMissingSequence] += int(missing)
				if len(report.Samples[LossMissingSequence]) < limit {
					report.Samples[LossMissingSequence] = append(report.Samples[LossMissingSequence], EventIntegritySample{
						Line: item.line, EventID: item.event.ID, ConversationID: conversation,
						Detail: fmt.Sprintf("missing sequence %d..%d before %d", expected, sequence-1, sequence),
					})
				}
			}
			if sequence >= expected {
				expected = sequence + 1
			}
		}
	}
}

func analyzeTimestamps(events []observedEvent, report *EventIntegrityReport, limit int) {
	last := make(map[string]observedEvent)
	for _, item := range events {
		previous, ok := last[item.event.ConversationID]
		if ok && item.event.Timestamp.Before(previous.event.Timestamp) {
			report.add(LossTimestampRegression, limit, sampleFor(item,
				fmt.Sprintf("timestamp %s precedes line %d timestamp %s", item.event.Timestamp.Format(time.RFC3339Nano), previous.line, previous.event.Timestamp.Format(time.RFC3339Nano))))
		}
		last[item.event.ConversationID] = item
	}
}

func analyzePairs(events []observedEvent, report *EventIntegrityReport, limit int) {
	var starts, completions, calls, results []observedEvent
	for _, item := range events {
		action := normalizedAction(item.event.Action)
		if item.event.Kind == trajectory.EventTool {
			switch action {
			case "started", "start", "proposed", "called", "call", "tool_call", "tool_use":
				calls = append(calls, item)
			case "completed", "complete", "result", "tool_result", "returned":
				results = append(results, item)
			}
			continue
		}
		switch action {
		case "started", "start":
			starts = append(starts, item)
		case "completed", "complete":
			completions = append(completions, item)
		}
	}
	matchPairs(starts, completions, LossUnmatchedStart, LossUnmatchedCompletion, report, limit)
	matchPairs(calls, results, LossOrphanedToolCall, LossOrphanedToolResult, report, limit)
}

func matchPairs(left, right []observedEvent, leftLoss, rightLoss EventIntegrityLoss, report *EventIntegrityReport, limit int) {
	used := make([]bool, len(right))
	for _, first := range left {
		match := -1
		for i, second := range right {
			if !used[i] && pairMatches(first.event, second.event) {
				match = i
				break
			}
		}
		if match >= 0 {
			used[match] = true
		} else {
			report.add(leftLoss, limit, sampleFor(first, "no matching completion/result"))
		}
	}
	for i, second := range right {
		if !used[i] {
			report.add(rightLoss, limit, sampleFor(second, "no matching start/call"))
		}
	}
}

func pairMatches(first, second trajectory.Event) bool {
	if first.ConversationID != second.ConversationID {
		return false
	}
	for _, parent := range second.ParentIDs {
		if parent == first.ID {
			return true
		}
	}
	firstKey := correlationKey(first.Payload)
	return firstKey != "" && firstKey == correlationKey(second.Payload)
}

func correlationKey(payload json.RawMessage) string {
	var fields map[string]any
	if json.Unmarshal(payload, &fields) != nil {
		return ""
	}
	for _, key := range []string{"tool_call_id", "call_id", "invocation_id", "operation_id", "request_id", "run_id", "span_id"} {
		if value, ok := fields[key].(string); ok && value != "" {
			return key + ":" + value
		}
	}
	return ""
}

func normalizedAction(action string) string {
	return strings.ToLower(strings.ReplaceAll(strings.TrimSpace(action), "-", "_"))
}

func sampleFor(item observedEvent, detail string) EventIntegritySample {
	return EventIntegritySample{Line: item.line, EventID: item.event.ID, ConversationID: item.event.ConversationID, Detail: detail}
}

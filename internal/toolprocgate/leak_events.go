package toolprocgate

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

const (
	LeakEventSchema  = "fak.toolprocgate.leak-event.v1"
	LeakReportSchema = "fak.toolprocgate.leak-report.v1"
)

// LeakAction is the closed leak-prevention event vocabulary. These are
// observability rows only; enforcement leaves own the actual deny/strip/reap.
type LeakAction string

const (
	LeakSpawnAllowed                LeakAction = "spawn_allowed"
	LeakSpawnDenied                 LeakAction = "spawn_denied"
	LeakEnvStripped                 LeakAction = "env_stripped"
	LeakSecretGranted               LeakAction = "secret_granted"
	LeakEgressDenied                LeakAction = "egress_denied"
	LeakFSDenied                    LeakAction = "fs_denied"
	LeakOutputQuarantined           LeakAction = "output_quarantined"
	LeakDescendantReaped            LeakAction = "descendant_reaped"
	LeakUnmanagedDescendantDetected LeakAction = "unmanaged_descendant_detected"
)

// DescendantState is the bounded process-tree state recorded with a leak event.
type DescendantState string

const (
	DescendantNone      DescendantState = "none"
	DescendantRunning   DescendantState = "running"
	DescendantExited    DescendantState = "exited"
	DescendantReaped    DescendantState = "reaped"
	DescendantUnmanaged DescendantState = "unmanaged"
	DescendantUnknown   DescendantState = "unknown"
)

// BoundedRef identifies blocked material without retaining or reporting bytes.
type BoundedRef struct {
	Kind   string `json:"kind"`
	Digest string `json:"digest"`
	Len    int64  `json:"len"`
}

// LeakEvent is one append-only leak-prevention row. It carries identity,
// policy, backend, reason, source channel, a byte-free reference, and process
// tree state. It intentionally has no payload/env/value fields.
type LeakEvent struct {
	Schema          string          `json:"schema,omitempty"`
	Action          LeakAction      `json:"action"`
	AtMS            int64           `json:"at_unix_ms"`
	AgentRunID      string          `json:"agent_run_id"`
	ParentRunID     string          `json:"parent_run_id"`
	ToolCallID      string          `json:"tool_call_id"`
	TraceID         string          `json:"trace_id"`
	PolicyDigest    string          `json:"policy_digest"`
	Backend         string          `json:"backend"`
	Reason          string          `json:"reason"`
	BoundedRef      BoundedRef      `json:"bounded_ref"`
	SourceChannel   string          `json:"source_channel"`
	DescendantState DescendantState `json:"descendant_state"`
}

type LeakReportCounts struct {
	ByAction          map[string]int `json:"by_action"`
	ByChannel         map[string]int `json:"by_channel"`
	ByReason          map[string]int `json:"by_reason"`
	ByDescendantState map[string]int `json:"by_descendant_state"`
}

type LeakReportRow struct {
	Action          LeakAction      `json:"action"`
	AgentRunID      string          `json:"agent_run_id"`
	ParentRunID     string          `json:"parent_run_id"`
	ToolCallID      string          `json:"tool_call_id"`
	TraceID         string          `json:"trace_id"`
	PolicyDigest    string          `json:"policy_digest"`
	Backend         string          `json:"backend"`
	Reason          string          `json:"reason"`
	BoundedRef      BoundedRef      `json:"bounded_ref"`
	SourceChannel   string          `json:"source_channel"`
	DescendantState DescendantState `json:"descendant_state"`
}

type LeakReport struct {
	Schema    string           `json:"schema"`
	Rows      int              `json:"rows"`
	Denied    int              `json:"denied"`
	Counts    LeakReportCounts `json:"counts"`
	EventRows []LeakReportRow  `json:"event_rows"`
}

// ParseLeakEvents reads leak-event JSONL. Unknown fields are refused so raw
// payload/env/value fields cannot silently enter an operator report.
func ParseLeakEvents(r io.Reader) ([]LeakEvent, error) {
	var out []LeakEvent
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	line := 0
	for sc.Scan() {
		line++
		raw := strings.TrimSpace(sc.Text())
		if raw == "" || strings.HasPrefix(raw, "#") {
			continue
		}
		dec := json.NewDecoder(strings.NewReader(raw))
		dec.DisallowUnknownFields()
		var ev LeakEvent
		if err := dec.Decode(&ev); err != nil {
			return nil, fmt.Errorf("toolprocgate leak event: line %d: %v", line, err)
		}
		if err := dec.Decode(&struct{}{}); err != io.EOF {
			return nil, fmt.Errorf("toolprocgate leak event: line %d: trailing JSON token", line)
		}
		if err := ValidateLeakEvent(ev); err != nil {
			return nil, fmt.Errorf("toolprocgate leak event: line %d: %v", line, err)
		}
		out = append(out, ev)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("toolprocgate leak event: %v", err)
	}
	return out, nil
}

func ValidateLeakEvent(ev LeakEvent) error {
	if ev.Schema != "" && ev.Schema != LeakEventSchema {
		return fmt.Errorf("unsupported schema %q", ev.Schema)
	}
	if !validLeakAction(ev.Action) {
		return fmt.Errorf("unknown action %q", ev.Action)
	}
	if ev.AtMS <= 0 {
		return fmt.Errorf("%s: at_unix_ms must be positive", ev.Action)
	}
	for name, value := range map[string]string{
		"agent_run_id":   ev.AgentRunID,
		"parent_run_id":  ev.ParentRunID,
		"tool_call_id":   ev.ToolCallID,
		"trace_id":       ev.TraceID,
		"policy_digest":  ev.PolicyDigest,
		"backend":        ev.Backend,
		"source_channel": ev.SourceChannel,
	} {
		if !boundedToken(value, 256) {
			return fmt.Errorf("%s: %s required as a bounded token", ev.Action, name)
		}
	}
	if !reasonToken(ev.Reason) {
		return fmt.Errorf("%s: reason token required", ev.Action)
	}
	if !validDescendantState(ev.DescendantState) {
		return fmt.Errorf("%s: unknown descendant_state %q", ev.Action, ev.DescendantState)
	}
	if !boundedToken(ev.BoundedRef.Kind, 64) {
		return fmt.Errorf("%s: bounded_ref.kind required", ev.Action)
	}
	if !boundedToken(ev.BoundedRef.Digest, 256) {
		return fmt.Errorf("%s: bounded_ref.digest required", ev.Action)
	}
	if ev.BoundedRef.Len < 0 {
		return fmt.Errorf("%s: bounded_ref.len must be non-negative", ev.Action)
	}
	return nil
}

// LeakEventFromSpawnAudit adapts the spawn broker's bounded audit row into the
// leak-prevention event stream consumed by the operator report.
func LeakEventFromSpawnAudit(a SpawnAudit, atMS int64) LeakEvent {
	action := LeakSpawnDenied
	reason := safeLeakReason(a.Reason, "SPAWN_DENIED")
	descendant := DescendantNone
	if a.Verdict == SpawnVerdictAllow {
		action = LeakSpawnAllowed
		reason = safeLeakReason(a.Reason, "SPAWN_ALLOWED")
		descendant = DescendantRunning
	}
	return LeakEvent{
		Schema:          LeakEventSchema,
		Action:          action,
		AtMS:            positiveLeakAtMS(atMS),
		AgentRunID:      safeLeakToken(a.AgentRunID, 256, "unknown"),
		ParentRunID:     safeLeakToken(a.ParentRunID, 256, "unknown"),
		ToolCallID:      safeLeakToken(a.ToolCallID, 256, "unknown"),
		TraceID:         safeLeakToken(a.ToolCallID, 256, "unknown"),
		PolicyDigest:    safeLeakToken(a.PolicyDigest, 256, "unknown"),
		Backend:         safeLeakToken(a.Backend, 256, "unknown"),
		Reason:          reason,
		BoundedRef:      BoundedRef{Kind: "spawn_audit", Digest: digest([]string{a.ArgvDigest, a.EnvDigest, a.CWD, a.Argv0}), Len: 0},
		SourceChannel:   "spawn",
		DescendantState: descendant,
	}
}

// LeakEventFromOutputAdmission adapts a child-output admission refusal into the
// same leak-prevention stream. Benign/admitted output is intentionally silent.
func LeakEventFromOutputAdmission(out OutputAdmission, atMS int64) (LeakEvent, bool) {
	switch out.Verdict.Kind {
	case abi.VerdictQuarantine, abi.VerdictDeny, abi.VerdictRequireWitness:
	default:
		return LeakEvent{}, false
	}
	digestToken := strings.TrimSpace(out.InputSHA256)
	if digestToken == "" {
		digestToken = digest([]string{out.AgentRunID, out.ToolCallID, out.TraceID, string(out.Channel)})
	} else if !strings.HasPrefix(digestToken, "sha256:") {
		digestToken = "sha256:" + digestToken
	}
	descendant := out.DescendantState
	if descendant == "" {
		descendant = DescendantUnknown
	}
	reason := safeLeakReason(abi.ReasonName(out.Verdict.Reason), "OUTPUT_QUARANTINED")
	return LeakEvent{
		Schema:          LeakEventSchema,
		Action:          LeakOutputQuarantined,
		AtMS:            positiveLeakAtMS(atMS),
		AgentRunID:      safeLeakToken(out.AgentRunID, 256, "unknown"),
		ParentRunID:     safeLeakToken(out.ParentRunID, 256, "unknown"),
		ToolCallID:      safeLeakToken(firstNonEmpty(out.ToolCallID, out.CallID), 256, "unknown"),
		TraceID:         safeLeakToken(firstNonEmpty(out.TraceID, out.CallID, out.ToolCallID), 256, "unknown"),
		PolicyDigest:    safeLeakToken(out.PolicyDigest, 256, "unknown"),
		Backend:         safeLeakToken(out.Backend, 256, "unknown"),
		Reason:          reason,
		BoundedRef:      BoundedRef{Kind: "sha256", Digest: safeLeakToken(digestToken, 256, "unknown"), Len: int64(out.InputLen)},
		SourceChannel:   safeLeakToken(string(out.Channel), 256, "unknown"),
		DescendantState: descendant,
	}, true
}

func LeakReportFromEvents(events []LeakEvent) LeakReport {
	rep := LeakReport{
		Schema: LeakReportSchema,
		Counts: LeakReportCounts{
			ByAction:          map[string]int{},
			ByChannel:         map[string]int{},
			ByReason:          map[string]int{},
			ByDescendantState: map[string]int{},
		},
	}
	for _, ev := range events {
		rep.Rows++
		if leakActionDenied(ev.Action) {
			rep.Denied++
		}
		rep.Counts.ByAction[string(ev.Action)]++
		rep.Counts.ByChannel[ev.SourceChannel]++
		rep.Counts.ByReason[ev.Reason]++
		rep.Counts.ByDescendantState[string(ev.DescendantState)]++
		rep.EventRows = append(rep.EventRows, LeakReportRow{
			Action: ev.Action, AgentRunID: ev.AgentRunID, ParentRunID: ev.ParentRunID,
			ToolCallID: ev.ToolCallID, TraceID: ev.TraceID, PolicyDigest: ev.PolicyDigest,
			Backend: ev.Backend, Reason: ev.Reason, BoundedRef: ev.BoundedRef,
			SourceChannel: ev.SourceChannel, DescendantState: ev.DescendantState,
		})
	}
	sort.SliceStable(rep.EventRows, func(i, j int) bool {
		if rep.EventRows[i].AgentRunID != rep.EventRows[j].AgentRunID {
			return rep.EventRows[i].AgentRunID < rep.EventRows[j].AgentRunID
		}
		if rep.EventRows[i].TraceID != rep.EventRows[j].TraceID {
			return rep.EventRows[i].TraceID < rep.EventRows[j].TraceID
		}
		return rep.EventRows[i].ToolCallID < rep.EventRows[j].ToolCallID
	})
	return rep
}

func RenderLeakReport(w io.Writer, rep LeakReport) {
	fmt.Fprintf(w, "toolproc leak-events: rows=%d denied=%d channels=%s reasons=%s descendant=%s\n",
		rep.Rows, rep.Denied, renderCounts(rep.Counts.ByChannel), renderCounts(rep.Counts.ByReason),
		renderCounts(rep.Counts.ByDescendantState))
	for _, row := range rep.EventRows {
		fmt.Fprintf(w, "  agent=%s parent=%s tool_call=%s trace=%s backend=%s action=%s channel=%s reason=%s descendant=%s policy=%s ref=%s:%s len=%d\n",
			row.AgentRunID, row.ParentRunID, row.ToolCallID, row.TraceID, row.Backend,
			row.Action, row.SourceChannel, row.Reason, row.DescendantState,
			row.PolicyDigest, row.BoundedRef.Kind, row.BoundedRef.Digest, row.BoundedRef.Len)
	}
}

func validLeakAction(a LeakAction) bool {
	switch a {
	case LeakSpawnAllowed, LeakSpawnDenied, LeakEnvStripped, LeakSecretGranted,
		LeakEgressDenied, LeakFSDenied, LeakOutputQuarantined, LeakDescendantReaped,
		LeakUnmanagedDescendantDetected:
		return true
	default:
		return false
	}
}

func validDescendantState(s DescendantState) bool {
	switch s {
	case DescendantNone, DescendantRunning, DescendantExited, DescendantReaped,
		DescendantUnmanaged, DescendantUnknown:
		return true
	default:
		return false
	}
}

func leakActionDenied(a LeakAction) bool {
	switch a {
	case LeakSpawnAllowed, LeakSecretGranted:
		return false
	default:
		return true
	}
}

func boundedToken(s string, limit int) bool {
	s = strings.TrimSpace(s)
	if s == "" || len(s) > limit {
		return false
	}
	return !strings.ContainsAny(s, "\r\n\t")
}

func reasonToken(s string) bool {
	if !boundedToken(s, 128) {
		return false
	}
	for _, r := range s {
		if r >= 'A' && r <= 'Z' {
			continue
		}
		if r >= '0' && r <= '9' {
			continue
		}
		if r == '_' || r == '-' || r == ':' {
			continue
		}
		return false
	}
	return true
}

func positiveLeakAtMS(atMS int64) int64 {
	if atMS <= 0 {
		return 1
	}
	return atMS
}

func safeLeakToken(s string, limit int, fallback string) string {
	s = strings.TrimSpace(s)
	if boundedToken(s, limit) {
		return s
	}
	if s == "" {
		return fallback
	}
	return digest([]string{s})
}

func safeLeakReason(s, fallback string) string {
	s = strings.TrimSpace(s)
	if reasonToken(s) {
		return s
	}
	return fallback
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func renderCounts(m map[string]int) string {
	if len(m) == 0 {
		return "-"
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s:%d", k, m[k]))
	}
	return strings.Join(parts, ",")
}

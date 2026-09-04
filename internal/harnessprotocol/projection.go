package harnessprotocol

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

// Projection is the materialized view of an ordered sequence of harness envelopes for a single run.
type Projection struct {
	RunID     string                                `json:"run_id"`
	Cursor    uint64                                `json:"cursor"`
	Status    string                                `json:"status,omitempty"`
	Messages  map[string]string                     `json:"messages,omitempty"`
	UI        []harnesskit.UIBlockPayload           `json:"ui,omitempty"`
	Tools     map[string]harnesskit.ToolPayload     `json:"tools,omitempty"`
	Plan      []harnesskit.PlanStep                 `json:"plan,omitempty"`
	Approvals map[string]harnesskit.ApprovalPayload `json:"approvals,omitempty"`
	Artifacts []harnesskit.ArtifactPayload          `json:"artifacts,omitempty"`
	Errors    []harnesskit.ErrorPayload             `json:"errors,omitempty"`
	Usage     harnesskit.UsagePayload               `json:"usage"`
}

// Project folds an ordered sequence of harness envelopes into a single run projection,
// validating sequence monotonicity and event ordering.
func Project(events []harnesskit.Envelope) (Projection, error) {
	p := Projection{Messages: map[string]string{}, Tools: map[string]harnesskit.ToolPayload{}, Approvals: map[string]harnesskit.ApprovalPayload{}}
	for _, e := range events {
		if err := e.Validate(); err != nil {
			return p, err
		}
		if p.RunID == "" {
			p.RunID = e.RunID
		}
		if e.RunID != p.RunID || e.Sequence != p.Cursor+1 {
			return p, fmt.Errorf("event ordering violation at %d", e.Sequence)
		}
		p.Cursor = e.Sequence
		if !e.Known() {
			continue
		}
		switch e.Type {
		case harnesskit.EventRunStarted, harnesskit.EventRunCompleted, harnesskit.EventRunCanceled:
			var v harnesskit.RunPayload
			if err := e.DecodePayload(&v); err != nil {
				return p, err
			}
			p.Status = v.Status
		case harnesskit.EventMessageStarted, harnesskit.EventMessageDelta, harnesskit.EventMessageCompleted:
			var v harnesskit.MessagePayload
			if err := e.DecodePayload(&v); err != nil {
				return p, err
			}
			p.Messages[v.MessageID] += v.Text
		case harnesskit.EventUIBlock:
			var v harnesskit.UIBlockPayload
			if err := e.DecodePayload(&v); err != nil {
				return p, err
			}
			p.UI = append(p.UI, v)
		case harnesskit.EventToolStarted, harnesskit.EventToolCompleted:
			var v harnesskit.ToolPayload
			if err := e.DecodePayload(&v); err != nil {
				return p, err
			}
			p.Tools[v.CallID] = v
		case harnesskit.EventPlanUpdated:
			var v harnesskit.PlanPayload
			if err := e.DecodePayload(&v); err != nil {
				return p, err
			}
			p.Plan = v.Steps
		case harnesskit.EventApprovalRequested, harnesskit.EventApprovalResolved:
			var v harnesskit.ApprovalPayload
			if err := e.DecodePayload(&v); err != nil {
				return p, err
			}
			p.Approvals[v.ApprovalID] = v
		case harnesskit.EventArtifactPublished:
			var v harnesskit.ArtifactPayload
			if err := e.DecodePayload(&v); err != nil {
				return p, err
			}
			p.Artifacts = append(p.Artifacts, v)
		case harnesskit.EventError:
			var v harnesskit.ErrorPayload
			if err := e.DecodePayload(&v); err != nil {
				return p, err
			}
			p.Errors = append(p.Errors, v)
		case harnesskit.EventUsage:
			var v harnesskit.UsagePayload
			if err := e.DecodePayload(&v); err != nil {
				return p, err
			}
			p.Usage.InputTokens += v.InputTokens
			p.Usage.OutputTokens += v.OutputTokens
			p.Usage.CostMicros += v.CostMicros
		}
	}
	return p, nil
}

// CLIText formats a Projection into a plain-text line-oriented summary suitable for CLI output.
func CLIText(p Projection) string {
	var b strings.Builder
	fmt.Fprintf(&b, "run %s %s\n", p.RunID, p.Status)
	for _, id := range sortedKeys(p.Messages) {
		fmt.Fprintf(&b, "message %s: %s\n", id, p.Messages[id])
	}
	for _, id := range sortedKeys(p.Tools) {
		t := p.Tools[id]
		fmt.Fprintf(&b, "tool %s %s %s\n", id, t.Name, t.Status)
	}
	fmt.Fprintf(&b, "usage %d/%d\n", p.Usage.InputTokens, p.Usage.OutputTokens)
	return b.String()
}

// TUIText formats a Projection into structured, bracketed sections suitable for terminal UI displays.
func TUIText(p Projection) string {
	var b strings.Builder
	fmt.Fprintf(&b, "[RUN] %s | %s\n", p.RunID, p.Status)
	for _, id := range sortedKeys(p.Messages) {
		fmt.Fprintf(&b, "[MESSAGE %s] %s\n", id, p.Messages[id])
	}
	for _, id := range sortedKeys(p.Tools) {
		t := p.Tools[id]
		fmt.Fprintf(&b, "[TOOL %s] %s (%s)\n", id, t.Name, t.Status)
	}
	fmt.Fprintf(&b, "[TOKENS] in=%d out=%d\n", p.Usage.InputTokens, p.Usage.OutputTokens)
	return b.String()
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for key := range m {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

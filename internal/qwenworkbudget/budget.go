// Package qwenworkbudget turns canonical trajectory audit rollups into
// attributable Qwen campaign admission and continuation receipts.
package qwenworkbudget

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

// Boundary identifies the campaign boundary being evaluated.
type Boundary string

const (
	// BoundaryLaunch marks the start of a new Qwen execution campaign before work commences.
	BoundaryLaunch Boundary = "launch"
	// BoundaryContinuation marks an intermediate checkpoint during an ongoing Qwen campaign run.
	BoundaryContinuation Boundary = "continuation"
)

// Contributor attributes canonical usage to one transcript.
type Contributor struct {
	Source          string
	TranscriptID    string
	InputTokens     uint64
	CacheReadTokens uint64
	OutputTokens    uint64
}

// Policy adds a ratio budget to the existing token and witness policy. A zero
// ratio disables that limit; Enforce remains false by default (observe-only).
type Policy struct {
	trajectory.QwenAmplificationPolicy
	MaxInputPerOutput float64
}

// Packet contains the canonical audit rollup available at a campaign boundary.
// A nil Audit means usage is unavailable; evidence is never truncated or filled in.
type Packet struct {
	Boundary        Boundary
	Engine          string
	Audit           *trajectory.AuditResult
	UsefulWitnesses uint64
	OverrideReason  string
}

// Receipt is the typed, attributable boundary decision.
type Receipt struct {
	Boundary       Boundary
	Engine         string
	Action         trajectory.QwenAmplificationAction
	Eligible       bool
	Reason         trajectory.QwenAmplificationNonEligibleReason
	Usage          *trajectory.QwenCanonicalUsage
	Contributors   []Contributor
	InputPerOutput *float64
	Override       *trajectory.QwenAmplificationOverrideReceipt
}

// Invariant: Qwen work budget evaluations are fail-closed and bounded.
// Missing audits, invalid engines, or unparseable usage records immediately yield
// non-eligible receipts rather than defaulting to permissive continuation.
// Guard: Campaign continuation requires explicit observed witness and ratio satisfaction.
//
// Evaluate aggregates only Qwen transcript rows from the canonical audit and
// evaluates them without mutating the packet or silently reducing its evidence.
func (p Policy) Evaluate(packet Packet) Receipt {
	receipt := Receipt{Boundary: packet.Boundary, Engine: packet.Engine, Action: trajectory.QwenAmplificationObserve}
	usage, contributors, ok := canonicalUsage(packet.Audit, packet.UsefulWitnesses)
	if !ok {
		decision := p.QwenAmplificationPolicy.Decide(packet.Engine, nil, packet.OverrideReason)
		receipt.Action = decision.Action
		receipt.Eligible = decision.Eligible
		receipt.Reason = decision.NonEligibleReason
		receipt.Override = decision.Override
		return receipt
	}

	decision := p.QwenAmplificationPolicy.Decide(packet.Engine, usage, packet.OverrideReason)
	receipt.Action = decision.Action
	receipt.Eligible = decision.Eligible
	receipt.Reason = decision.NonEligibleReason
	receipt.Usage = usage
	receipt.Contributors = contributors
	receipt.Override = decision.Override
	if !decision.Eligible {
		return receipt
	}
	input := usage.InputTokens + usage.CacheReadTokens
	breached := false
	if usage.OutputTokens > 0 {
		ratio := float64(input) / float64(usage.OutputTokens)
		receipt.InputPerOutput = &ratio
		breached = p.MaxInputPerOutput != 0 && ratio > p.MaxInputPerOutput
	} else {
		// Positive input with no output has unbounded amplification. Keep the
		// ratio absent rather than emitting a non-JSON infinity.
		breached = p.MaxInputPerOutput != 0 && input > 0
	}
	if !breached {
		return receipt
	}

	receipt.Action = trajectory.QwenAmplificationAlert
	if packet.OverrideReason != "" {
		receipt.Override = &trajectory.QwenAmplificationOverrideReceipt{Reason: packet.OverrideReason}
	} else if p.Enforce {
		receipt.Action = trajectory.QwenAmplificationHold
	}
	return receipt
}

func canonicalUsage(audit *trajectory.AuditResult, witnesses uint64) (*trajectory.QwenCanonicalUsage, []Contributor, bool) {
	if audit == nil {
		return nil, nil, false
	}
	usage := &trajectory.QwenCanonicalUsage{UsefulWitnesses: witnesses}
	contributors := make([]Contributor, 0)
	for _, row := range audit.Transcripts {
		if !isQwen(row.Models) || row.UsageRecords == 0 || row.Tokens.InputTokens < 0 || row.Tokens.CacheCreateTokens < 0 || row.Tokens.CacheReadTokens < 0 || row.Tokens.OutputTokens < 0 {
			continue
		}
		contributor := Contributor{
			Source: row.Source, TranscriptID: row.TranscriptID,
			InputTokens:     uint64(row.Tokens.InputTokens + row.Tokens.CacheCreateTokens),
			CacheReadTokens: uint64(row.Tokens.CacheReadTokens),
			OutputTokens:    uint64(row.Tokens.OutputTokens),
		}
		contributors = append(contributors, contributor)
		usage.InputTokens += contributor.InputTokens
		usage.CacheReadTokens += contributor.CacheReadTokens
		usage.OutputTokens += contributor.OutputTokens
	}
	if len(contributors) == 0 {
		return nil, nil, false
	}
	return usage, contributors, true
}

func isQwen(models []string) bool {
	for _, model := range models {
		if strings.Contains(strings.ToLower(model), "qwen") {
			return true
		}
	}
	return false
}

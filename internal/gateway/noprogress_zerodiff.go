package gateway

import "strings"

const (
	NoProgressContinue          = "continue"
	NoProgressOutputRepeat      = "output_repeat"
	NoProgressWorktreeUnchanged = "worktree_unchanged"
)

// NoProgressSample is one completed tool-loop turn. OutputDigest identifies the
// model-visible result while WorktreeDigest identifies the independently read
// back workspace state after the call. A changing output is not progress when
// the workspace effect remains unchanged (#5893).
type NoProgressSample struct {
	Tool           string `json:"tool,omitempty"`
	OutputDigest   string `json:"output_digest,omitempty"`
	WorktreeDigest string `json:"worktree_digest,omitempty"`
}

// NoProgressVerdict is a typed, machine-readable loop decision. Retryable is
// false only after the bounded threshold has been reached.
type NoProgressVerdict struct {
	Kind             string `json:"kind"`
	Tool             string `json:"tool,omitempty"`
	ConsecutiveTurns int    `json:"consecutive_turns"`
	Threshold        int    `json:"threshold"`
	Retryable        bool   `json:"retryable"`
}

// WorktreeNoProgress tracks consecutive turns whose independently witnessed
// worktree state does not change. It also preserves the older repeated-output
// signal as a more specific reason when both digests repeat.
type WorktreeNoProgress struct {
	threshold int
	previous  NoProgressSample
	seen      bool
	count     int
}

func NewWorktreeNoProgress(threshold int) *WorktreeNoProgress {
	if threshold < 1 {
		threshold = 3
	}
	return &WorktreeNoProgress{threshold: threshold}
}

func (d *WorktreeNoProgress) Observe(s NoProgressSample) NoProgressVerdict {
	s.Tool = strings.TrimSpace(s.Tool)
	kind := NoProgressContinue
	if d.seen && s.WorktreeDigest != "" && s.WorktreeDigest == d.previous.WorktreeDigest {
		d.count++
		kind = NoProgressWorktreeUnchanged
		if s.OutputDigest != "" && s.OutputDigest == d.previous.OutputDigest {
			kind = NoProgressOutputRepeat
		}
	} else {
		d.count = 0
	}
	d.previous, d.seen = s, true
	return NoProgressVerdict{
		Kind: kind, Tool: s.Tool, ConsecutiveTurns: d.count,
		Threshold: d.threshold, Retryable: d.count < d.threshold,
	}
}

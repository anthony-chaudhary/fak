package dispatchpost

import (
	"fmt"
	"strings"
	"time"
)

// Result is the outcome of one background dispatch run, decoupled from how it was
// produced. `fak loop run` fills it on child exit; the renderer (Text/Blocks) has a
// single input shape — the same pattern as benchpost.Post / scoreboard.Update.
//
// The git fields are the WITNESS. HeadBefore is `git rev-parse HEAD` captured before
// the dispatch started; HeadAfter is captured after it ended; Commits is the
// `git log --oneline HeadBefore..HeadAfter` delta — the commits the run actually
// landed. A run that exits 0 with an empty Commits delta did NOT ship code, and the
// card says so. Self-reported success (exit code) and witnessed progress (the delta)
// are rendered as two distinct facts, never conflated.
type Result struct {
	LoopID     string   // the loop this run belongs to (e.g. "nightly-fix")
	RunID      string   // this run's id
	ExitCode   int      // child process exit code (self-reported success signal)
	DurationMS int64    // wall-clock the dispatch ran
	Command    string   // the dispatched command (base name), for context
	HeadBefore string   // git HEAD before the run (short sha), "" if unknown
	HeadAfter  string   // git HEAD after the run (short sha), "" if unknown
	Commits    []string // `git log --oneline` subjects landed by the run (the witness)
	Source     string   // who posted: "cron" | "agent" | hostname (optional)
}

// Shipped reports whether the run landed at least one commit — the witnessed-progress
// bit, independent of the exit code. A green exit with no commit is NOT shipped.
func (r Result) Shipped() bool { return len(r.Commits) > 0 }

// emoji picks the leading glyph from the two orthogonal facts: a failed dispatch is
// red regardless of commits; a green dispatch that shipped is a check; a green
// dispatch that landed nothing is a neutral "ran, no change" glyph so a no-op run is
// visibly distinct from a landed one.
func (r Result) emoji() string {
	switch {
	case r.ExitCode != 0:
		return ":red_circle:"
	case r.Shipped():
		return ":white_check_mark:"
	default:
		return ":white_circle:"
	}
}

// title is the headline: the loop id and a one-word outcome.
func (r Result) title() string {
	outcome := "ok"
	switch {
	case r.ExitCode != 0:
		outcome = fmt.Sprintf("FAILED (exit %d)", r.ExitCode)
	case r.Shipped():
		outcome = "shipped"
	default:
		outcome = "ran, no commit"
	}
	loop := r.LoopID
	if loop == "" {
		loop = "dispatch"
	}
	return fmt.Sprintf("dispatch %s — %s", loop, outcome)
}

// lead is the one-line summary under the title: duration, command, and the HEAD delta.
func (r Result) lead() string {
	var parts []string
	if d := time.Duration(r.DurationMS) * time.Millisecond; d > 0 {
		parts = append(parts, "ran "+humaniseDuration(d))
	}
	if r.Command != "" {
		parts = append(parts, "`"+r.Command+"`")
	}
	switch {
	case r.HeadBefore != "" && r.HeadAfter != "" && r.HeadBefore != r.HeadAfter:
		parts = append(parts, fmt.Sprintf("HEAD %s→%s", r.HeadBefore, r.HeadAfter))
	case r.HeadBefore != "" && r.HeadAfter != "" && r.HeadBefore == r.HeadAfter:
		parts = append(parts, fmt.Sprintf("HEAD unchanged at %s", r.HeadAfter))
	}
	if r.RunID != "" {
		parts = append(parts, "run `"+r.RunID+"`")
	}
	return strings.Join(parts, " · ")
}

// lines is the body: one line per landed commit (the witness), or a single honest
// "no commit landed" line when the delta is empty.
func (r Result) lines() []string {
	if !r.Shipped() {
		return []string{"_no commit landed — the dispatch produced no git change (witnessed via HEAD delta)_"}
	}
	out := make([]string, 0, len(r.Commits))
	for _, c := range r.Commits {
		out = append(out, c)
	}
	return out
}

// Text renders the plain-text fallback — what Slack shows in notifications and what
// tests and --dry-run assert on. Mirrors benchpost.Post.Text.
func (r Result) Text() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s *%s*", r.emoji(), r.title())
	if lead := r.lead(); lead != "" {
		fmt.Fprintf(&b, "\n%s", lead)
	}
	for _, ln := range r.lines() {
		fmt.Fprintf(&b, "\n• %s", ln)
	}
	if r.Source != "" {
		fmt.Fprintf(&b, "\n_posted by %s_", r.Source)
	}
	return b.String()
}

// Blocks renders the Block Kit payload, carrying the same facts as Text so a
// non-Block client loses nothing. Mirrors benchpost.Post.Blocks.
func (r Result) Blocks() []any {
	blocks := []any{
		map[string]any{
			"type": "section",
			"text": map[string]any{"type": "mrkdwn", "text": fmt.Sprintf("*%s %s*", r.emoji(), r.title())},
		},
	}
	if lead := r.lead(); lead != "" {
		blocks = append(blocks, map[string]any{
			"type": "section",
			"text": map[string]any{"type": "mrkdwn", "text": lead},
		})
	}
	if lines := r.lines(); len(lines) > 0 {
		body := "• " + strings.Join(lines, "\n• ")
		blocks = append(blocks, map[string]any{
			"type": "section",
			"text": map[string]any{"type": "mrkdwn", "text": body},
		})
	}
	if r.Source != "" {
		blocks = append(blocks, map[string]any{
			"type":     "context",
			"elements": []any{map[string]any{"type": "mrkdwn", "text": "posted by " + r.Source}},
		})
	}
	return blocks
}

// humaniseDuration renders a duration compactly (e.g. "2h3m", "45s", "320ms") for the
// lead line — a slow background dispatch can run for hours, so the unit scales.
func humaniseDuration(d time.Duration) string {
	switch {
	case d >= time.Hour:
		h := d / time.Hour
		m := (d % time.Hour) / time.Minute
		return fmt.Sprintf("%dh%dm", h, m)
	case d >= time.Minute:
		m := d / time.Minute
		s := (d % time.Minute) / time.Second
		return fmt.Sprintf("%dm%ds", m, s)
	case d >= time.Second:
		return fmt.Sprintf("%ds", d/time.Second)
	default:
		return fmt.Sprintf("%dms", d.Milliseconds())
	}
}

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const scratchPreamble = `---
loop: issue-3453
witness: commit-audit
---
# Objective
Bound the goal scratch section.

# Plan
- [ ] cap the scratch
`

// TestCapGoalScratchBoundsSection is the pure-function witness: the scratch section is
// trimmed to its last `cap` entries while the preamble and header survive verbatim, and the
// single value any reader consumes — the newest entry — is preserved.
func TestCapGoalScratchBoundsSection(t *testing.T) {
	text := scratchPreamble + "\n# Scratch / last-refusal\n"
	for i := 0; i < 200; i++ {
		text += "- NOT_YET refusal " + itoaScratch(i) + "\n"
	}
	got := capGoalScratch(text, goalScratchCap)

	entries := scratchEntryLines(got)
	if len(entries) != goalScratchCap {
		t.Fatalf("scratch entries = %d, want cap %d", len(entries), goalScratchCap)
	}
	// The most recent entry must be the last one written, and the oldest kept must be
	// exactly cap-from-the-end — i.e. we kept the TAIL, not the head.
	if want := "- NOT_YET refusal 199"; entries[len(entries)-1] != want {
		t.Fatalf("newest entry = %q, want %q", entries[len(entries)-1], want)
	}
	if want := "- NOT_YET refusal " + itoaScratch(200-goalScratchCap); entries[0] != want {
		t.Fatalf("oldest kept entry = %q, want %q", entries[0], want)
	}
	if !strings.Contains(got, "# Objective") || !strings.Contains(got, "# Plan") {
		t.Fatalf("preamble lost after cap:\n%s", got)
	}
	if !strings.Contains(got, "# Scratch / last-refusal") {
		t.Fatalf("scratch header lost after cap:\n%s", got)
	}
	if last := lastLoopGoalScratchLine(got); last != "NOT_YET refusal 199" {
		t.Fatalf("lastLoopGoalScratchLine = %q, want newest", last)
	}
}

// TestCapGoalScratchNoopWithinCap: a section already within cap (or with no scratch header)
// is returned byte-for-byte unchanged, so the common short-run path pays no rewrite churn.
func TestCapGoalScratchNoopWithinCap(t *testing.T) {
	within := scratchPreamble + "\n# Scratch / last-refusal\n- one\n- two\n"
	if got := capGoalScratch(within, goalScratchCap); got != within {
		t.Fatalf("within-cap text mutated:\n%q", got)
	}
	if got := capGoalScratch(scratchPreamble, goalScratchCap); got != scratchPreamble {
		t.Fatalf("no-scratch-section text mutated:\n%q", got)
	}
	if got := capGoalScratch(within, 0); got != within {
		t.Fatalf("non-positive cap must be a no-op")
	}
}

// TestAppendGoalScratchStaysBounded drives far more turns than the cap through the real
// appendGoalScratch path and asserts the on-disk file stops growing with the turn count —
// the O(N^2)/unbounded-disk defect in #3453 — while the newest refusal stays readable.
func TestAppendGoalScratchStaysBounded(t *testing.T) {
	goal := filepath.Join(t.TempDir(), "GOAL.md")
	if err := os.WriteFile(goal, []byte(scratchPreamble), 0o644); err != nil {
		t.Fatal(err)
	}
	const turns = goalScratchCap * 4
	// Fixed-width payloads so a stable steady-state byte size is expected once the cap is
	// hit — any growth then is the unbounded-disk defect, not counter-digit width.
	payload := func(i int) string { return "NOT_YET turn " + pad4(i) }
	var sizeAtCap int
	for i := 0; i < turns; i++ {
		if err := appendGoalScratch(goal, payload(i)); err != nil {
			t.Fatalf("append turn %d: %v", i, err)
		}
		if i == goalScratchCap { // record the steady-state size once the cap is reached
			b, _ := os.ReadFile(goal)
			sizeAtCap = len(b)
		}
	}
	b, err := os.ReadFile(goal)
	if err != nil {
		t.Fatal(err)
	}
	// Bounded: after many more turns the file must not have grown past the steady-state
	// size reached at the cap (fixed-width entries, so equality is expected).
	if len(b) != sizeAtCap {
		t.Fatalf("goal file size changed after cap: %d != steady-state %d bytes", len(b), sizeAtCap)
	}
	if entries := scratchEntryLines(string(b)); len(entries) != goalScratchCap {
		t.Fatalf("on-disk scratch entries = %d, want cap %d", len(entries), goalScratchCap)
	}
	if !strings.Contains(string(b), payload(turns-1)) {
		t.Fatalf("newest refusal missing after cap:\n%s", string(b))
	}
}

// scratchEntryLines returns the non-empty lines under the last scratch header.
func scratchEntryLines(text string) []string {
	lines := strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
	header := -1
	for i, ln := range lines {
		l := strings.ToLower(strings.TrimSpace(ln))
		if strings.HasPrefix(l, "#") && strings.HasPrefix(strings.TrimSpace(strings.TrimLeft(l, "#")), "scratch") {
			header = i
		}
	}
	if header < 0 {
		return nil
	}
	var out []string
	for _, ln := range lines[header+1:] {
		if strings.TrimSpace(ln) != "" {
			out = append(out, ln)
		}
	}
	return out
}

// pad4 renders n as a fixed 4-char zero-padded decimal so scratch entries are equal width.
func pad4(n int) string {
	s := itoaScratch(n)
	for len(s) < 4 {
		s = "0" + s
	}
	return s
}

func itoaScratch(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

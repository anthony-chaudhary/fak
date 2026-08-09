package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/growthgate"
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

// TestAppendGoalScratchLogRetainsUnboundedHistory is the retention half of #3826: driving far
// more turns than goalScratchCap must leave EVERY refusal readable in the append-only sidecar
// — including the earliest ones the bounded GOAL.md section drops — while GOAL.md itself stays
// bounded so the loop driver's per-turn re-parse does not regress to #3453.
func TestAppendGoalScratchLogRetainsUnboundedHistory(t *testing.T) {
	goal := filepath.Join(t.TempDir(), "GOAL.md")
	if err := os.WriteFile(goal, []byte(scratchPreamble), 0o644); err != nil {
		t.Fatal(err)
	}
	const turns = goalScratchCap * 4 // 200 turns vs a 50-entry cap
	payload := func(i int) string { return "NOT_YET turn " + pad4(i) }
	for i := 0; i < turns; i++ {
		if err := appendGoalScratch(goal, payload(i)); err != nil {
			t.Fatalf("append turn %d: %v", i, err)
		}
	}

	b, err := os.ReadFile(goalScratchLogPath(goal))
	if err != nil {
		t.Fatalf("read scratch log: %v", err)
	}
	history := nonEmptyLines(string(b))
	if len(history) != turns {
		t.Fatalf("history entries = %d, want every one of %d turns", len(history), turns)
	}
	if len(history) <= goalScratchCap {
		t.Fatalf("history retained %d entries, want > cap %d (unbounded retention)", len(history), goalScratchCap)
	}
	// The oldest entry is precisely the one the bounded section discards — that is the
	// observability gap this ticket exists to close.
	if oldest := payload(0); !strings.Contains(history[0], oldest) {
		t.Fatalf("oldest history entry = %q, want it to carry %q", history[0], oldest)
	}
	if newest := payload(turns - 1); !strings.Contains(history[len(history)-1], newest) {
		t.Fatalf("newest history entry = %q, want it to carry %q", history[len(history)-1], newest)
	}
	if strings.Contains(string(b), payload(0)) == strings.Contains(string(b), payload(turns-1)) {
		// both true — assert explicitly that the dropped-from-GOAL.md range survives here.
		if !strings.Contains(string(b), payload(goalScratchCap/2)) {
			t.Fatalf("mid-history entry %q missing from sidecar", payload(goalScratchCap/2))
		}
	}
	// Every entry carries a timestamp, which is what makes the sidecar answer the "why did
	// this drive cycle the same plan item for hours" question the ticket names.
	if _, err := time.Parse(goalScratchLogStampLayout, strings.Fields(history[0])[0]); err != nil {
		t.Fatalf("history entry %q lacks a parsable %s stamp: %v", history[0], goalScratchLogStampLayout, err)
	}

	// No #3453 regression: the hot file the drive loop re-parses every turn is still capped.
	g, err := os.ReadFile(goal)
	if err != nil {
		t.Fatal(err)
	}
	if entries := scratchEntryLines(string(g)); len(entries) != goalScratchCap {
		t.Fatalf("GOAL.md scratch entries = %d, want it still bounded at %d", len(entries), goalScratchCap)
	}
	if strings.Contains(string(g), payload(0)) {
		t.Fatalf("GOAL.md unexpectedly retained the oldest entry — bounded section regressed")
	}
}

// TestAppendGoalScratchLogWriteCostIsConstant is the cost half of #3826. It asserts the
// append-only WRITE SIZE stays constant per turn — deliberately NOT that total file size is
// constant, which is the bounded-file invariant unbounded retention contradicts.
//
// What this proves: the bytes added per turn do not scale with the entries already retained,
// so N turns cost O(N) total rather than the O(N^2) whole-file rewrite #3453 fixed. It also
// pins the hot re-parsed surface (GOAL.md) to a constant steady-state size, so neither the
// write nor the read side of a long drive grows with the turn count.
func TestAppendGoalScratchLogWriteCostIsConstant(t *testing.T) {
	goal := filepath.Join(t.TempDir(), "GOAL.md")
	if err := os.WriteFile(goal, []byte(scratchPreamble), 0o644); err != nil {
		t.Fatal(err)
	}
	logPath := goalScratchLogPath(goal)
	const turns = goalScratchCap * 4
	payload := func(i int) string { return "NOT_YET turn " + pad4(i) } // fixed width

	size := func(p string) int64 {
		fi, err := os.Stat(p)
		if err != nil {
			return 0
		}
		return fi.Size()
	}

	var firstDelta, goalSteady int64
	for i := 0; i < turns; i++ {
		before := size(logPath)
		if err := appendGoalScratch(goal, payload(i)); err != nil {
			t.Fatalf("append turn %d: %v", i, err)
		}
		delta := size(logPath) - before
		switch {
		case i == 0:
			firstDelta = delta
		case delta != firstDelta:
			// A read-modify-rewrite of the history would make this grow with i.
			t.Fatalf("turn %d wrote %d bytes, want the constant %d of turn 0 (append-only)", i, delta, firstDelta)
		}
		if i == goalScratchCap {
			goalSteady = size(goal)
		}
		if i > goalScratchCap && size(goal) != goalSteady {
			t.Fatalf("turn %d: GOAL.md = %d bytes, want steady-state %d (bounded re-parse)", i, size(goal), goalSteady)
		}
	}
	if firstDelta <= 0 {
		t.Fatalf("per-turn append wrote %d bytes", firstDelta)
	}
	// Retention is linear in turns while the per-turn cost is flat — the point of the split.
	if got, want := size(logPath), firstDelta*int64(turns); got != want {
		t.Fatalf("history = %d bytes after %d turns, want %d (linear total, constant per-turn)", got, turns, want)
	}
}

// TestRotateGoalScratchLogSealsWithoutLoss: the sidecar is append-only UNTIL ROTATED, so an
// oversized active segment is sealed to "<path>.NNN" and appends continue in a fresh file.
// Rotation must bound the HOT file without dropping history — sealed + active together still
// hold every entry (the #3287 / #3455 tension the ticket flags, resolved by sealing not
// truncating).
func TestRotateGoalScratchLogSealsWithoutLoss(t *testing.T) {
	goal := filepath.Join(t.TempDir(), "GOAL.md")
	if err := os.WriteFile(goal, []byte(scratchPreamble), 0o644); err != nil {
		t.Fatal(err)
	}
	logPath := goalScratchLogPath(goal)
	const (
		turns  = 40
		rotate = int64(200) // tiny, so several rotations happen inside the run
	)
	for i := 0; i < turns; i++ {
		appendGoalScratchLog(goal, "NOT_YET turn "+pad4(i), rotate)
	}

	// Sealed segments infix the index before the extension ("GOAL.scratch.001.log"), so the
	// glob is stem-anchored; the ACTIVE file does not match it.
	segments, err := filepath.Glob(strings.TrimSuffix(logPath, ".log") + ".*.log")
	if err != nil {
		t.Fatal(err)
	}
	if len(segments) == 0 {
		t.Fatalf("no sealed segment beside %s — rotation never fired", logPath)
	}
	total := 0
	for _, p := range append(segments, logPath) {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatalf("read %s: %v", p, err)
		}
		total += len(nonEmptyLines(string(b)))
	}
	if total != turns {
		t.Fatalf("sealed+active entries = %d, want all %d turns (rotation must not drop history)", total, turns)
	}
	// The hot append target is what rotation exists to bound.
	fi, err := os.Stat(logPath)
	if err != nil {
		t.Fatal(err)
	}
	if fi.Size() >= rotate {
		t.Fatalf("active segment = %d bytes, want it held under the %d rotate threshold", fi.Size(), rotate)
	}
	// A non-positive threshold disables rotation entirely.
	before, _ := os.Stat(logPath)
	rotateGoalScratchLog(logPath, 0)
	after, err := os.Stat(logPath)
	if err != nil || after.Size() != before.Size() {
		t.Fatalf("non-positive rotate threshold must be a no-op")
	}
}

// TestGoalScratchLogSegmentsStayGrowthVisible closes the #3826 retention promise against the
// leak budget epics (#3287 / #3455): unbounded history is only acceptable while the growth
// ratchet can still SEE it. Rotation is where that nearly broke — growthgate's walk pre-filter
// admits a file by its ".jsonl"/".log"/".err" suffix, so sealing to "<path>.NNN" (the natural
// naming) would have made every sealed segment invisible to `fak growthgate`, and the sealed
// segments are exactly where the history piles up. This pins BOTH halves of the contract: a
// sealed segment is admitted by the census walk AND classified ClassLog, whose remedy
// ("rotate by size; reap when COLD") is the one that actually applies.
func TestGoalScratchLogSegmentsStayGrowthVisible(t *testing.T) {
	goal := filepath.Join(t.TempDir(), "GOAL.md")
	if err := os.WriteFile(goal, []byte(scratchPreamble), 0o644); err != nil {
		t.Fatal(err)
	}
	logPath := goalScratchLogPath(goal)
	const rotate = int64(200) // tiny, so several segments are sealed inside the run
	for i := 0; i < 40; i++ {
		appendGoalScratchLog(goal, "NOT_YET turn "+pad4(i), rotate)
	}

	segments, err := filepath.Glob(strings.TrimSuffix(logPath, ".log") + ".*.log")
	if err != nil {
		t.Fatal(err)
	}
	if len(segments) == 0 {
		t.Fatalf("no sealed segment beside %s — rotation never fired", logPath)
	}
	// The ACTIVE file has always been growth-visible; the regression risk is the sealed tail,
	// so assert over both together.
	for _, p := range append(segments, logPath) {
		if !isGrowthCandidate(p) {
			t.Fatalf("%s is not a growth candidate — the census walk skips it, so its bytes are unbudgeted", p)
		}
		if got := growthgate.ClassifyPath(p); got != growthgate.ClassLog {
			t.Fatalf("growthgate.ClassifyPath(%s) = %q, want %q (remedy %q must apply)",
				p, got, growthgate.ClassLog, growthgate.ClassLog.Remedy())
		}
	}
}

// TestGoalScratchLogPath pins the sidecar naming: the extension is replaced (not appended) so
// a goal file's history is the "GOAL.scratch.log" the ticket names, and an empty goal path
// yields no surface rather than a stray "./.scratch.log".
func TestGoalScratchLogPath(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{filepath.Join("d", "GOAL.md"), filepath.Join("d", "GOAL.scratch.log")},
		{"GOAL.md", "GOAL.scratch.log"},
		{"GOAL", "GOAL.scratch.log"},
		{"  ", ""},
		{"", ""},
	} {
		if got := goalScratchLogPath(tc.in); got != tc.want {
			t.Fatalf("goalScratchLogPath(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
	// An unconfigured goal path must not create anything.
	appendGoalScratchLog("", "NOT_YET orphan", goalScratchLogRotateBytes)
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

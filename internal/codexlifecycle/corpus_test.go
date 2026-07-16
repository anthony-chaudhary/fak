package codexlifecycle

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// meta renders a real session_meta line. cwd is taken as a plain OS path and
// JSON-escaped here: a Windows path embedded raw would make `\w` an invalid JSON
// escape and silently drop the whole record (the real store writes "C:\\work\\fak").
func meta(id, provider, ver, cwd string) string {
	esc := strings.ReplaceAll(cwd, `\`, `\\`)
	return `{"timestamp":"2026-06-10T16:00:00.000Z","type":"session_meta","payload":{"id":"` + id +
		`","model_provider":"` + provider + `","cli_version":"` + ver + `","cwd":"` + esc + `"}}`
}

func writeRollout(t *testing.T, dir, name string, mtime time.Time, lines ...string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	body := ""
	for _, l := range lines {
		body += l + "\n"
	}
	if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
		t.Fatalf("write %s: %v", p, err)
	}
	if err := os.Chtimes(p, mtime, mtime); err != nil {
		t.Fatalf("chtimes %s: %v", p, err)
	}
	return p
}

// Only the FIRST session_meta identifies a rollout: a subagent carries the PARENT's
// metadata further down, and a last-wins read would relabel the child as its parent.
func TestReadRollout_FirstMetaWins(t *testing.T) {
	body := meta("child", "fak", "0.144.4", `C:\work\fak`) + "\n" +
		started("2026-06-10T16:00:09.000Z", "A") + "\n" +
		meta("parent", "openai", "0.142.2", `C:\other`) + "\n" +
		complete("2026-06-10T16:00:10.000Z", "A") + "\n"
	m, ev, err := ReadRollout(strings.NewReader(body))
	if err != nil {
		t.Fatalf("ReadRollout: %v", err)
	}
	if m.RolloutID != "child" || m.Provider != "fak" || m.CLIVersion != "0.144.4" { //boundarylint:ignore CHANGE_DETECTOR_TEST — closed fixture contract
		t.Errorf("meta = %+v, want the FIRST (child/fak/0.144.4)", m)
	}
	if m.ProviderVersion() != "fak 0.144.4" {
		t.Errorf("ProviderVersion = %q", m.ProviderVersion())
	}
	if len(ev) != 2 {
		t.Errorf("events = %d, want 2", len(ev))
	}
}

// The scan's central contract: whatever the store contains, every unmatched start is
// TYPED after reconciliation — UnclassifiedAfter is zero while UnmatchedBefore keeps
// reporting the real gap population (classify, don't hide).
func TestScanCorpus_TypesEveryUnmatchedStart(t *testing.T) {
	dir := t.TempDir()
	now := time.Date(2026, 7, 16, 12, 0, 0, 0, time.UTC)
	stale := now.Add(-72 * time.Hour)

	// A mid-session gap (A never terminates; B completes) + a stale open final start.
	writeRollout(t, dir, "gap.jsonl", stale,
		meta("s1", "fak", "0.144.4", `C:\work\fak`),
		started("2026-06-10T16:00:09.000Z", "A"),
		started("2026-06-10T16:05:00.000Z", "B"),
		complete("2026-06-10T16:05:30.000Z", "B"),
		started("2026-06-10T16:06:00.000Z", "C")) // open at end, stale => process_death

	// A clean session on a different provider/version.
	writeRollout(t, dir, "clean.jsonl", stale,
		meta("s2", "openai", "0.142.2", `C:\work\fak`),
		started("2026-06-10T16:00:09.000Z", "X"),
		complete("2026-06-10T16:00:10.000Z", "X"))

	w, err := ScanCorpus(dir, ScanOptions{Now: now, FreshWithin: time.Hour})
	if err != nil {
		t.Fatalf("ScanCorpus: %v", err)
	}
	if w.Scanned != 2 {
		t.Fatalf("scanned = %d, want 2", w.Scanned)
	}
	// THE ACCEPTANCE CRITERION.
	if !w.AllStartsTyped || w.Totals.UnclassifiedAfter != 0 {
		t.Errorf("unclassified_after = %d, want 0", w.Totals.UnclassifiedAfter)
	}
	// The gaps are still REPORTED, not hidden: A superseded, C dead, X complete.
	if w.Totals.Superseded != 1 || w.Totals.ProcessDeath != 1 || w.Totals.Complete != 2 {
		t.Errorf("totals = superseded %d / death %d / complete %d, want 1/1/2",
			w.Totals.Superseded, w.Totals.ProcessDeath, w.Totals.Complete)
	}
	if w.Totals.UnmatchedBefore != 2 { // A + C had no observed terminal
		t.Errorf("unmatched_before = %d, want 2", w.Totals.UnmatchedBefore)
	}
	if w.Totals.RolloutsWithGap != 1 {
		t.Errorf("rollouts_with_gap = %d, want 1", w.Totals.RolloutsWithGap)
	}
	// Rows are split by provider/version — the #4785 table axis.
	if got := w.ByProvider["fak 0.144.4"]; got == nil || got.Superseded != 1 {
		t.Errorf("fak row = %+v, want the superseded gap", got)
	}
	if got := w.ByProvider["openai 0.142.2"]; got == nil || got.Complete != 1 || got.UnmatchedBefore != 0 {
		t.Errorf("openai row = %+v, want a clean single completion", got)
	}
}

// Freshness — read from the file's own mtime — is what separates a dead writer from
// a live one on the SAME open-final-start bytes.
func TestScanCorpus_FreshnessSplitsLiveFromProcessDeath(t *testing.T) {
	now := time.Date(2026, 7, 16, 12, 0, 0, 0, time.UTC)
	for _, tc := range []struct {
		name  string
		mtime time.Time
		want  Outcome
	}{
		{"stale", now.Add(-72 * time.Hour), ProcessDeath},
		{"fresh", now.Add(-1 * time.Minute), Live},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			writeRollout(t, dir, "r.jsonl", tc.mtime,
				meta("s", "fak", "0.144.4", `C:\work\fak`),
				started("2026-06-10T16:00:09.000Z", "A"))
			w, err := ScanCorpus(dir, ScanOptions{Now: now, FreshWithin: time.Hour})
			if err != nil {
				t.Fatalf("ScanCorpus: %v", err)
			}
			got := w.ByProvider["fak 0.144.4"]
			if got == nil {
				t.Fatalf("no provider row — session_meta did not parse: %+v", w.ByProvider)
			}
			if tc.want == ProcessDeath && got.ProcessDeath != 1 {
				t.Errorf("stale rollout: process_death = %d, want 1 (%+v)", got.ProcessDeath, got)
			}
			if tc.want == Live && got.Live != 1 {
				t.Errorf("fresh rollout: live = %d, want 1 (%+v)", got.Live, got)
			}
		})
	}
}

// A mangled rollout is counted, never fatal: the store is append-only and a torn tail
// is normal, so refusing to report the rest of the corpus over one bad file would
// defeat the witness.
func TestScanCorpus_UnreadableIsCountedNotFatal(t *testing.T) {
	dir := t.TempDir()
	now := time.Now()
	writeRollout(t, dir, "ok.jsonl", now,
		meta("s", "fak", "0.144.4", `C:\work\fak`),
		started("2026-06-10T16:00:09.000Z", "A"),
		complete("2026-06-10T16:00:10.000Z", "A"))
	writeRollout(t, dir, "junk.jsonl", now, "not json at all", "{oh no")
	w, err := ScanCorpus(dir, ScanOptions{Now: now, FreshWithin: time.Hour})
	if err != nil {
		t.Fatalf("ScanCorpus must survive a mangled rollout: %v", err)
	}
	if w.Scanned != 1 || w.Totals.Complete != 1 {
		t.Errorf("scanned = %d complete = %d, want 1/1 (the junk file contributes no tasks)", w.Scanned, w.Totals.Complete)
	}
}

// CWD scoping reproduces #4785's evidence method (restrict to one repo's sessions).
func TestScanCorpus_CWDFilter(t *testing.T) {
	dir := t.TempDir()
	now := time.Now()
	writeRollout(t, dir, "mine.jsonl", now,
		meta("s1", "fak", "0.144.4", `C:\work\fak`),
		started("2026-06-10T16:00:09.000Z", "A"), complete("2026-06-10T16:00:10.000Z", "A"))
	writeRollout(t, dir, "other.jsonl", now,
		meta("s2", "fak", "0.144.4", `C:\other\repo`),
		started("2026-06-10T16:00:09.000Z", "B"), complete("2026-06-10T16:00:10.000Z", "B"))
	w, err := ScanCorpus(dir, ScanOptions{Now: now, FreshWithin: time.Hour, CWD: `C:\work\fak`})
	if err != nil {
		t.Fatalf("ScanCorpus: %v", err)
	}
	if w.Scanned != 1 {
		t.Errorf("scanned = %d, want 1 (only the matching cwd)", w.Scanned)
	}
}

// TestCorpusReport_LiveStore is the #4785 CORPUS WITNESS. It folds the real local
// rollout store and asserts the acceptance criterion — zero unclassified starts after
// reconciliation — printing the before/after table by provider/version. It SKIPS when
// no store is present (CI, a fresh clone), so it is a witness where the evidence
// exists and never a false red where it does not.
func TestCorpusReport_LiveStore(t *testing.T) {
	root := os.Getenv("CODEX_HOME")
	if root == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			t.Skip("no home dir; cannot locate a Codex rollout store")
		}
		root = filepath.Join(home, ".codex")
	}
	sessions := filepath.Join(root, "sessions")
	if _, err := os.Stat(sessions); err != nil {
		t.Skipf("no Codex rollout store at %s — corpus witness needs a real store", sessions)
	}

	w, err := ScanCorpus(sessions, ScanOptions{FreshWithin: 2 * time.Hour})
	if err != nil {
		t.Fatalf("ScanCorpus(%s): %v", sessions, err)
	}
	if w.Scanned == 0 {
		t.Skipf("store at %s holds no task-bearing rollouts", sessions)
	}

	var b string
	b += fmt.Sprintf("\n#4785 corpus witness — root=%s scanned=%d unreadable=%d\n", sessions, w.Scanned, w.Unreadable)
	b += fmt.Sprintf("%-24s %8s %8s %11s %11s %8s %8s %8s\n",
		"provider/version", "sessions", "starts", "unmatched", "superseded", "death", "live", "unclass")
	for _, k := range w.ProviderVersions() {
		c := w.ByProvider[k]
		b += fmt.Sprintf("%-24s %8d %8d %11d %11d %8d %8d %8d\n",
			k, c.Rollouts, c.Starts, c.UnmatchedBefore, c.Superseded, c.ProcessDeath, c.Live, c.UnclassifiedAfter)
	}
	tt := w.Totals
	b += fmt.Sprintf("%-24s %8d %8d %11d %11d %8d %8d %8d\n",
		"TOTAL", tt.Rollouts, tt.Starts, tt.UnmatchedBefore, tt.Superseded, tt.ProcessDeath, tt.Live, tt.UnclassifiedAfter)
	b += fmt.Sprintf("orphans=%d reused=%d multiply_terminated=%d rollouts_with_gap=%d\n",
		tt.Orphans, tt.Reused, tt.MultiplyTerminated, tt.RolloutsWithGap)
	t.Log(b)

	// THE ACCEPTANCE CRITERION: after reconciliation nothing is unclassified.
	if !w.AllStartsTyped || tt.UnclassifiedAfter != 0 {
		t.Errorf("unclassified starts after reconciliation = %d, want 0", tt.UnclassifiedAfter)
	}
	// Sanity: the store really did exercise the reconciler.
	if tt.Starts == 0 {
		t.Error("witness folded no task starts — the scan is not exercising the corpus")
	}
}

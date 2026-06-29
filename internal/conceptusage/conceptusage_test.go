package conceptusage

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// writeJournal drops a .dos/<name> file under root with the given JSONL lines.
func writeJournal(t *testing.T, root, name string, lines ...string) {
	t.Helper()
	dir := filepath.Join(root, ".dos")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, name), []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
}

func disciplinedCommits(n int) func(string, int) []commit {
	return func(string, int) []commit {
		cs := make([]commit, n)
		for i := range cs {
			cs[i] = commit{
				subject: "feat(model): add resident expert quant path (fak model)",
				body:    "Signed-off-by: Dev <dev@example.com>",
			}
		}
		return cs
	}
}

// A healthy corpus — disciplined commits, real verify/improve syscalls, resolved
// recalls, lane acquires — greens with zero debt.
func TestBuild_HealthyCorpusGreens(t *testing.T) {
	root := t.TempDir()
	// 20 recall rows, but a strong proactive-witness share (8 verify + 4 improve) and
	// >=40% resolved recalls, plus lane acquires.
	var lines []string
	for i := 0; i < 8; i++ {
		lines = append(lines, `{"syscall":"verify","verdict":"SHIPPED"}`)
	}
	for i := 0; i < 4; i++ {
		lines = append(lines, `{"syscall":"improve","verdict":"KEEP"}`)
	}
	for i := 0; i < 20; i++ {
		v := "RECALL_UNVERIFIABLE"
		if i < 10 {
			v = "RECALL_FRESH"
		}
		lines = append(lines, `{"syscall":"memory_recall","verdict":"`+v+`"}`)
	}
	writeJournal(t, root, "verdict-journal.jsonl", lines...)
	writeJournal(t, root, "lane-journal.jsonl",
		`{"op":"ACQUIRE","lane":"model"}`,
		`{"op":"ENFORCE","lane":""}`,
		`{"op":"RELEASE","lane":"model"}`)

	p := Build(Options{Root: root, Now: time.Unix(1_700_000_000, 0).UTC(), gitLog: disciplinedCommits(50)})
	if !p.OK {
		t.Fatalf("expected healthy corpus to green, got debt=%v reason=%q", p.Corpus["conceptusage_debt"], p.Reason)
	}
	if anyInt(p.Corpus["conceptusage_debt"]) != 0 {
		t.Fatalf("expected 0 debt, got %v", p.Corpus["conceptusage_debt"])
	}
	if g := anyStr(p.Corpus["grade"]); g != "A" && g != "B" {
		t.Fatalf("expected A/B grade for healthy corpus, got %q (composite %v)", g, p.Corpus["score"])
	}
}

// A thin corpus — disciplined commits but recall-ONLY (no verify/improve, all
// UNVERIFIABLE) — reds on the witness axis, which is exactly the gap the 3x program
// targets. This is the keystone proof: the score cannot be green without real
// witnessing, only by editing the journal to contain verify syscalls.
func TestBuild_ThinWitnessReds(t *testing.T) {
	root := t.TempDir()
	var lines []string
	for i := 0; i < 30; i++ {
		lines = append(lines, `{"syscall":"memory_recall","verdict":"RECALL_UNVERIFIABLE"}`)
	}
	writeJournal(t, root, "verdict-journal.jsonl", lines...)

	p := Build(Options{Root: root, Now: time.Unix(1_700_000_000, 0).UTC(), gitLog: disciplinedCommits(50)})
	if p.OK {
		t.Fatalf("expected thin-witness corpus to red, but it greened: %q", p.Reason)
	}
	// The witness axis must be the thing that's red — usage discipline is fine.
	if uScore := anyInt(p.Corpus["usage_score"]); uScore < 90 {
		t.Fatalf("usage axis should stay strong (disciplined commits), got %d", uScore)
	}
	if wScore := anyInt(p.Corpus["witness_score"]); wScore >= 70 {
		t.Fatalf("witness axis should be weak with recall-only evidence, got %d", wScore)
	}
	if !strings.Contains(p.Reason, "verify_syscall_used") && !strings.Contains(p.Reason, "witness_share") {
		t.Fatalf("expected a witness-axis defect to lead the finding, got %q", p.Reason)
	}
}

// Driving the witness axis up (adding verify syscalls) must measurably lift the
// witness score and cut debt — proving the score responds to MORE concept usage, not
// to data editing. This is the 3x mechanism under test.
func TestCompare_WitnessLiftIsDetected(t *testing.T) {
	root := t.TempDir()
	// Baseline: recall-only.
	var thin []string
	for i := 0; i < 30; i++ {
		thin = append(thin, `{"syscall":"memory_recall","verdict":"RECALL_UNVERIFIABLE"}`)
	}
	writeJournal(t, root, "verdict-journal.jsonl", thin...)
	base := Build(Options{Root: root, Now: time.Unix(1_700_000_000, 0).UTC(), gitLog: disciplinedCommits(50)})
	baseMap := map[string]any{"corpus": base.Corpus}

	// After: same recalls, but now with real verify/improve witnessing added.
	after := append([]string{}, thin...)
	for i := 0; i < 12; i++ {
		after = append(after, `{"syscall":"verify","verdict":"SHIPPED"}`)
	}
	for i := 0; i < 12; i++ {
		after = append(after, `{"syscall":"memory_recall","verdict":"RECALL_FRESH"}`)
	}
	writeJournal(t, root, "verdict-journal.jsonl", after...)
	improved := Build(Options{Root: root, Now: time.Unix(1_700_000_000, 0).UTC(), gitLog: disciplinedCommits(50)})

	if anyInt(improved.Corpus["witness_score"]) <= anyInt(base.Corpus["witness_score"]) {
		t.Fatalf("witness score should rise after adding verify syscalls: %v -> %v",
			base.Corpus["witness_score"], improved.Corpus["witness_score"])
	}
	cmp := Compare(improved, baseMap)
	if !strings.Contains(cmp, "VERDICT") {
		t.Fatalf("compare should render a verdict, got %q", cmp)
	}
}

// Missing journal must degrade to a HARD fail (journal_present), never a false pass —
// no evidence is not the same as healthy.
func TestBuild_NoJournalRedsNotFalsePass(t *testing.T) {
	root := t.TempDir()
	p := Build(Options{Root: root, Now: time.Unix(1_700_000_000, 0).UTC(), gitLog: disciplinedCommits(50)})
	if p.OK {
		t.Fatalf("expected no-journal tree to red (no witnessing evidence), got green: %q", p.Reason)
	}
}

// A malformed tail line (concurrent append) must not crash the scan; the good rows
// before it still count.
func TestScan_ToleratesMalformedTail(t *testing.T) {
	root := t.TempDir()
	writeJournal(t, root, "verdict-journal.jsonl",
		`{"syscall":"verify","verdict":"SHIPPED"}`,
		`{"syscall":"verify","verdict":"NOT_SHIPPED"}`,
		`{"syscall":"verify",`) // truncated/garbled
	var ev Evidence
	scanVerdictJournal(filepath.Join(root, ".dos", "verdict-journal.jsonl"), &ev)
	if ev.VerifySyscalls != 2 {
		t.Fatalf("expected 2 verify rows counted before the malformed tail, got %d", ev.VerifySyscalls)
	}
}

// Windowing proof: a long tail of OLD recalls followed by a recent burst of witnessing
// must register on the windowed share — the all-time ratio would stay pinned near zero,
// but the window reflects that the CURRENT dev loop witnesses. This is the metric-honesty
// fix that lets genuine recent witnessing move the number without becoming gameable.
func TestWindowedWitnessShare_RecentBurstRegisters(t *testing.T) {
	root := t.TempDir()
	var lines []string
	// 300 old passive recalls (history) ...
	for i := 0; i < 300; i++ {
		lines = append(lines, `{"syscall":"memory_recall","verdict":"RECALL_UNVERIFIABLE"}`)
	}
	// ... then a recent burst of 20 real witnessing decisions.
	for i := 0; i < 20; i++ {
		lines = append(lines, `{"syscall":"improve","verdict":"KEEP"}`)
	}
	writeJournal(t, root, "verdict-journal.jsonl", lines...)
	var ev Evidence
	scanVerdictJournal(filepath.Join(root, ".dos", "verdict-journal.jsonl"), &ev)
	// All-time share is ~6% (20/320); the windowed share over the last 50 must be far
	// higher because the recent tail is mostly witnessing.
	windowShare := pct(ev.WindowProactive, ev.WindowProactive+ev.WindowRecall)
	if windowShare < 15 {
		t.Fatalf("recent witnessing burst should register on the window, got %d%% (proactive=%d recall=%d)",
			windowShare, ev.WindowProactive, ev.WindowRecall)
	}
	allTime := pct(ev.VerifySyscalls+ev.ImproveCalls, ev.VerifySyscalls+ev.ImproveCalls+ev.RecallRows)
	if allTime >= windowShare {
		t.Fatalf("windowed share (%d%%) should exceed the all-time share (%d%%) for a recent burst", windowShare, allTime)
	}
}

// Noise-exclusion proof: passive RECALL_UNVERIFIABLE rows (the kernel auto-injecting a
// memory it could not check) are NOT dev decisions and must be excluded from the
// witness-share denominator — counting them as failed witnesses conflates kernel
// background chatter with dev behavior. A window of mostly-noise plus a few real
// witnessing decisions must score on the real decisions, not be drowned by the noise.
func TestWindowedWitnessShare_ExcludesPassiveNoise(t *testing.T) {
	root := t.TempDir()
	var lines []string
	// 40 passive UNVERIFIABLE auto-recalls (kernel noise) ...
	for i := 0; i < 40; i++ {
		lines = append(lines, `{"syscall":"memory_recall","verdict":"RECALL_UNVERIFIABLE"}`)
	}
	// ... plus 6 proactive witnesses and 4 resolved recalls (10 real decisions, 60% witnessed).
	for i := 0; i < 6; i++ {
		lines = append(lines, `{"syscall":"improve","verdict":"KEEP"}`)
	}
	for i := 0; i < 4; i++ {
		lines = append(lines, `{"syscall":"memory_recall","verdict":"RECALL_FRESH"}`)
	}
	writeJournal(t, root, "verdict-journal.jsonl", lines...)
	var ev Evidence
	scanVerdictJournal(filepath.Join(root, ".dos", "verdict-journal.jsonl"), &ev)
	if ev.WindowNoise != 40 {
		t.Fatalf("expected 40 passive-noise rows tracked separately, got %d", ev.WindowNoise)
	}
	dec := ev.WindowProactive + ev.WindowRecall // real decisions only
	if dec != 10 {
		t.Fatalf("expected 10 real dev decisions (noise excluded), got %d", dec)
	}
	share := pct(ev.WindowProactive, dec)
	if share < 50 {
		t.Fatalf("witness-share over REAL decisions should be ~60%%, got %d%% (noise must not dilute it)", share)
	}
}

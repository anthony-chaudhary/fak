package agent

// message_elide_corpus_replay_test.go — the #5254 REPLAYED BOUND: run the shipped cross-turn
// fold (ElideMessages, the mechanism 2ab350829c landed) over the SAME native-Codex rollout corpus
// the 2026-07-18 regrowth witness measured, and report what it would actually have collapsed.
//
// Why this test and not the DoD's own witness. #5254 item 3 asks for a compact-audit re-run
// showing REPEATED_TOOL_RESULT share and tool_result/* dup_bytes "materially reduced on NEW
// sessions". That is unmeasurable in one pass — the sessions have to accumulate first — and it is
// additionally blind by construction, because the fold rewrites what fak forwards upstream while
// the rollout row the audit mines is written by Codex before fak sees the turn and stays
// byte-identical (docs/notes/COMPACTION-REPEATED-TOOL-RESULT-DEDUP-2026-07-19.md).
//
// What IS measurable today: a counterfactual over windows that already exist. This drives the real
// session.ScanCompactRolloutReplay scorer with the real ElideMessages fold, and reports three
// numbers that a "materially reduced" headline would otherwise hide:
//
//   - the WIN: windows whose REPEATED_TOOL_RESULT the fold collapses, and bytes shed;
//   - the REACH CEILING: duplicate bytes whose earlier copy sits on the far side of the compaction
//     fire, which no within-wire fold can ever match (the issue's own re-entry case);
//   - the LOSS: spans that were not duplicates and were altered anyway — the false-positive rate,
//     which must be 0 for the win to mean anything.
//
// Corpus-gated on purpose: it is a measurement harness over a machine-local corpus, not a
// behavioural assertion, so it SKIPS unless FAK_COMPACT_CORPUS names the rollout root. Set
// FAK_COMPACT_CORPUS_LIMIT to pilot on a subset.
//
// Tier note: internal/agent is tier 4 and internal/session is tier 1, so this direction is legal
// (the reverse — session importing the elider — is not, which is why the scorer takes the
// mechanism as an injected func rather than calling it).

import (
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// corpusElideThreshold mirrors gateway.DefaultElideResultBytes (= DocumentedElideResultBytes =
// 16384), the value cmd/fak/guard.go and cmd/fak/serve.go actually arm in production. It is
// restated rather than imported because internal/gateway is tier 4 alongside this package and
// importing it here would add a dependency for a constant; if the shipped default ever moves, this
// replay is measuring the wrong wire.
const corpusElideThreshold = 16384

// corpusMessages renders one window's tool-result bodies as tool-role messages with DISTINCT call
// ids — matching the issue's finding that byte-identical spans arrive under different call ids, so
// a mechanism keyed on call id would find nothing and this one must key on content.
//
// Fidelity note, stated because it bounds the result: a real wire interleaves assistant/user
// messages between tool results, so the protected recent window (elideRecentKeepMsgs = 4 MESSAGES)
// covers fewer than 4 tool results in production. Handing it tool results only means 4 tool results
// are protected here, i.e. this replay folds LESS than production would. The reported win is
// therefore a conservative floor, not a ceiling.
func corpusMessages(bodies []string) []Message {
	msgs := make([]Message, len(bodies))
	for i, b := range bodies {
		msgs[i] = Message{Role: "tool", ToolCallID: "call_" + strconv.Itoa(i), Content: b}
	}
	return msgs
}

func corpusStrings(msgs []Message) []string {
	res := make([]string, len(msgs))
	for i := range msgs {
		res[i] = msgs[i].Content
	}
	return res
}

// dedupOnlyFold is the mechanism #5254 actually added: the cross-turn verbatim-span fold ALONE.
//
// This isolation is load-bearing, and the first corpus run is what proved it. ElideMessages runs
// TWO orthogonal levels — this dedup, and the pre-existing size-gated head+tail elision. Scoring
// the combined pass reported 740 of 2,906 non-duplicate spans as "altered", which reads as a
// catastrophic false-positive rate and is nothing of the kind: those are unique bodies over the
// 16 KB threshold being head+tail shrunk, which is intended, bounded, and predates this issue.
// Attributing them to the dedup would have been a fabricated defect in exactly the way a
// fabricated win would have been a fabricated success. The dedup level is size-INDEPENDENT, so it
// is the level whose false-positive rate must be zero.
func dedupOnlyFold(bodies []string) []string {
	msgs := corpusMessages(bodies)
	out, _, _ := dedupMessagesCrossTurn(msgs, len(msgs)-elideRecentKeepMsgs)
	return corpusStrings(out)
}

// shippedFullFold is the whole production pass (dedup THEN head+tail), reported for scale only.
// Its extra shed is head+tail elision, which is deliberately lossy-but-bounded and is NOT #5254's
// lane, so the loss columns are not asserted against it.
func shippedFullFold(bodies []string) []string {
	out, _ := ElideMessages(corpusMessages(bodies), corpusElideThreshold)
	return corpusStrings(out)
}

func TestCorpusReplayRepeatedToolResultFold(t *testing.T) {
	root := os.Getenv("FAK_COMPACT_CORPUS")
	if root == "" {
		t.Skip("set FAK_COMPACT_CORPUS to the Codex rollout root (e.g. ~/.codex/sessions) to replay the #5254 bound")
	}
	var paths []string
	if err := filepath.Walk(root, func(p string, fi os.FileInfo, err error) error {
		if err != nil || fi == nil || fi.IsDir() || !strings.HasSuffix(p, ".jsonl") {
			return nil //nolint:nilerr // an unreadable corpus entry is skipped, never fatal
		}
		paths = append(paths, p)
		return nil
	}); err != nil {
		t.Fatalf("walk corpus: %v", err)
	}
	sort.Strings(paths)
	if lim := os.Getenv("FAK_COMPACT_CORPUS_LIMIT"); lim != "" {
		if n, err := strconv.Atoi(lim); err == nil && n > 0 && n < len(paths) {
			paths = paths[len(paths)-n:]
		}
	}
	if len(paths) == 0 {
		t.Skipf("no .jsonl rollouts under %s", root)
	}

	sweep := func(fold session.RegrowthReplayFolder) (session.RegrowthReplayStat, int, int) {
		var total session.RegrowthReplayStat
		var fires, unreadable int
		for _, p := range paths {
			f, err := os.Open(p)
			if err != nil {
				unreadable++
				continue
			}
			fi, err := f.Stat()
			if err != nil {
				f.Close()
				unreadable++
				continue
			}
			rep, st, err := session.ScanCompactRolloutReplay(f, p, fi.Size(), session.RegrowthReplayOptions{
				Fold: fold,
				// crossTurnMinDupLines: a duplicate run shorter than this can never fold, so the scorer
				// reports how much of the in-window duplication is unreachable for that reason alone.
				MinDupLines: crossTurnMinDupLines,
			})
			f.Close()
			if err != nil {
				unreadable++
				continue
			}
			fires += rep.FireCount
			total.Add(st)
		}
		return total, fires, unreadable
	}

	total, fires, unreadable := sweep(dedupOnlyFold)
	full, _, _ := sweep(shippedFullFold)

	pct := func(a, b int64) float64 {
		if b == 0 {
			return 0
		}
		return 100 * float64(a) / float64(b)
	}
	t.Logf("#5254 REPLAYED BOUND over %s (fold = cross-turn dedup ONLY, the mechanism 2ab350829c added)", root)
	t.Logf("  corpus:      %d rollouts scanned (%d unreadable), %d compaction fires, %d post-fire windows carrying tool results",
		total.Rollouts, unreadable, fires, total.Windows)
	t.Logf("  tool results: %d rows, %d bytes (%d truncated by the 128 KB head bound)",
		total.ToolResultRows, total.ToolResultBytes, total.TruncatedRows)
	t.Logf("  BEFORE:      %d windows carry %s; %d dup rows, %d dup bytes",
		total.AnomalyWindowsBefore, session.AnomalyRepeatedToolResult, total.DupRowsBefore, total.DupBytesBefore)
	t.Logf("  REACH:       in-window %d rows / %d bytes (%.1f%% of dup bytes) | cross-fire %d rows / %d bytes (%.1f%%, UNREACHABLE by a within-wire fold)",
		total.InWindowDupRows, total.InWindowDupBytes, pct(total.InWindowDupBytes, total.DupBytesBefore),
		total.CrossFireDupRows, total.CrossFireDupBytes, pct(total.CrossFireDupBytes, total.DupBytesBefore))
	t.Logf("  under %d-line fold floor: %d in-window dup rows", crossTurnMinDupLines, total.DupRowsUnderLineFloor)
	t.Logf("  AFTER:       %d windows carry it (%d COLLAPSED); %d dup rows, %d dup bytes; %d rows folded, %d bytes shed",
		total.AnomalyWindowsAfter, total.WindowsCollapsed, total.DupRowsAfter, total.DupBytesAfter,
		total.FoldedRows, total.ShedBytes)
	t.Logf("  LOSS:        %d lines removed and relocated (still reachable earlier), %d lines DESTROYED (false positives)",
		total.RemovedLinesRelocated, total.RemovedLinesLost)
	t.Logf("  fold shape:  %d whole-body duplicate rows, %d partial (span-level) folds the audit's body-level dup accounting cannot see",
		total.WholeBodyDupRows, total.PartialFoldRows)
	t.Logf("  skipped:     %d windows over byte budget, %d shape errors", total.BudgetSkippedWindows, total.ShapeErrorWindows)
	t.Logf("  for scale, the FULL shipped pass (dedup + size-gated head+tail): %d rows folded, %d bytes shed, %d lines destroyed",
		full.FoldedRows, full.ShedBytes, full.RemovedLinesLost)
	t.Logf("  (head+tail is deliberately lossy-but-bounded and predates #5254, so its destroyed-line count is expected to be nonzero and is NOT asserted)")

	// The only hard assertions are the LOSS ones. The win is a measurement whose value is whatever
	// the corpus says; a false positive is a correctness bug at any magnitude, because it means the
	// shipped fold rewrote a span that never repeated.
	if total.RemovedLinesLost != 0 {
		t.Errorf("false positives: the shipped dedup destroyed %d lines that appear nowhere earlier in the window", total.RemovedLinesLost)
	}
	if total.ShapeErrorWindows != 0 {
		t.Errorf("shape errors: %d windows had their span count changed by the fold", total.ShapeErrorWindows)
	}
}

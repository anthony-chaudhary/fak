package toolproc

// zz_replaywitness_test.go — THROWAWAY (not committed): replays the real captured
// Codex rollout corpus through the shipped IngestRollout+ClassifyRepeats to produce
// the #4764 Witness part (A) — the distribution + avoidable-saving over real data.

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

func TestZZReplayRealCorpus(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping real-corpus replay-witness in -short mode")
	}
	root := os.Getenv("FAK_ROLLOUT_DIR")
	if root == "" {
		root = filepath.Join(os.Getenv("HOME"), ".codex", "sessions")
	}
	var files []string
	filepath.Walk(root, func(p string, info os.FileInfo, err error) error {
		if err == nil && info != nil && !info.IsDir() && filepath.Ext(p) == ".jsonl" {
			files = append(files, p)
		}
		return nil
	})
	if len(files) == 0 {
		t.Skip("no rollout corpus on this host")
	}
	// top-100 by size, mirroring the issue's audit.
	sort.Slice(files, func(i, j int) bool {
		fi, _ := os.Stat(files[i])
		fj, _ := os.Stat(files[j])
		return fi.Size() > fj.Size()
	})
	if len(files) > 100 {
		files = files[:100]
	}
	var all []CallRecord
	for _, f := range files {
		fh, err := os.Open(f)
		if err != nil {
			continue
		}
		all = append(all, IngestRollout(fh)...)
		fh.Close()
	}
	rep := ClassifyRepeats(all, RepeatConfig{})
	fmt.Printf("REPLAY files=%d records=%d groups=%d outputBytes=%d avoidableSpawns=%d avoidableInputBytes=%d\n",
		len(files), rep.Totals.Records, rep.Totals.Groups, rep.Totals.OutputBytes,
		rep.Totals.AvoidableSpawns, rep.Totals.AvoidableInputBytes)
	fmt.Printf("PER_CLASS %v\n", rep.Totals.PerClass)
	n := 0
	for _, g := range rep.Groups {
		if g.AvoidableSpawns == 0 {
			continue
		}
		fmt.Printf("TOP class=%s reuse=%s count=%d avoidSpawns=%d avoidBytes=%d canon=%q\n",
			g.Class, g.Reuse, g.Count, g.AvoidableSpawns, g.AvoidableInputBytes, g.Canonical)
		if n++; n >= 12 {
			break
		}
	}
}

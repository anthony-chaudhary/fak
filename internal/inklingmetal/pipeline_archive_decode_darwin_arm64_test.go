//go:build darwin && arm64 && cgo

package inklingmetal

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// decodeRun is one model load: pipeline stats after load, after the prompt
// prefill and after speculative decode, and the greedy tokens produced.
type decodeRun struct {
	load, prefill, decode PipelineStats
	ids                   []int32
	cycles                int
}

// speculativeArchiveRun opens the model, prefills a fixed prompt and decodes
// gen greedy tokens through fused propose-verify-commit cycles, the path the
// Go greedy loop takes when a draft is present, then closes it so the archive
// flushes.
func speculativeArchiveRun(t *testing.T, path string, gen int) decodeRun {
	t.Helper()
	m, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if !m.hasDraft {
		t.Skip("speculative decode needs the adjacent DFlash2 draft")
	}
	run := decodeRun{load: m.PipelineStats()}
	s, err := m.NewSession(128)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	prompt := []int32{371, 151451, 149879, 151194, 149884, 152717, 13, 149917, 151451, 149879, 153067, 149884, 154328, 13, 149917, 151451, 149879, 153630, 149884}
	next, err := s.PrefillID(prompt)
	if err != nil {
		t.Fatal(err)
	}
	run.prefill = m.PipelineStats()
	for {
		run.ids = append(run.ids, next)
		remaining := gen - len(run.ids)
		if remaining == 0 {
			break
		}
		depth := min(15, remaining-1)
		if depth == 0 {
			ids, err := s.BeginBlockIDs([]int32{next})
			if err == nil {
				err = s.Commit(1)
			}
			if err != nil {
				t.Fatal(err)
			}
			next = ids[0]
			continue
		}
		tree, ids, kept, err := s.ProposeVerifyCommit(next, depth, 4, depth+1, -1)
		if err != nil {
			t.Fatalf("cycle after %v: %v", run.ids, err)
		}
		if len(tree) == 0 || kept&1 == 0 {
			t.Fatalf("cycle kept no anchor: %d rows, mask %b", len(tree), kept)
		}
		run.cycles++
		last := 0
		for row := 1; row < len(tree) && len(run.ids) < gen; row++ {
			if kept&(1<<row) == 0 {
				continue
			}
			if int(tree[row][1]) != last || ids[last] != tree[row][0] {
				t.Fatalf("retained row %d is not the target's choice after row %d", row, last)
			}
			run.ids = append(run.ids, tree[row][0])
			last = row
		}
		if len(run.ids) >= gen {
			break
		}
		next = ids[last]
	}
	run.decode = m.PipelineStats()
	return run
}

func requireNoDecodeCompiles(t *testing.T, label string, r decodeRun) {
	t.Helper()
	t.Logf("%s load %+v prefill compiled %d decode compiled %d cycles %d greedy %v", label, r.load, r.prefill.Compiled, r.decode.Compiled, r.cycles, r.ids)
	if r.cycles == 0 {
		t.Fatalf("%s ran no speculative cycle", label)
	}
	if r.decode.Compiled != r.prefill.Compiled {
		t.Fatalf("%s compiled %d pipelines during %d-token speculative decode (%d after prefill, %d after); want none",
			label, r.decode.Compiled-r.prefill.Compiled, len(r.ids), r.prefill.Compiled, r.decode.Compiled)
	}
	if r.prefill.Compiled != r.load.Compiled {
		t.Errorf("%s compiled %d pipelines at load and %d after the %d-token prompt prefill; want no compiles after load",
			label, r.load.Compiled, r.prefill.Compiled, 19)
	}
}

// fak-test:runtime slow est=40s
func TestPipelineArchiveWarmsSpeculativeDecode(t *testing.T) {
	path := os.Getenv("FAK_INKLING2_TEST_GGUF")
	if path == "" {
		t.Skip("set FAK_INKLING2_TEST_GGUF to run on Apple GPU")
	}
	root := t.TempDir()
	t.Setenv(PipelineCacheEnv, root)
	const gen = 24

	cold := speculativeArchiveRun(t, path, gen)
	requireCold(t, "cold", cold.load)
	requireNoDecodeCompiles(t, "cold", cold)
	if cold.ids[0] != 182416 {
		t.Fatalf("cold prefill token %d, want 182416", cold.ids[0])
	}
	dir := archiveDir(t, root)

	warm := speculativeArchiveRun(t, path, gen)
	if !warm.load.Attached || warm.load.Archived == 0 || warm.load.ArchiveHits == 0 || warm.load.LoadWarmupMS != 0 {
		t.Fatalf("warm load stats %+v, want archived > 0, hits > 0, no load warmup", warm.load)
	}
	requireNoDecodeCompiles(t, "warm", warm)
	if !slices.Equal(warm.ids, cold.ids) {
		t.Fatalf("warm greedy %v, cold %v", warm.ids, cold.ids)
	}

	if err := os.WriteFile(filepath.Join(dir, "pipelines.metalarchive"), []byte("not a metal archive\x00\x01\x02"), 0o644); err != nil {
		t.Fatal(err)
	}
	corrupt := speculativeArchiveRun(t, path, gen)
	requireCold(t, "corrupt-archive", corrupt.load)
	requireNoDecodeCompiles(t, "corrupt-archive", corrupt)
	if !slices.Equal(corrupt.ids, cold.ids) {
		t.Fatalf("corrupt-archive greedy %v, cold %v", corrupt.ids, cold.ids)
	}
}

//go:build darwin && arm64 && cgo

package inklingmetal

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// archiveDir is the one inkling-<key> subdirectory under root, holding a
// non-empty archive and list.
func archiveDir(t *testing.T, root string) string {
	t.Helper()
	dirs, err := filepath.Glob(filepath.Join(root, "inkling-*"))
	if err != nil {
		t.Fatal(err)
	}
	if len(dirs) != 1 {
		t.Fatalf("archive subdirectories under %s = %v, want exactly one", root, dirs)
	}
	for _, name := range []string{"pipelines.metalarchive", "pipelines.list"} {
		info, err := os.Stat(filepath.Join(dirs[0], name))
		if err != nil {
			t.Fatal(err)
		}
		if info.Size() == 0 {
			t.Fatalf("%s is empty", name)
		}
	}
	return dirs[0]
}

// archiveRun opens the model, reads its pipeline stats and the greedy prefill
// token of a fixed prompt plus 8 single-row decode steps, then closes it so
// the archive flushes.
func archiveRun(t *testing.T, path string) (PipelineStats, []int32) {
	t.Helper()
	m, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	st := m.PipelineStats()
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
	ids := []int32{next}
	for len(ids) < 9 {
		got, err := s.BeginBlockIDs([]int32{next})
		if err != nil {
			t.Fatalf("decode step %d after %v: %v", len(ids), ids, err)
		}
		if err := s.Commit(1); err != nil {
			t.Fatal(err)
		}
		next = got[0]
		ids = append(ids, next)
	}
	return st, ids
}

func requireCold(t *testing.T, label string, st PipelineStats) {
	t.Helper()
	if !st.Attached || st.Archived != 0 || st.Compiled == 0 || st.LoadWarmupMS <= 0 {
		t.Fatalf("%s load stats %+v, want attached, nothing archived, compiled > 0, load warmup > 0", label, st)
	}
}

// fak-test:runtime slow est=30s
func TestPipelineArchiveWarmsSecondLoad(t *testing.T) {
	path := os.Getenv("FAK_INKLING2_TEST_GGUF")
	if path == "" {
		t.Skip("set FAK_INKLING2_TEST_GGUF to run on Apple GPU")
	}
	root := t.TempDir()
	t.Setenv(PipelineCacheEnv, root)

	cold, coldFirst := archiveRun(t, path)
	requireCold(t, "cold", cold)
	t.Logf("cold %+v greedy %v", cold, coldFirst)
	if coldFirst[0] != 182416 {
		t.Fatalf("cold prefill token %d, want 182416", coldFirst[0])
	}
	dir := archiveDir(t, root)

	warm, warmFirst := archiveRun(t, path)
	t.Logf("warm %+v greedy %v", warm, warmFirst)
	if !warm.Attached || warm.Archived == 0 || warm.ArchiveHits == 0 || warm.ArchiveHits < warm.Archived-1 || warm.Compiled > 2 {
		t.Fatalf("warm load stats %+v, want archived > 0, hits >= archived-1, compiled <= 2", warm)
	}
	if warm.LoadWarmupMS != 0 {
		t.Fatalf("warm load ran the warmup prefill: %+v", warm)
	}
	if !slices.Equal(warmFirst, coldFirst) {
		t.Fatalf("warm greedy %v, cold %v", warmFirst, coldFirst)
	}

	if err := os.WriteFile(filepath.Join(dir, "pipelines.metalarchive"), []byte("not a metal archive\x00\x01\x02"), 0o644); err != nil {
		t.Fatal(err)
	}
	corrupt, corruptFirst := archiveRun(t, path)
	requireCold(t, "corrupt-archive", corrupt)
	t.Logf("corrupt %+v greedy %v", corrupt, corruptFirst)
	if !slices.Equal(corruptFirst, coldFirst) {
		t.Fatalf("corrupt-archive greedy %v, cold %v", corruptFirst, coldFirst)
	}
	if got := archiveDir(t, root); got != dir {
		t.Fatalf("archive moved to %s from %s", got, dir)
	}
}

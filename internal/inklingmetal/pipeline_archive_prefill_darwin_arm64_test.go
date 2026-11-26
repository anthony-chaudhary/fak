//go:build darwin && arm64 && cgo

package inklingmetal

import (
	"os"
	"testing"
)

// A prefill chunk's kernels change with its row count, so each length here
// sits in, or on an edge of, a distinct row class the load warmup must cover.
var warmedPrefillLengths = []int{1, 2, 3, 5, 8, 9, 12, 16, 17, 19, 24, 25, 31, 32, 33, 40, 48, 49, 63, 64, 65, 100, 128, 129, 200, 257}

// fak-test:runtime slow est=60s
func TestPipelineArchiveWarmsEveryPrefillLength(t *testing.T) {
	path := os.Getenv("FAK_INKLING2_TEST_GGUF")
	if path == "" {
		t.Skip("set FAK_INKLING2_TEST_GGUF to run on Apple GPU")
	}
	t.Setenv(PipelineCacheEnv, t.TempDir())
	m, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	load := m.PipelineStats()
	requireCold(t, "cold", load)
	t.Logf("load %+v", load)

	ids := make([]int32, warmedPrefillLengths[len(warmedPrefillLengths)-1])
	for i := range ids {
		ids[i] = int32((uint64(i)*104729 + 17) % 150000)
	}
	for _, n := range warmedPrefillLengths {
		s, err := m.NewSession(512)
		if err != nil {
			t.Fatal(err)
		}
		_, err = s.PrefillID(ids[:n])
		s.Close()
		if err != nil {
			t.Fatalf("prefill %d tokens: %v", n, err)
		}
		if got := m.PipelineStats().Compiled; got != load.Compiled {
			t.Errorf("a %d-token prefill compiled %d pipelines after load", n, got-load.Compiled)
			load.Compiled = got
		}
	}
}

package bench

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/benchckpt"
	"github.com/anthony-chaudhary/fak/internal/turnbench"
)

// ckptOpts is the model-free fan-out config from fanrun_test.go with a write-ahead checkpoint
// path attached (issue #2382).
func ckptOpts(grid []int, ckpt string) FanrunOptions {
	o := smallOpts(turnbench.FanoutResearch, grid, 1)
	o.Checkpoint = ckpt
	return o
}

// TestFanrunCheckpointResumeParity is the #2382 item-4 witness for the fanrun N-grid: a sweep
// that stops partway leaves its completed cells on disk (write-ahead), and a resume over the
// full grid reuses those cells verbatim while measuring only the missing width — producing an
// artifact whose counter+geometry projection is identical to an uninterrupted run.
func TestFanrunCheckpointResumeParity(t *testing.T) {
	ckpt := filepath.Join(t.TempDir(), "fanrun.jsonl")

	// Phase A — a partial sweep that "crashes" after recording widths 1 and 4. A real process
	// would die mid-grid; here we stop the sweep after those cells are write-ahead appended,
	// which is the state a crash leaves behind: every completed cell is on disk before the next.
	repA, err := RunFanoutLiveResumable(context.Background(), ckptOpts([]int{1, 4}, ckpt))
	if err != nil {
		t.Fatalf("phase A: %v", err)
	}
	if len(repA.Cells) != 2 {
		t.Fatalf("phase A cells = %d, want 2", len(repA.Cells))
	}

	// The checkpoint is a write-ahead ledger: a header line plus one line per completed cell,
	// strictly fewer than the full grid's cells (nothing for N=16 has been measured yet).
	if lines := countJSONLines(t, ckpt); lines <= 0 || lines >= 4 {
		// header + N=1 + N=4 = 3 lines; a completed [1,4,16] grid would add the N=16 line.
		t.Fatalf("checkpoint lines = %d, want >0 and <4 (partial write-ahead survived the crash)", lines)
	}

	// Phase B — resume over the FULL grid. Widths 1 and 4 must be REUSED from the checkpoint
	// (byte-identical cells, including the recorded wall-clock), only N=16 freshly measured.
	repB, err := RunFanoutLiveResumable(context.Background(), ckptOpts([]int{1, 4, 16}, ckpt))
	if err != nil {
		t.Fatalf("phase B (resume): %v", err)
	}
	if len(repB.Cells) != 3 {
		t.Fatalf("resume cells = %d, want 3", len(repB.Cells))
	}

	// Reuse witness: the resumed N=1 and N=4 cells equal phase A's exactly. Identical *_ms
	// wall-clock proves they were loaded from the checkpoint, not silently re-measured.
	for i := range repA.Cells {
		if repA.Cells[i] != repB.Cells[i] {
			t.Errorf("cell %d not reused verbatim on resume:\n recorded=%+v\n resumed =%+v",
				i, repA.Cells[i], repB.Cells[i])
		}
	}

	// Parity witness: the resumed artifact's counter+geometry projection is identical to an
	// uninterrupted run over the same grid. The *_ms wall-clock halves may legitimately differ
	// (N=1 and N=4 carry phase A's measured times), which is the documented "modulo _ms" caveat.
	fresh := RunFanoutLive(context.Background(), smallOpts(turnbench.FanoutResearch, []int{1, 4, 16}, 1))
	if len(fresh.Cells) != len(repB.Cells) {
		t.Fatalf("uninterrupted cells = %d, want %d", len(fresh.Cells), len(repB.Cells))
	}
	for i := range fresh.Cells {
		if !sameCounterGeometry(fresh.Cells[i], repB.Cells[i]) {
			t.Errorf("N=%d counter+geometry differs resume vs uninterrupted:\n resume=%+v\n fresh =%+v",
				fresh.Cells[i].Agents, repB.Cells[i], fresh.Cells[i])
		}
	}
}

// TestFanrunResumeFingerprintMismatchRefuses is the #2382 item-5 witness at the executor tier:
// resuming a checkpoint with a different seed or profile refuses with a typed error instead of
// blending incompatible cells into one artifact.
func TestFanrunResumeFingerprintMismatchRefuses(t *testing.T) {
	ckpt := filepath.Join(t.TempDir(), "fanrun.jsonl")

	// Seed a checkpoint (research profile, seed 1).
	if _, err := RunFanoutLiveResumable(context.Background(), ckptOpts([]int{1, 4}, ckpt)); err != nil {
		t.Fatalf("seed run: %v", err)
	}

	// Resume with a different seed against the same checkpoint -> typed refusal, not a merge.
	changed := ckptOpts([]int{1, 4, 16}, ckpt)
	changed.Seed = 999
	if _, err := RunFanoutLiveResumable(context.Background(), changed); !errors.Is(err, benchckpt.ErrFingerprintMismatch) {
		t.Fatalf("resume with changed seed err = %v, want ErrFingerprintMismatch", err)
	}

	// A different sharing regime likewise refuses: the profile changes what every cell means.
	prof := ckptOpts([]int{1, 4}, ckpt)
	prof.Profile = turnbench.FanoutNoShare
	if _, err := RunFanoutLiveResumable(context.Background(), prof); !errors.Is(err, benchckpt.ErrFingerprintMismatch) {
		t.Fatalf("resume with changed profile err = %v, want ErrFingerprintMismatch", err)
	}
}

// sameCounterGeometry compares the reproducible counter+geometry projection of two cells,
// ignoring the *_ms wall-clock fields (which a resume inherits from the crashed run).
func sameCounterGeometry(a, b FanrunCell) bool {
	return a.Agents == b.Agents &&
		a.WaveHits == b.WaveHits &&
		a.CrossHits == b.CrossHits &&
		a.CrossHitsStable == b.CrossHitsStable &&
		a.VDSOLookups == b.VDSOLookups &&
		a.VDSOFills == b.VDSOFills &&
		a.TurnsTotal == b.TurnsTotal &&
		a.PromptTokens == b.PromptTokens &&
		a.CompletionTokens == b.CompletionTokens &&
		a.ToolErrorsTotal == b.ToolErrorsTotal &&
		a.TasksCompleted == b.TasksCompleted &&
		a.PrefixTokensElided == b.PrefixTokensElided
}

// countJSONLines counts the non-empty lines of a JSONL checkpoint.
func countJSONLines(t *testing.T, path string) int {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	n := 0
	for _, ln := range strings.Split(string(raw), "\n") {
		if strings.TrimSpace(ln) != "" {
			n++
		}
	}
	return n
}

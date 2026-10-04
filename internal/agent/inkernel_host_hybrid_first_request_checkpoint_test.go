package agent

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// fak#12742 (a): a short earlier request sharing only a template preamble teaches
// req1 a shallow structural boundary; req1's single checkpoint lands there instead
// of the deepest grid point, so the req2 sibling restores only the preamble.
// fak-test:runtime fast est=1s lane=default
func TestInKernelHostHybridFirstRequestDeepCheckpoint(t *testing.T) {
	cfg := tinyHybridCfg()
	const preamble = 5
	block := inKernelSnapshotCheckpointTokens
	common := synthIDs(cfg.VocabSize, 150, 12742)
	withSuffix := func(prefix []int, n int, seed uint64, first int) []int {
		s := synthIDs(cfg.VocabSize, n, seed)
		s[0] = first
		return append(append([]int(nil), prefix...), s...)
	}
	probe := withSuffix(common[:preamble], 20, 12743, (common[preamble]+1)%cfg.VocabSize)
	req1 := withSuffix(common, 30, 12744, 1)
	req2 := withSuffix(common, 30, 12745, 2)
	grid := ((len(req1) - 1) / block) * block
	if grid <= preamble || grid > len(common) {
		t.Fatalf("precondition grid=%d preamble=%d common=%d", grid, preamble, len(common))
	}

	p := reusePlanner(true, false, cfg)
	if !inKernelHostSnapshotReuse(p) {
		t.Fatal("precondition: host-session snapshot seam")
	}
	run := func(p *InKernelPlanner, ids []int) (tokens []int, cacheable, matched int, tier radixkv.SnapshotTier) {
		t.Helper()
		_, _, cacheable, matched, tier, _, _, _, err := p.generateReusedContextWithBias(
			context.Background(), ids, 4, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(id int) bool {
				tokens = append(tokens, id)
				return false
			})
		if err != nil {
			t.Fatalf("generate len=%d: %v", len(ids), err)
		}
		return
	}
	if _, _, m, _ := run(p, probe); m != 0 {
		t.Fatalf("probe reused %d, want 0", m)
	}
	if _, c, m, _ := run(p, req1); c != preamble || m != 0 {
		t.Fatalf("req1 cacheable=%d matched=%d, want %d/0", c, m, preamble)
	}
	warm, c2, m2, tier2 := run(p, req2)
	t.Logf("req2 cacheable=%d reused=%d tier=%s grid=%d", c2, m2, tier2, grid)
	if c2 != len(common) {
		t.Fatalf("req2 cacheable=%d, want %d", c2, len(common))
	}
	if m2 < grid || m2 > len(common) || tier2 != radixkv.SnapshotTierDeviceL1 {
		t.Fatalf("req2 reused=%d tier=%s, want in [%d,%d] device-l1 (shallow preamble=%d hid the first-request grid checkpoint)", m2, tier2, grid, len(common), preamble)
	}
	cold, _, cm, _ := run(reusePlanner(false, false, cfg), req2)
	if cm != 0 || !eqInts(warm, cold) {
		t.Fatalf("restored greedy %v != cold %v (cold matched=%d)", warm, cold, cm)
	}
}

// The single checkpoint goes to the exact shared boundary only when that boundary is
// at least the grid point or saves a whole block; a sub-block structural match keeps
// the deepest grid checkpoint (#12742 hardware arithmetic: 1922-token shared prefix,
// 1961-token prompt, a 5-token template preamble already in the tree).
// fak-test:runtime fast est=1s lane=default
func TestInKernelAdaptiveSnapshotCheckpointKeepsGridOverSubBlockMatch(t *testing.T) {
	for _, tc := range []struct {
		name                       string
		matched, cacheable, prompt int
		want                       int
	}{
		{"req1 sub-block preamble keeps grid", 0, 5, 1961, 1920},
		{"req2 exact tail past restored grid", 1920, 1922, 1961, 1922},
		{"whole-block shared boundary below grid", 0, 1000, 1961, 1000},
		{"shared boundary at or past grid", 0, 1940, 1961, 1940},
		{"sub-block boundary on short prompt with no grid", 0, 5, 40, 5},
		{"no structural match falls back to grid", 0, 0, 1961, 1920},
		{"cacheable equals prompt falls back to grid", 0, 1961, 1961, 1920},
		{"restored prefix beyond grid", 1940, 1940, 1961, 0},
		// Accepted trade-off: a sub-block tail past the restored prefix yields to a
		// deeper grid point; at most one block (<64 tokens) is recomputed per sibling.
		{"sub-block tail yields to deeper grid", 1920, 1950, 2100, 2048},
		{"sub-block shared prefix yields to grid", 0, 40, 100, 64},
	} {
		if got := inKernelAdaptiveSnapshotCheckpoint(tc.matched, tc.cacheable, tc.prompt); got != tc.want {
			t.Errorf("%s: inKernelAdaptiveSnapshotCheckpoint(%d,%d,%d)=%d, want %d", tc.name, tc.matched, tc.cacheable, tc.prompt, got, tc.want)
		}
	}
}

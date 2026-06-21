package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fixtureProfile builds a small but realistic-shaped profile: one 20-turn track at 40%
// tool-call fraction, decode 100/turn, result 50 on tool turns.
func fixtureProfile() (*workloadProfile, replayTrack) {
	track := make([]replayStep, 20)
	for i := range track {
		track[i].Decode = 100
		// every other-ish turn is a tool turn → 8/20 = 0.40
		if i%5 == 1 || i%5 == 3 {
			track[i].Tool = true
			track[i].Result = 50
		}
	}
	tr := replayTrack{
		Session: "fixture", PercentileByTurns: 90, PrefixTokens: 2000,
		NTurns: 20, ToolCallFraction: 0.40, Track: track,
	}
	p := &workloadProfile{
		Schema: "fak.workload.v1", ToolCallFraction: 0.40,
		PrefixTokens:            statDist{Median: 2000},
		ResultTokensPerToolTurn: statDist{Median: 50},
		TurnsPerSession:         statDist{Median: 20},
		Replay:                  []replayTrack{tr},
	}
	return p, tr
}

func TestBuildScheduleReplaysTrackGeometry(t *testing.T) {
	p, tr := fixtureProfile()
	prefix, steps, eff := buildSchedule(p, tr, tune{toolFrac: 1, result: 1, decode: 1, prefix: 1})
	if prefix != 2000 {
		t.Fatalf("prefix = %d, want 2000", prefix)
	}
	if len(steps) != 20 {
		t.Fatalf("steps = %d, want 20", len(steps))
	}
	dec, res, toolTurns := scheduleTotals(steps)
	if dec != 2000 {
		t.Fatalf("decode total = %d, want 2000 (20×100)", dec)
	}
	// 40% of 20 turns ≈ 8 tool turns, each 50 tok result → ~400
	if toolTurns < 7 || toolTurns > 9 {
		t.Fatalf("tool turns = %d, want ~8 at 40%%", toolTurns)
	}
	if res < 350 || res > 450 {
		t.Fatalf("result total = %d, want ~400", res)
	}
	if eff < 0.35 || eff > 0.45 {
		t.Fatalf("effective tool fraction = %.3f, want ~0.40", eff)
	}
}

func TestTuneToolFracIsMonotone(t *testing.T) {
	p, tr := fixtureProfile()
	count := func(mul float64) int {
		_, steps, _ := buildSchedule(p, tr, tune{toolFrac: mul, result: 1, decode: 1, prefix: 1})
		_, _, tt := scheduleTotals(steps)
		return tt
	}
	lo, mid, hi := count(0.5), count(1.0), count(2.0)
	if !(lo < mid && mid < hi) {
		t.Fatalf("tool-call fraction knob not monotone: 0.5×=%d 1.0×=%d 2.0×=%d", lo, mid, hi)
	}
	// 0.40×0.5=0.20 → ~4 ; 0.40×2.0=0.80 → ~16
	if lo < 3 || lo > 5 {
		t.Fatalf("0.5× tool turns = %d, want ~4", lo)
	}
	if hi < 15 || hi > 17 {
		t.Fatalf("2.0× tool turns = %d, want ~16", hi)
	}
}

func TestTuneToolFracCannotExceedAllTurns(t *testing.T) {
	p, tr := fixtureProfile()
	_, steps, eff := buildSchedule(p, tr, tune{toolFrac: 10, result: 1, decode: 1, prefix: 1})
	_, _, tt := scheduleTotals(steps)
	if tt != 20 {
		t.Fatalf("saturated tool turns = %d, want 20 (all turns)", tt)
	}
	if eff != 1.0 {
		t.Fatalf("saturated effective fraction = %.3f, want 1.0", eff)
	}
}

func TestTuneScalesDecodeResultPrefix(t *testing.T) {
	p, tr := fixtureProfile()
	prefix, steps, _ := buildSchedule(p, tr, tune{toolFrac: 1, result: 2, decode: 0.5, prefix: 3})
	if prefix != 6000 {
		t.Fatalf("prefix = %d, want 6000 (2000×3)", prefix)
	}
	dec, res, _ := scheduleTotals(steps)
	if dec != 1000 {
		t.Fatalf("decode total = %d, want 1000 (20×100×0.5)", dec)
	}
	// ~8 tool turns × (50×2) = ~800
	if res < 700 || res > 900 {
		t.Fatalf("result total = %d, want ~800 (R doubled)", res)
	}
}

func TestTurnCapLimitsReplay(t *testing.T) {
	p, tr := fixtureProfile()
	_, steps, _ := buildSchedule(p, tr, tune{toolFrac: 1, result: 1, decode: 1, prefix: 1, turnCap: 6})
	if len(steps) != 6 {
		t.Fatalf("capped steps = %d, want 6", len(steps))
	}
}

func TestScaleIntFloorsAtMin(t *testing.T) {
	if got := scaleInt(100, 0.0001, 1); got != 1 {
		t.Fatalf("scaleInt floor = %d, want 1", got)
	}
	if got := scaleInt(0, 1, 1); got != 1 {
		t.Fatalf("scaleInt min on zero = %d, want 1", got)
	}
}

func TestLoadWorkloadRejectsBadSchema(t *testing.T) {
	dir := t.TempDir()
	bad := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(bad, []byte(`{"schema":"nope","replay":[]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadWorkload(bad); err == nil {
		t.Fatal("expected error on wrong schema, got nil")
	}
	good := filepath.Join(dir, "good.json")
	blob := `{"schema":"fak.workload.v1","tool_call_fraction":0.4,
	  "replay":[{"session":"x","percentile_by_turns":90,"prefix_tokens":10,"n_turns":2,
	  "tool_call_fraction":0.5,"track":[{"decode":5,"tool":true,"result":3},{"decode":4,"tool":false,"result":0}]}]}`
	if err := os.WriteFile(good, []byte(blob), 0o644); err != nil {
		t.Fatal(err)
	}
	p, err := loadWorkload(good)
	if err != nil {
		t.Fatalf("load good: %v", err)
	}
	if len(p.Replay) != 1 || len(p.Replay[0].Track) != 2 {
		t.Fatalf("parsed replay wrong: %+v", p.Replay)
	}
}

// runScheduleTurns must advance every agent's KV by sum(decode)+sum(result), matching the
// runTurns geometry contract but with per-turn-varying shapes.
func TestRunScheduleTurnsAdvancesCache(t *testing.T) {
	cfg := model.Config{
		HiddenSize: 32, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 101, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	m := model.NewSynthetic(cfg)
	bs := m.NewBatchSession(3)
	prompts := [][]int{{1, 2}, {3, 4, 5}, {6}}
	bs.PrefillEach(prompts)

	steps := []turnStep{{decode: 4, result: 2}, {decode: 3, result: 0}, {decode: 2, result: 5}}
	ids0 := []int{7, 8, 9}
	idsCopy := append([]int(nil), ids0...)
	runScheduleTurns(bs, ids0, steps, cfg.VocabSize, 0)
	for i := range ids0 {
		if ids0[i] != idsCopy[i] {
			t.Fatalf("runScheduleTurns mutated ids0[%d]", i)
		}
	}
	wantExtra := (4 + 3 + 2) + (2 + 0 + 5) // sum decode + sum result
	for agent, s := range bs.Seqs {
		wantLen := len(prompts[agent]) + wantExtra
		if got := s.Cache.Len(); got != wantLen {
			t.Fatalf("agent %d cache len = %d, want %d", agent, got, wantLen)
		}
	}
}

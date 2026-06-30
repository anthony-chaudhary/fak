package frontierswe

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// rewardFixtureDir holds the committed reward.json fixtures plus the
// expected.json sidecar — the ground truth recorded by running the upstream
// oracle scripts/score_from_reward.py on each fixture (see
// docs/benchmarks/FRONTIERSWE-SCORING-PARITY.md).
const rewardFixtureDir = "testdata/frontierswe/reward"

// expectedEntry is one row of expected.json: the task the fixture is scored as,
// the optional SSIM threshold the oracle ran with (nil => the library default
// 0.99), and the oracle's correctness / speedup / gated score.
type expectedEntry struct {
	Task        string   `json:"task"`
	SSIM        *float64 `json:"ssim"`
	Correctness float64  `json:"correctness"`
	Speedup     *float64 `json:"speedup"`
	Score       float64  `json:"score"`
}

const scoreEpsilon = 1e-9

func floatsEqual(a, b float64) bool { return math.Abs(a-b) <= scoreEpsilon }

func ptrEqual(a, b *float64) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return floatsEqual(*a, *b)
}

func loadExpected(t *testing.T) map[string]expectedEntry {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(rewardFixtureDir, "expected.json"))
	if err != nil {
		t.Fatalf("read expected.json: %v", err)
	}
	var m map[string]expectedEntry
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("parse expected.json: %v", err)
	}
	if len(m) == 0 {
		t.Fatal("expected.json is empty — no parity fixtures to check")
	}
	return m
}

func loadReward(t *testing.T, name string) *Reward {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(rewardFixtureDir, name+".json"))
	if err != nil {
		t.Fatalf("read fixture %s.json: %v", name, err)
	}
	var r Reward
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatalf("parse fixture %s.json: %v", name, err)
	}
	return &r
}

// TestGatedScoreParity is the acceptance test: for every committed reward.json
// fixture, the Go scorer must reproduce the oracle's correctness, speedup, and
// gated leaderboard score recorded in expected.json. This covers all three
// categories (implementation / performance / ml_research) plus the two special
// cases (notebook-compression, libexpat-to-x86asm).
func TestGatedScoreParity(t *testing.T) {
	expected := loadExpected(t)

	// Track which categories + special cases the fixture set actually covers,
	// so the suite fails if someone deletes a whole category's fixtures.
	covered := map[string]bool{}

	for name, want := range expected {
		name, want := name, want
		t.Run(name, func(t *testing.T) {
			reward := loadReward(t, name)

			ssim := defaultSSIMThreshold
			if want.SSIM != nil {
				ssim = *want.SSIM
			}
			gotCorr, gotSpeedup := ExtractScoreSSIM(reward, want.Task, ssim)
			if !floatsEqual(gotCorr, want.Correctness) {
				t.Errorf("correctness: got %v, oracle %v", gotCorr, want.Correctness)
			}
			if !ptrEqual(gotSpeedup, want.Speedup) {
				t.Errorf("speedup: got %v, oracle %v", fmtPtr(gotSpeedup), fmtPtr(want.Speedup))
			}

			gotScore := GatedScore(gotCorr, gotSpeedup, want.Task)
			if !floatsEqual(gotScore, want.Score) {
				t.Errorf("gated score: got %v, oracle %v", gotScore, want.Score)
			}

			// Score() (extract -> gate, no anti-cheat flag) must equal the gate
			// of the extracted values — and with the library default SSIM that
			// is the oracle score for every fixture run at 0.99.
			if want.SSIM == nil {
				if s := Score(reward, want.Task, false); !floatsEqual(s, want.Score) {
					t.Errorf("Score(): got %v, oracle %v", s, want.Score)
				}
				if s := Score(reward, want.Task, true); !floatsEqual(s, 0.0) {
					t.Errorf("Score() anti-cheat: got %v, want 0", s)
				}
			}

			if cat, ok := CategoryOf(want.Task); ok {
				covered[cat.String()] = true
			}
			if want.Task == "notebook-compression" {
				covered["notebook-compression"] = true
			}
			if want.Task == "libexpat-to-x86asm" {
				covered["libexpat-to-x86asm"] = true
			}
		})
	}

	for _, need := range []string{
		"implementation", "performance", "ml_research",
		"notebook-compression", "libexpat-to-x86asm",
	} {
		if !covered[need] {
			t.Errorf("no fixture covers required case %q", need)
		}
	}
}

func fmtPtr(p *float64) string {
	if p == nil {
		return "nil"
	}
	return "&" + jsonFloat(*p)
}

func jsonFloat(f float64) string {
	b, _ := json.Marshal(f)
	return string(b)
}

// TestCategoryListsMatchOracle cross-checks the catalog's category lists against
// the IMPLEMENTATION / PERFORMANCE / ML_RESEARCH lists hardcoded in the upstream
// scripts/score_from_reward.py. A drift between this Go port and the oracle's
// task->category mapping fails here.
func TestCategoryListsMatchOracle(t *testing.T) {
	// The exact lists from scripts/score_from_reward.py (the oracle).
	oracleImpl := []string{
		"git-to-zig", "dart-style-haskell", "lua-native-compiler",
		"postgres-sqlite-wire-adapter", "modular-stack-wan21",
	}
	oraclePerf := []string{
		"libexpat-to-x86asm", "ffmpeg-swscale-rewrite",
		"pyright-type-checking-optimization", "granite-mamba2-inference-optimization",
		"notebook-compression", "revideo-perf-opt", "cranelift-codegen-opt",
		"dependent-type-checker", "inference-system-optimization",
	}
	oracleML := []string{"pcqm4mv2-autoresearch", "frogsgame-rl", "optimizer-design"}

	check := func(label string, oracle []string, cat Category) {
		got := TaskNamesByCategory(cat)
		oc := append([]string(nil), oracle...)
		gc := append([]string(nil), got...)
		sort.Strings(oc)
		sort.Strings(gc)
		if len(oc) != len(gc) {
			t.Errorf("%s: got %d tasks, oracle %d", label, len(gc), len(oc))
			return
		}
		for i := range oc {
			if oc[i] != gc[i] {
				t.Errorf("%s: task mismatch at %d: got %q, oracle %q", label, i, gc[i], oc[i])
			}
		}
		for _, name := range oracle {
			if c, ok := CategoryOf(name); !ok || c != cat {
				t.Errorf("%s: CategoryOf(%q) = (%q,%v), want (%q,true)", label, name, c, ok, cat)
			}
		}
	}
	check("implementation", oracleImpl, CategoryImplementation)
	check("performance", oraclePerf, CategoryPerformance)
	check("ml_research", oracleML, CategoryMLResearch)

	// The full roster is exactly 17 tasks.
	if n := len(TaskNames()); n != 17 {
		t.Errorf("TaskNames(): got %d, want 17", n)
	}
}

// TestGateFormulas cross-checks the GatedScore gate formulas against the
// compute_gated_score branches in the oracle, independent of any fixture:
//   - implementation: returns correctness unchanged
//   - performance: 0.5*correctness, or 0.5+0.5*speedup once correctness==1.0
//   - notebook-compression: speedup if fully correct, else 0
//   - libexpat-to-x86asm: uncapped 0.5+0.5*speedup once fully correct, else 0.5*correctness
//   - ml_research: raw (frogsgame-rl divided by 500)
func TestGateFormulas(t *testing.T) {
	sp := func(f float64) *float64 { return &f }

	cases := []struct {
		name        string
		task        string
		correctness float64
		speedup     *float64
		want        float64
	}{
		{"impl raw", "git-to-zig", 0.42, nil, 0.42},
		{"impl full", "git-to-zig", 1.0, nil, 1.0},
		{"perf partial half", "ffmpeg-swscale-rewrite", 0.6, nil, 0.3},
		{"perf full speedup", "ffmpeg-swscale-rewrite", 1.0, sp(1.8), 0.5 + 0.9},
		{"perf full no speedup", "ffmpeg-swscale-rewrite", 1.0, nil, 0.5},
		{"notebook correct", "notebook-compression", 1.0, sp(0.7), 0.7},
		{"notebook not full", "notebook-compression", 0.9, sp(0.7), 0.0},
		{"notebook full no speedup", "notebook-compression", 1.0, nil, 0.0},
		{"libexpat partial", "libexpat-to-x86asm", 0.45, sp(0.5), 0.225},
		{"libexpat full uncapped", "libexpat-to-x86asm", 1.0, sp(0.9), 0.95},
		{"libexpat full big speedup", "libexpat-to-x86asm", 1.0, sp(3.0), 0.5 + 1.5},
		{"ml raw", "optimizer-design", 0.83, nil, 0.83},
		{"frogsgame board count", "frogsgame-rl", 250.0, nil, 0.5},
		{"frogsgame full board", "frogsgame-rl", 500.0, nil, 1.0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := GatedScore(c.correctness, c.speedup, c.task)
			if !floatsEqual(got, c.want) {
				t.Errorf("GatedScore(%v, %v, %q) = %v, want %v",
					c.correctness, c.speedup, c.task, got, c.want)
			}
		})
	}

	// Unknown task -> 0 (the Go port returns 0 where the oracle raises KeyError).
	if got := GatedScore(1.0, sp(2.0), "not-a-real-task"); got != 0 {
		t.Errorf("GatedScore(unknown) = %v, want 0", got)
	}
}

// TestAntiCheatZeroing checks the anti-cheat layer: a trial flagged in
// scoring/anticheat.json scores 0 regardless of its raw gated score. This is the
// runtime's composition point (the reference compute_gated_score has no such
// flag — see FRONTIERSWE-SCORING-PARITY.md).
func TestAntiCheatZeroing(t *testing.T) {
	sp := func(f float64) *float64 { return &f }

	// A would-be-high score, flagged, must collapse to 0.
	if got := GatedScoreAntiCheat(1.0, sp(1.8), "ffmpeg-swscale-rewrite", true); got != 0 {
		t.Errorf("flagged ffmpeg gated score = %v, want 0", got)
	}
	// Unflagged passes through unchanged.
	if got, want := GatedScoreAntiCheat(1.0, sp(1.8), "ffmpeg-swscale-rewrite", false), 1.4; !floatsEqual(got, want) {
		t.Errorf("unflagged ffmpeg = %v, want %v", got, want)
	}
	// Score end-to-end with the flag set is 0 for a fully-correct trial.
	reward := loadReward(t, "ffmpeg_full_speedup")
	if got := Score(reward, "ffmpeg-swscale-rewrite", true); got != 0 {
		t.Errorf("Score(flagged) = %v, want 0", got)
	}
	if got, want := Score(reward, "ffmpeg-swscale-rewrite", false), 1.4; !floatsEqual(got, want) {
		t.Errorf("Score(unflagged) = %v, want %v", got, want)
	}
}

// TestNilAndEmptyReward checks the edge that the oracle's `if not reward_data`
// guards: a nil reward extracts to (0, nil) and gates to 0.
func TestNilAndEmptyReward(t *testing.T) {
	corr, speedup := ExtractScore(nil, "git-to-zig")
	if corr != 0 || speedup != nil {
		t.Errorf("ExtractScore(nil) = (%v, %v), want (0, nil)", corr, fmtPtr(speedup))
	}
}

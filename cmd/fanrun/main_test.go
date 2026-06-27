package main

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/bench"
	"github.com/anthony-chaudhary/fak/internal/turnbench"
)

// TestBuildAgentGridFanrun pins the grid semantics to cmd/fanbench's: explicit wins; the
// log ladder always includes 1 and max; canonical is the D-001 acceptance ladder.
func TestBuildAgentGridFanrun(t *testing.T) {
	if got := buildAgentGrid("1,4,1024", 0, "log"); !reflect.DeepEqual(got, []int{1, 4, 1024}) {
		t.Errorf("explicit grid = %v, want [1 4 1024]", got)
	}
	log := buildAgentGrid("", 1024, "log")
	if log[0] != 1 || log[len(log)-1] != 1024 {
		t.Errorf("log grid must include 1 and 1024; got %v", log)
	}
	if got := buildAgentGrid("", 1000, "canonical"); !reflect.DeepEqual(got, []int{1, 100, 500, 1000}) {
		t.Errorf("canonical grid = %v, want [1 100 500 1000]", got)
	}
}

func TestProfileByNameFanrun(t *testing.T) {
	for _, name := range []string{"research", "write-heavy", "no-share"} {
		if _, ok := profileByName(name); !ok {
			t.Errorf("profile %q should resolve", name)
		}
	}
	if _, ok := profileByName("nonsense"); ok {
		t.Errorf("unknown profile must not resolve")
	}
}

// TestFanrunReportShape witnesses the artifact contract: a real small run produces a
// fanrun/1 report marked no-GPU, with a measured wall-clock, a real cross-hit count, the
// exact geometry, and — critically — NO modeled fields (no token-multiplier, no dollars, no
// parallel_speedup leak from fanbench).
func TestFanrunReportShape(t *testing.T) {
	rep := bench.RunFanoutLive(context.Background(), bench.FanrunOptions{
		Profile: turnbench.FanoutResearch, Grid: []int{1, 4}, SubTurns: 8, Prefix: 2048, Trials: 1,
	})
	b, err := json.Marshal(rep)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	js := string(b)

	if rep.Schema != bench.FanrunSchema {
		t.Errorf("schema = %q, want %q", rep.Schema, bench.FanrunSchema)
	}
	if rep.Host.HasGPU {
		t.Errorf("has_gpu must be false on this CPU-only capstone")
	}
	// No modeled fields may appear in the artifact — the modeled 72.8× stays in fanbench.
	for _, banned := range []string{"token_mult", "parallel_speedup", "dollars", "tax_clawed", "net_dollars"} {
		if strings.Contains(js, banned) {
			t.Errorf("artifact must carry NO modeled field; found %q in JSON", banned)
		}
	}
	// No wall-clock timestamp embedded (byte-reproducible projection).
	for _, ts := range []string{"timestamp", "generated_at", "\"time\":", "wall_clock_at"} {
		if strings.Contains(js, ts) {
			t.Errorf("artifact must embed no timestamp; found %q", ts)
		}
	}
	last := rep.Cells[len(rep.Cells)-1]
	if last.PrefixTokensElided != (last.Agents-1)*rep.Prefix {
		t.Errorf("prefix_tokens_elided = %d, want (N-1)*P", last.PrefixTokensElided)
	}
	if last.CrossHits <= 0 {
		t.Errorf("N=%d cross_hits = %d, want > 0", last.Agents, last.CrossHits)
	}
	// Timing fields are absent (omitempty) when no model is supplied.
	if strings.Contains(js, "prefill_reuse_total_ms") {
		t.Errorf("prefill timing fields must be omitted with no --model-dir; JSON: %s", js)
	}
}

// TestFanrunCounterProjectionReproducible re-runs the sweep and asserts the
// counter+geometry projection is byte-identical (the reproducibility gate).
func TestFanrunCounterProjectionReproducible(t *testing.T) {
	mk := func() string {
		rep := bench.RunFanoutLive(context.Background(), bench.FanrunOptions{
			Profile: turnbench.FanoutResearch, Grid: []int{1, 4, 16}, SubTurns: 8, Prefix: 2048, Trials: 1,
		})
		// project to the byte-stable fields only (zero the wall-clock halves)
		type proj struct{ Agents, Wave, Cross, Turns, Prompt, Compl, Tasks, Elided int }
		var ps []proj
		for _, c := range rep.Cells {
			ps = append(ps, proj{c.Agents, c.WaveHits, c.CrossHits, c.TurnsTotal,
				c.PromptTokens, c.CompletionTokens, c.TasksCompleted, c.PrefixTokensElided})
		}
		b, _ := json.Marshal(ps)
		return string(b)
	}
	if a, b := mk(), mk(); a != b {
		t.Errorf("counter+geometry projection not reproducible:\n a=%s\n b=%s", a, b)
	}
}

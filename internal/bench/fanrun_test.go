package bench

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/turnbench"
)

// smallOpts is a model-free fan-out config: the counter+geometry halves exercise the real
// kernel fully without any model on disk (ModelDir:"" => prefill timing skipped).
func smallOpts(profile turnbench.FanoutProfile, grid []int, trials int) FanrunOptions {
	return FanrunOptions{
		Profile: profile, Grid: grid, SubTurns: 8, Prefix: 2048,
		Trials: trials, Reps: 0, Seed: 1, ModelDir: "",
	}
}

// TestFanrunSmallEndToEnd is the core witness: N real RunArm sessions actually drive the
// canonical multi-tool task to completion THROUGH the kernel, and siblings hit the warm
// shared cache. This is "we ran N real agents on one goal," at a CI-tractable N.
func TestFanrunSmallEndToEnd(t *testing.T) {
	rep := RunFanoutLive(context.Background(), smallOpts(turnbench.FanoutResearch, []int{4}, 1))
	if rep.Schema != FanrunSchema {
		t.Fatalf("schema = %q, want %q", rep.Schema, FanrunSchema)
	}
	if len(rep.Cells) != 1 {
		t.Fatalf("cells = %d, want 1", len(rep.Cells))
	}
	c := rep.Cells[0]
	if c.Agents != 4 {
		t.Fatalf("agents = %d, want 4", c.Agents)
	}
	if c.TasksCompleted != 4 {
		t.Errorf("tasks_completed = %d, want 4 (every sub-agent must finish the goal)", c.TasksCompleted)
	}
	if c.ToolErrorsTotal != 0 {
		t.Errorf("tool_errors_total = %d, want 0 (the fak arm repairs in-syscall, never retries)", c.ToolErrorsTotal)
	}
	if c.TurnsTotal <= 0 {
		t.Errorf("turns_total = %d, want > 0 (real model round-trips)", c.TurnsTotal)
	}
	if c.CrossHits <= 0 {
		t.Errorf("cross_hits = %d, want > 0 (siblings 1..3 hit sub-agent 0's warm shared cache)", c.CrossHits)
	}
	if c.WaveHits < c.CrossHits {
		t.Errorf("wave_hits %d < cross_hits %d (cross must be the sibling-only subset)", c.WaveHits, c.CrossHits)
	}
	// A tiny offline wave can finish within a single monotonic-clock tick on a fast
	// host, so the measured serial wall-clock is legitimately 0 there — a strict >0
	// floor is not portable (#2649). Require it present and non-negative, its per-agent
	// mean a consistent projection of the measured sum, and — only when the clock DID
	// advance — the derived throughput positive. No fabricated timing.
	if c.AgentsWallSerialMs < 0 {
		t.Errorf("agents_wall_serial_ms = %v, want >= 0 (measured, non-negative)", c.AgentsWallSerialMs)
	}
	if c.Agents > 0 && c.PerAgentMsMean != c.AgentsWallSerialMs/float64(c.Agents) {
		t.Errorf("per_agent_ms_mean = %v, want wall/agents = %v (consistent projection)", c.PerAgentMsMean, c.AgentsWallSerialMs/float64(c.Agents))
	}
	if c.AgentsWallSerialMs > 0 && c.AgentsPerSecSerial <= 0 {
		t.Errorf("wall %v > 0 but agents_per_sec_serial = %v (throughput must derive when measurable)", c.AgentsWallSerialMs, c.AgentsPerSecSerial)
	}
	if !rep.SharedGoal {
		t.Errorf("research profile must be marked shared_goal=true")
	}
}

// TestFanrunDeterminism proves the counter halves are reproducible: the deterministic
// offline planner + per-cell epoch reset means two trials yield identical counts, so
// cross_hits_stable is a real witness, not a hope.
func TestFanrunDeterminism(t *testing.T) {
	rep := RunFanoutLive(context.Background(), smallOpts(turnbench.FanoutResearch, []int{8}, 2))
	c := rep.Cells[0]
	if !c.CrossHitsStable {
		t.Errorf("cross_hits_stable = false; the deterministic planner must yield identical counts across trials")
	}

	// Two whole-sweep runs must agree on the counter+geometry projection.
	a := RunFanoutLive(context.Background(), smallOpts(turnbench.FanoutResearch, []int{8}, 1)).Cells[0]
	b := RunFanoutLive(context.Background(), smallOpts(turnbench.FanoutResearch, []int{8}, 1)).Cells[0]
	if a.WaveHits != b.WaveHits || a.CrossHits != b.CrossHits || a.TurnsTotal != b.TurnsTotal ||
		a.PromptTokens != b.PromptTokens || a.CompletionTokens != b.CompletionTokens ||
		a.TasksCompleted != b.TasksCompleted || a.PrefixTokensElided != b.PrefixTokensElided {
		t.Errorf("counter+geometry projection not reproducible:\n a=%+v\n b=%+v", a, b)
	}
}

// TestFanrunNoShareZeroUplift is the anti-inflation control. With the no-share profile each
// sub-agent's reads carry a distinct user_id salt, so no sibling can reuse another's cached
// read — cross_hits MUST be exactly 0 at every N. A non-zero value is a harness bug, the
// same guarantee turnbench.FanoutNoShare enforces.
func TestFanrunNoShareZeroUplift(t *testing.T) {
	rep := RunFanoutLive(context.Background(), smallOpts(turnbench.FanoutNoShare, []int{4, 8}, 1))
	if rep.SharedGoal {
		t.Fatalf("no-share profile must be marked shared_goal=false")
	}
	for _, c := range rep.Cells {
		if c.CrossHits != 0 {
			t.Errorf("N=%d: cross_hits = %d, want 0 (no-share control: distinct arg hashes, no sibling reuse)", c.Agents, c.CrossHits)
		}
		// Tasks still complete — salting the user_id does not break the goal.
		if c.TasksCompleted != c.Agents {
			t.Errorf("N=%d: tasks_completed = %d, want %d (salt must not break the task)", c.Agents, c.TasksCompleted, c.Agents)
		}
	}
}

// TestPrefixElisionGeometry checks the exact (N-1)*P geometry across a grid. The clone
// bit-identity itself is already proven by cmd/fanbench/main_test.go:TestPrefixReuseFanoutWitness
// and model.TestKVPrefixReuseMatchesRecompute; here we only assert the arithmetic fanrun reports.
func TestPrefixElisionGeometry(t *testing.T) {
	const P = 2048
	opts := smallOpts(turnbench.FanoutResearch, []int{1, 4, 16}, 1)
	opts.Prefix = P
	rep := RunFanoutLive(context.Background(), opts)
	for _, c := range rep.Cells {
		want := (c.Agents - 1) * P
		if c.PrefixTokensElided != want {
			t.Errorf("N=%d: prefix_tokens_elided = %d, want (N-1)*P = %d", c.Agents, c.PrefixTokensElided, want)
		}
	}
	// N=1 elides nothing — a lone agent shares no prefix.
	if rep.Cells[0].PrefixTokensElided != 0 {
		t.Errorf("N=1 must elide 0 prefix tokens, got %d", rep.Cells[0].PrefixTokensElided)
	}
}

// TestCrossHitsScaleWithN sanity-checks that sibling dedup grows with the fan-out width in
// the shared regime (more siblings hitting the warm cache => more cross hits).
func TestCrossHitsScaleWithN(t *testing.T) {
	rep := RunFanoutLive(context.Background(), smallOpts(turnbench.FanoutResearch, []int{1, 4, 16}, 1))
	n1, n4, n16 := rep.Cells[0], rep.Cells[1], rep.Cells[2]
	if n1.CrossHits != 0 {
		t.Errorf("N=1 cross_hits = %d, want 0 (no sibling)", n1.CrossHits)
	}
	if !(n16.CrossHits > n4.CrossHits && n4.CrossHits > 0) {
		t.Errorf("cross_hits must grow with N: N=4=%d N=16=%d", n4.CrossHits, n16.CrossHits)
	}
}

// TestFanrunWallClockZeroIsPortable pins the #2649 fix deterministically: a wave that
// completes within a single monotonic-clock tick on a fast host measures wallMs == 0,
// and that is a VALID sub-granularity measurement — the derived per-agent mean and
// serial throughput stay zero-safe (no divide-by-zero, no +Inf) and the projection
// remains internally consistent. The old TestFanrunSmallEndToEnd floor asserted
// agents_wall_serial_ms > 0, which this legitimate zero-elapsed wave trips.
func TestFanrunWallClockZeroIsPortable(t *testing.T) {
	// Sub-granularity wave: a real wave whose summed elapsed rounds to zero ns.
	var zero FanrunCell
	projectWallClock(&zero, 0, 4)
	if zero.AgentsWallSerialMs != 0 {
		t.Fatalf("agents_wall_serial_ms = %v, want 0 (sub-tick wave)", zero.AgentsWallSerialMs)
	}
	if zero.PerAgentMsMean != 0 {
		t.Fatalf("per_agent_ms_mean = %v, want 0 for a zero-elapsed wave", zero.PerAgentMsMean)
	}
	if zero.AgentsPerSecSerial != 0 {
		t.Fatalf("agents_per_sec_serial = %v, want 0 (guarded, no divide-by-zero/+Inf)", zero.AgentsPerSecSerial)
	}

	// A measurable wave still projects a positive, consistent throughput.
	var pos FanrunCell
	projectWallClock(&pos, 2.0, 4)
	if pos.PerAgentMsMean != 0.5 {
		t.Fatalf("per_agent_ms_mean = %v, want 2.0/4 = 0.5", pos.PerAgentMsMean)
	}
	if pos.AgentsPerSecSerial != 2000 {
		t.Fatalf("agents_per_sec_serial = %v, want 4/(2.0/1e3) = 2000", pos.AgentsPerSecSerial)
	}
}

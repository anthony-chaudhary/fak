package gateway

import "testing"

// TestCrossAgentReuseTelemetry witnesses the #12312 cross-agent prefix-overlap and
// token-reuse accounting over a multi-agent (coordinator + parallel subagent) session
// trace. It drives the ledger the way the served path does — one ObserveCrossAgentTurn
// per subagent turn, each naming its parent coordinator trace — then asserts the rollup
// the /debug/vars agents pane and `fak info` consumption contract depend on:
//
//   - per-trace rows carry ParentSessionID + SubagentType so the pane roots each
//     subagent under its coordinator;
//   - the rollup CrossAgentSharedTokens / CrossAgentReusePct count ONLY subagent turns
//     (a coordinator's own prefix reuse is intra-agent, never cross-agent);
//   - AvoidedPrefillLatencySeconds prices the shared tokens, never a fabricated speedup.
//
// The witness is deterministic and in-memory (software logic only): no hardware claim.
func TestCrossAgentReuseTelemetry(t *testing.T) {
	led := newCrossAgentReuseLedger()
	// A measured prefill-rate witness (seconds per prompt token) prices avoided prefill;
	// without it the seconds stay 0 (unmeasured, never invented).
	led.SetPrefillRate(0.0002)

	// Two parallel workers and one researcher fold under coordinator "coord-1"; each
	// shares a common system-prompt / tool-catalog prefix with the coordinator.
	led.ObserveCrossAgentTurn("coord-1", "sub-w1", "worker", 32000, 25000)
	led.ObserveCrossAgentTurn("coord-1", "sub-w2", "worker", 24000, 21000)
	led.ObserveCrossAgentTurn("coord-1", "sub-r1", "researcher", 18000, 16000)

	// A second coordinator with an independent subagent must stay its own series.
	led.ObserveCrossAgentTurn("coord-2", "sub-x1", "tester", 10000, 4000)

	// A coordinator's OWN turn is intra-agent reuse: it must NOT be counted cross-agent.
	led.ObserveCrossAgentTurn("coord-1", "coord-1", "", 80000, 70000)

	rows := led.Snapshot()
	if len(rows) != 2 {
		t.Fatalf("Snapshot rows = %d, want 2 (one per coordinator)", len(rows))
	}

	// Rows are deterministically ordered by parent trace id.
	if rows[0].ParentSessionID != "coord-1" || rows[1].ParentSessionID != "coord-2" {
		t.Fatalf("row parents = [%q, %q], want [coord-1, coord-2]",
			rows[0].ParentSessionID, rows[1].ParentSessionID)
	}

	first := rows[0]
	if first.SubagentCount != 3 {
		t.Errorf("coord-1 SubagentCount = %d, want 3", first.SubagentCount)
	}
	if first.PromptTokens != 74000 { // 32000 + 24000 + 18000
		t.Errorf("coord-1 PromptTokens = %d, want 74000", first.PromptTokens)
	}
	if first.SharedTokens != 62000 { // 25000 + 21000 + 16000
		t.Errorf("coord-1 SharedTokens = %d, want 62000", first.SharedTokens)
	}
	// 62000 / 74000 = 0.837837... -> 83.78%
	if got := first.ReusePct(); got < 83.7 || got > 83.9 {
		t.Errorf("coord-1 ReusePct = %.2f, want ~83.78", got)
	}
	if first.AvoidedPrefillLatencySeconds <= 0 {
		t.Errorf("coord-1 AvoidedPrefillLatencySeconds = %v, want > 0", first.AvoidedPrefillLatencySeconds)
	}

	// The second coordinator's subagent shared only 4000 of 10000 tokens cross-agent.
	second := rows[1]
	if second.SharedTokens != 4000 || second.PromptTokens != 10000 {
		t.Errorf("coord-2 shared/prompt = %d/%d, want 4000/10000", second.SharedTokens, second.PromptTokens)
	}

	// The rollup folds ALL coordinators; the coordinator's own 80000/70000 turn is excluded.
	roll := led.Rollup()
	if roll.CrossAgentSharedTokens != 66000 { // 62000 + 4000
		t.Errorf("CrossAgentSharedTokens = %d, want 66000", roll.CrossAgentSharedTokens)
	}
	if roll.CrossAgentPromptTokens != 84000 { // 74000 + 10000
		t.Errorf("CrossAgentPromptTokens = %d, want 84000", roll.CrossAgentPromptTokens)
	}
	if roll.CrossAgentSubagentCount != 4 {
		t.Errorf("CrossAgentSubagentCount = %d, want 4", roll.CrossAgentSubagentCount)
	}
	// 66000 / 84000 = 0.785714... -> 78.57%
	if roll.CrossAgentReusePct < 78.5 || roll.CrossAgentReusePct > 78.6 {
		t.Errorf("CrossAgentReusePct = %.2f, want ~78.57", roll.CrossAgentReusePct)
	}
}

// TestCrossAgentReuseTelemetryEmpty pins the idle-process contract: no observations
// yields no rows and a zero rollup whose ratio is 0 (never a phantom or NaN ratio), so a
// cold gateway omits the block rather than reporting a fabricated reuse rate.
func TestCrossAgentReuseTelemetryEmpty(t *testing.T) {
	led := newCrossAgentReuseLedger()
	if rows := led.Snapshot(); len(rows) != 0 {
		t.Fatalf("empty Snapshot rows = %d, want 0", len(rows))
	}
	roll := led.Rollup()
	if roll.CrossAgentReusePct != 0 || roll.CrossAgentSharedTokens != 0 {
		t.Errorf("empty rollup = %+v, want zero ratio and tokens", roll)
	}
}

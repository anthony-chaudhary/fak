package gateway

// served_inline_guardtrace_test.go — the DOMINANT-PATH arm of the #1350 measurement.
//
// served_inline_dedup_test.go already measures the served-inline dedup rate on a
// hand-rolled transcript, renaming the tools twice: once into read-only-SHAPED names
// (get_/read_/search_) so the mechanism CAN fire, once into Claude Code natives to
// witness the name gate. This arm removes the hand-rolled transcript entirely: it
// replays the repo's CANONICAL, SHARED, named guard trace (testdata/guard-trace-e2e.json,
// the same fixture guard_trace_e2e_test.go and `fak guard --replay-trace` drive) through
// the REAL served-inline seam. That fixture carries real Claude Code native tool names
// (Read / Bash / Write) — the dominant path today — so it answers "measure the real
// proxy-dedup hit-rate on an agent trace" on a trace we did NOT author for this test.
//
// PROVENANCE (net-true doctrine, docs/standards/net-true-value.md):
//   - The SERVE is WITNESSED: fak's real adjudicateProposedServed + admitInboundResults
//     seam ran these calls in-process (the exact per-turn order messages.go drives:
//     warm the vDSO from the client's returned results, probe/serve the next turn's
//     proposals). Not a model of what it would do.
//   - The TRACE is the repo's canonical guard-trace-e2e fixture — AUTHORED (synthetic),
//     but shared with the guard harness and using real Claude Code native tool names.
//     It is NOT a captured live /v1/messages session; the promotion path (feed a captured
//     `fak guard -- claude` session) stays open and is named in the committed report.
//
// The WITNESSED result: on native Claude Code tool names the proxy-fill loop is fully
// INERT — 0 vDSO fills and 0 served_inline across the whole trace — because the read-only
// NAME gate (readOnlyPrefix: get_/read_/search_/list_/lookup_/find_/calc) matches none of
// Read/Bash/Write/Grep/Glob. The dominant-path dedup rate is 0%, and it is 0 for a
// STRUCTURAL reason (the name gate), independent of how redundant the reads are.
//
// Re-run: `go test ./internal/gateway -run TestServedInlineGuardTrace -count=1`
// Regenerate the committed report: `UPDATE_GOLDEN=1 go test ./internal/gateway -run TestServedInlineGuardTrace`

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// gtCall / gtTurn / gtTrace mirror the on-disk shape of testdata/guard-trace-e2e.json.
// We read only the fields this measurement needs (id, tool, args); the `class` field is
// the guard-floor verdict a DIFFERENT test asserts — this arm runs under allowAllAdj, so
// no call is ever denied and a read-only call drops out of `kept` ONLY when served inline.
type gtCall struct {
	ID   string          `json:"id"`
	Tool string          `json:"tool"`
	Args json.RawMessage `json:"args"`
}

type gtTurn struct {
	Calls []gtCall `json:"calls"`
}

type gtTrace struct {
	SliceID string   `json:"slice_id"`
	Turns   []gtTurn `json:"turns"`
}

// isNativeReadTool reports whether a Claude Code native tool NAME is a read (the honest
// denominator: the reads the mechanism would WANT to serve). readOnlyPrefix is folded in
// so a snake_case read (get_/read_/...) also counts — on this native-name trace it adds
// nothing, which is exactly the point being witnessed.
func isNativeReadTool(tool string) bool {
	switch tool {
	case "Read", "Grep", "Glob", "LS", "NotebookRead":
		return true
	}
	return readOnlyPrefix(tool)
}

type gtPerTurn struct {
	Turn             int `json:"turn"`
	ReadOnlyProposed int `json:"read_only_proposed"`
	NameEligible     int `json:"name_eligible"`
	ServedInline     int `json:"served_inline"`
	VDSOFillsAfter   int `json:"vdso_fills_after"`
}

type servedInlineGuardTraceReport struct {
	Schema           string            `json:"schema"`
	Issue            string            `json:"issue"`
	Provenance       map[string]string `json:"provenance"`
	TraceName        string            `json:"trace_name"`
	TraceTurns       int               `json:"trace_turns"`
	Saves            string            `json:"saves"`
	ReadOnlyProposed int               `json:"read_only_proposed"`
	NameEligible     int               `json:"name_eligible"`
	CrossTurnReread  int               `json:"cross_turn_reread"`
	ServedInline     int               `json:"served_inline"`
	VDSOFills        int               `json:"vdso_fills"`
	DedupRate        float64           `json:"dedup_rate"`
	PerTurn          []gtPerTurn       `json:"per_turn"`
	FallbackReason   string            `json:"fallback_reason"`
	Recommendation   string            `json:"recommendation"`
	Promotion        string            `json:"promotion"`
	Invalidating     string            `json:"invalidating_assumption"`
}

// replayGuardTrace drives the canonical guard trace through the real served-inline seam and
// folds the per-turn / whole-trace measurement. The warm loop is real: after each turn the
// client's KEPT read results are fed back through admitInboundResults, exactly as the proxy
// flow does, so a later turn COULD serve — the measured 0 is a fired-and-missed 0, not a
// never-attempted one.
func replayGuardTrace(t *testing.T, tr gtTrace) servedInlineGuardTraceReport {
	t.Helper()
	srv, v := newProxyFillServer(t)
	ctx := WithPrincipal(context.Background(), "tenantGuardTrace")

	rep := servedInlineGuardTraceReport{}
	seen := map[string]bool{} // (tool|args) of every read proposed in a PRIOR turn — for cross-turn re-read.

	for ti, turn := range tr.Turns {
		var calls []agent.ToolCall
		idToTool := map[string]string{}
		for _, c := range turn.Calls {
			calls = append(calls, agent.ToolCall{ID: c.ID, Type: "function",
				Function: agent.Func{Name: c.Tool, Arguments: string(c.Args)}})
			idToTool[c.ID] = c.Tool
		}

		kept, _, _, _, _ := srv.adjudicateProposedServed(ctx, calls, "guardtrace")
		keptID := map[string]bool{}
		for _, k := range kept {
			keptID[k.ID] = true
		}

		pt := gtPerTurn{Turn: ti}
		for _, c := range turn.Calls {
			if !isNativeReadTool(c.Tool) {
				continue // not a read — outside the read-only denominator
			}
			pt.ReadOnlyProposed++
			rep.ReadOnlyProposed++
			if readOnlyPrefix(c.Tool) {
				pt.NameEligible++
				rep.NameEligible++
			}
			key := c.Tool + "|" + normalizeArgs(string(c.Args))
			if seen[key] {
				rep.CrossTurnReread++
			}
			if !keptID[c.ID] { // a read-only call not kept (and never denied here) was served inline
				pt.ServedInline++
				rep.ServedInline++
			}
		}

		// Feed the KEPT results back — the real proxy warm loop (f2d1bec5). Native-name reads
		// are not read-only-shaped, so this warms nothing; that inertness is the finding.
		var inbound []agent.Message
		inbound = append(inbound, agent.Message{Role: agent.RoleUser, Content: "go"})
		var toolResults []agent.Message
		for _, c := range calls {
			if !keptID[c.ID] {
				continue
			}
			inbound = append(inbound, agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{c}})
			toolResults = append(toolResults, agent.Message{Role: agent.RoleTool, ToolCallID: c.ID,
				Name: c.Function.Name, Content: `{"result":"bytes for ` + c.ID + `"}`})
		}
		inbound = append(inbound, toolResults...)
		if len(toolResults) > 0 {
			if _, err := srv.admitInboundResults(ctx, inbound, nil, "guardtrace"); err != nil {
				t.Fatalf("turn %d admitInboundResults: %v", ti, err)
			}
		}

		// Now mark this turn's reads as seen for the NEXT turn's cross-turn-re-read check.
		for _, c := range turn.Calls {
			if isNativeReadTool(c.Tool) {
				seen[c.Tool+"|"+normalizeArgs(string(c.Args))] = true
			}
		}

		_, _, fills, _ := v.Stats()
		pt.VDSOFillsAfter = int(fills)
		rep.PerTurn = append(rep.PerTurn, pt)
	}

	_, _, fills, _ := v.Stats()
	rep.VDSOFills = int(fills)
	if rep.ReadOnlyProposed > 0 {
		rep.DedupRate = round4dedup(float64(rep.ServedInline) / float64(rep.ReadOnlyProposed))
	}
	return rep
}

// TestServedInlineGuardTrace is the re-runnable dominant-path witness for #1350. It replays
// the canonical guard-trace-e2e fixture through the real seam, asserts the honest
// invariants (native names never warm and never serve), and regenerates the committed
// golden report (UPDATE_GOLDEN=1).
func TestServedInlineGuardTrace(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "guard-trace-e2e.json"))
	if err != nil {
		t.Fatalf("read guard-trace-e2e.json: %v", err)
	}
	var tr gtTrace
	if err := json.Unmarshal(raw, &tr); err != nil {
		t.Fatalf("decode guard trace: %v", err)
	}
	if len(tr.Turns) == 0 {
		t.Fatal("guard trace has no turns")
	}

	rep := replayGuardTrace(t, tr)
	rep.Schema = "served_inline_guardtrace.v1"
	rep.Issue = "#1350"
	rep.TraceName = tr.SliceID
	rep.TraceTurns = len(tr.Turns)
	rep.Saves = "EXECUTIONS (the client tool round-trip of a duplicate/repeat read), NOT the surviving call's tokens"
	rep.Provenance = map[string]string{
		"serve":   "WITNESSED — replayed through fak's real adjudicateProposedServed + admitInboundResults seam in-process",
		"trace":   "guard-trace-e2e — the repo's canonical end-to-end guard trace fixture (AUTHORED/synthetic, real Claude Code native tool names Read/Bash/Write); NOT a captured live /v1/messages session",
		"command": "go test ./internal/gateway -run TestServedInlineGuardTrace -count=1",
	}
	rep.FallbackReason = "NAME_NOT_ELIGIBLE — readOnlyPrefix (get_/read_/search_/list_/lookup_/find_/calc) matches none of the native names Read/Bash/Write/Grep/Glob, so no native read is ever probed or filled"
	rep.Recommendation = "Keep --vdso-proxy-fill OPT-IN. On the canonical Claude Code-shaped trace the proxy-fill loop is fully inert (0 fills, 0 served_inline), consistent with the modeled native-name arm in served_inline_dedup_test.go. A miss is byte-identical to today, so it is safe to leave ON where reads are already read-only-shaped (MCP/snake_case), but it delivers ZERO on native Read/Grep/Glob until the name gate is widened."
	rep.Promotion = "Feed the SAME seam a CAPTURED live /v1/messages session (fak guard -- claude, or a tau2/SWE run) instead of this authored fixture; if that real session is redundant AND its reads are name-eligible, the WITNESSED rate promotes the lever toward default-on."
	rep.Invalidating = "The single assumption most likely to flip the result is the tool-NAMING regime: on real Claude Code traffic the read tool is Read (not read_*), so served_inline is 0 regardless of redundancy until readOnlyPrefix recognizes the native read tools."

	// Dominant-path invariants: native names neither warm the cache nor get served.
	if rep.ReadOnlyProposed == 0 {
		t.Fatal("measured zero read-only-proposed native reads — the fixture shape changed")
	}
	if rep.NameEligible != 0 {
		t.Fatalf("name_eligible=%d; want 0 — native Read/Bash/Grep must NOT match readOnlyPrefix", rep.NameEligible)
	}
	if rep.ServedInline != 0 {
		t.Fatalf("served_inline=%d; want 0 — native-name reads are never served (the name gate rejects them)", rep.ServedInline)
	}
	if rep.VDSOFills != 0 {
		t.Fatalf("vdso_fills=%d; want 0 — native-name results must not warm the proxy-fill cache", rep.VDSOFills)
	}
	if rep.DedupRate != 0 {
		t.Fatalf("dedup_rate=%v; want 0", rep.DedupRate)
	}

	t.Logf("guard-trace-e2e dominant-path: read-only-proposed=%d name-eligible=%d served_inline=%d vdso_fills=%d dedup=%.1f%%",
		rep.ReadOnlyProposed, rep.NameEligible, rep.ServedInline, rep.VDSOFills, rep.DedupRate*100)

	got, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		t.Fatalf("marshal report: %v", err)
	}
	golden := filepath.Join("testdata", "served_inline_guardtrace_report.json")
	if os.Getenv("UPDATE_GOLDEN") != "" {
		if err := os.WriteFile(golden, append(got, '\n'), 0o644); err != nil {
			t.Fatalf("write golden: %v", err)
		}
		t.Logf("updated golden %s", golden)
		return
	}
	want, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("read golden (run with UPDATE_GOLDEN=1 to create): %v", err)
	}
	wantNorm := bytes.TrimRight(bytes.ReplaceAll(want, []byte("\r\n"), []byte("\n")), "\n")
	gotNorm := bytes.TrimRight(bytes.ReplaceAll(got, []byte("\r\n"), []byte("\n")), "\n")
	if !bytes.Equal(wantNorm, gotNorm) {
		t.Errorf("report drifted from golden %s; re-run with UPDATE_GOLDEN=1 if intended", golden)
	}
}

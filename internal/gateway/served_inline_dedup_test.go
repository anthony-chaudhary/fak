package gateway

// served_inline_dedup_test.go — the WITNESSED measurement for issue #1350:
// "measure the real proxy-dedup hit-rate (served_inline) on an agent trace".
//
// We shipped the mechanism (serve duplicate read-only tool calls inline from the
// vDSO: 36122bc9) and the proxy cache-warming that makes it fire on a pure proxy
// (f2d1bec5), and we count fak_gateway_served_inline_total — but until now had NO
// measurement of how often it actually fires on a real read pattern. The headline
// value (saving the EXECUTION of duplicate/repeat reads) was a reasoned claim with a
// zero baseline. This replays a named, reproducible coding-session trace through the
// REAL served-inline seam (admitInboundResults -> adjudicateProposedServed, the exact
// per-turn order messages.go drives) and reads out the dedup rate, decomposed by win
// class, plus a recommendation on whether --vdso-proxy-fill should default on.
//
// PROVENANCE (net-true doctrine, docs/standards/net-true-value.md):
//   - The SERVE is WITNESSED: fak's real adjudication seam actually served these calls
//     in-process (not a model of what it would do). A served hit drops the tool_use so
//     the client never re-runs it — it saves the EXECUTION (the client tool round-trip),
//     NOT the surviving call's tokens (the honesty fence from the served-inline commit).
//   - The TRACE redundancy is MODELED: a representative constructed transcript with a
//     named, reproducible read pattern (the three win classes + non-repeated reads +
//     one write + one force-fresh). The promotion path is to feed the SAME harness a
//     captured real /v1/messages session; the invalidating assumption is the tool-NAMING
//     regime (see the claude-native arm below), which flips the real-Claude-Code rate to 0.
//
// Re-run: `go test ./internal/gateway -run TestServedInlineDedup -count=1`
// Regenerate the committed report: `UPDATE_GOLDEN=1 go test ./internal/gateway -run TestServedInlineDedup`

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// dedupTraceCall is one proposed tool call in the modeled transcript. write=true marks
// a write-shaped tool (never vDSO-eligible); fresh=true sets the _fak_fresh escape hatch.
type dedupTraceCall struct {
	tool  string
	args  string
	write bool
	fresh bool
}

// dedupTraceTurn is one model turn: the calls it proposes. In the real /v1/messages
// flow the client runs the kept calls and returns their results, which admitInboundResults
// warms into the vDSO for the NEXT turn — exactly what the harness replays below.
type dedupTraceTurn struct {
	calls []dedupTraceCall
}

// representativeCodingTrace is the named, reproducible read pattern the report measures.
// It uses READ-ONLY-SHAPED tool names (get_/read_/search_/list_/... — the readOnlyPrefix
// family the serve gate recognizes) so the mechanism CAN fire; the claude-native arm below
// re-runs the same shape under Read/Grep/Glob names to witness the name gate. It embeds:
//   - W3 cross-turn re-read: a file/dir/search re-proposed in a later turn after the client
//     already returned its result (the win the mechanism is built for).
//   - W1/W2 within-turn parallel/same-call-twice: two identical reads in ONE turn while the
//     cache is cold — the mechanism has no prior fill to serve from, so both survive.
//   - non-repeated reads (the denominator) + one write (ineligible) + one force-fresh re-read.
var representativeCodingTrace = []dedupTraceTurn{
	{calls: []dedupTraceCall{ // turn 1: first exploration — all cold, all client-run.
		{tool: "list_dir", args: `{"path":"."}`},
		{tool: "read_file", args: `{"path":"strpad.go"}`},
		{tool: "read_file", args: `{"path":"README.md"}`},
	}},
	{calls: []dedupTraceCall{ // turn 2: re-read strpad.go (W3 cross-turn) + new reads.
		{tool: "read_file", args: `{"path":"strpad.go"}`},
		{tool: "search_code", args: `{"pattern":"Pad"}`},
		{tool: "read_file", args: `{"path":"strpad_test.go"}`},
	}},
	{calls: []dedupTraceCall{ // turn 3: two identical cold reads in ONE turn (W1/W2).
		{tool: "read_file", args: `{"path":"util.go"}`},
		{tool: "read_file", args: `{"path":"util.go"}`},
	}},
	{calls: []dedupTraceCall{ // turn 4: re-read README (W3) + a new config read.
		{tool: "read_file", args: `{"path":"README.md"}`},
		{tool: "get_config", args: `{"key":"app"}`},
	}},
	{calls: []dedupTraceCall{ // turn 5: re-read strpad.go (W3) + a write + re-read util.go (W3).
		{tool: "read_file", args: `{"path":"strpad.go"}`},
		{tool: "edit_file", args: `{"path":"strpad.go","content":"..."}`, write: true},
		{tool: "read_file", args: `{"path":"util.go"}`},
	}},
	{calls: []dedupTraceCall{ // turn 6: re-search (W3) + a force-fresh re-read (escape hatch declines the serve).
		{tool: "search_code", args: `{"pattern":"Pad"}`},
		{tool: "read_file", args: `{"path":"strpad.go","_fak_fresh":true}`, fresh: true},
	}},
}

// win-class labels for the decomposition (structural, derived from the trace, not from
// the measured outcome — so the report can say "class X was proposed N times, served M").
const (
	winFirstOccurrence = "first_occurrence"     // a (tool,args) not seen earlier — never served (expected).
	winCrossTurnReread = "cross_turn_reread"    // W3: a repeat whose result a PRIOR turn already returned.
	winWithinTurnDup   = "within_turn_parallel" // W1/W2: a repeat of an earlier call in the SAME turn (cold).
	winForceFreshDecl  = "force_fresh_declined" // _fak_fresh set: the model opted out of the serve.
)

type classCount struct {
	Proposed int `json:"proposed"`
	Served   int `json:"served"`
}

type regimeResult struct {
	ToolNameShape    string                `json:"tool_name_shape"`
	ReadOnlyProposed int                   `json:"read_only_proposed"`
	ServedInline     int                   `json:"served_inline"`
	DedupRate        float64               `json:"dedup_rate"`
	ByWinClass       map[string]classCount `json:"by_win_class"`
}

type dedupRecommendation struct {
	DefaultOn bool   `json:"default_on"`
	Rationale string `json:"rationale"`
	Gate      string `json:"gate"`
}

type servedInlineDedupReport struct {
	Schema         string              `json:"schema"`
	Provenance     map[string]string   `json:"provenance"`
	TraceName      string              `json:"trace_name"`
	TraceTurns     int                 `json:"trace_turns"`
	Saves          string              `json:"saves"`
	EligibleRegime regimeResult        `json:"eligible_name_regime"`
	ClaudeNative   regimeResult        `json:"claude_native_name_regime"`
	Recommendation dedupRecommendation `json:"recommendation"`
	Assumptions    []string            `json:"assumptions"`
	Promotion      string              `json:"promotion"`
	Demotion       string              `json:"demotion_or_retirement"`
	Invalidating   string              `json:"invalidating_assumption"`
}

// round4 rounds to 4 decimals so the golden JSON is stable across platforms.
func round4dedup(f float64) float64 {
	return float64(int64(f*10000+0.5)) / 10000
}

// classifyTrace assigns each read-only-shaped proposed call a structural win class, keyed
// by (turnIdx, callIdx). Write-shaped calls are excluded (not vDSO-eligible). The class is
// derived from the trace shape alone — whether a (tool,args) repeats an earlier call, and
// whether that earlier call was in a prior turn (its result already returned) or the same
// turn (still cold).
func classifyTrace(turns []dedupTraceTurn) map[[2]int]string {
	out := map[[2]int]string{}
	// key -> the (turnIdx,callIdx) of its first appearance, used to tell prior-turn from same-turn.
	firstTurn := map[string]int{}
	for ti, turn := range turns {
		seenThisTurn := map[string]bool{}
		for ci, c := range turn.calls {
			if c.write {
				continue // write-shaped: excluded from the read-only denominator entirely.
			}
			key := c.tool + "|" + normalizeArgs(c.args)
			switch {
			case c.fresh:
				out[[2]int{ti, ci}] = winForceFreshDecl
			case seenThisTurn[key]:
				out[[2]int{ti, ci}] = winWithinTurnDup
			case func() bool { ft, ok := firstTurn[key]; return ok && ft < ti }():
				out[[2]int{ti, ci}] = winCrossTurnReread
			default:
				out[[2]int{ti, ci}] = winFirstOccurrence
			}
			seenThisTurn[key] = true
			if _, ok := firstTurn[key]; !ok {
				firstTurn[key] = ti
			}
		}
	}
	return out
}

// normalizeArgs strips the _fak_fresh marker so a force-fresh re-read still keys to the
// same (tool,args) as its warm original for classification purposes.
func normalizeArgs(args string) string {
	var m map[string]json.RawMessage
	if json.Unmarshal([]byte(args), &m) != nil {
		return args
	}
	delete(m, "_fak_fresh")
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	for _, k := range keys {
		b.WriteString(k)
		b.WriteByte('=')
		b.Write(m[k])
		b.WriteByte(';')
	}
	return b.String()
}

// replayDedupTrace drives the trace through the real served-inline seam and returns the
// per-regime measurement. rename maps a read-only-shaped tool name to the name actually
// proposed on the wire (identity for the eligible regime; Read/Grep/... for the native one).
func replayDedupTrace(t *testing.T, turns []dedupTraceTurn, classes map[[2]int]string, rename map[string]string) regimeResult {
	t.Helper()
	srv, _ := newProxyFillServer(t)
	ctx := WithPrincipal(context.Background(), "tenantDedup")

	res := regimeResult{ByWinClass: map[string]classCount{}}
	wireName := func(tool string) string {
		if rename == nil {
			return tool
		}
		if n, ok := rename[tool]; ok {
			return n
		}
		return tool
	}

	for ti, turn := range turns {
		// Build the model's proposed calls for this turn with stable IDs so we can diff
		// which survived to the wire (kept) from which were served inline (dropped).
		var calls []agent.ToolCall
		idToPos := map[string][2]int{}
		for ci, c := range turn.calls {
			id := "t" + itoaDedup(ti) + "c" + itoaDedup(ci)
			calls = append(calls, agent.ToolCall{ID: id, Type: "function",
				Function: agent.Func{Name: wireName(c.tool), Arguments: c.args}})
			idToPos[id] = [2]int{ti, ci}
		}

		// The real per-turn order: proposed calls are adjudicated against the CURRENT vDSO
		// (warmed by prior turns), a fresh hit is served inline and dropped from kept.
		kept, _, _, _, servedHits := srv.adjudicateProposedServed(ctx, calls, "dedup-trace")

		keptID := map[string]bool{}
		for _, k := range kept {
			keptID[k.ID] = true
		}
		servedCount := 0
		for _, c := range calls {
			pos := idToPos[c.ID]
			cls, isReadOnly := classes[pos]
			if !isReadOnly {
				continue // write-shaped: not part of the read-only measurement.
			}
			cc := res.ByWinClass[cls]
			cc.Proposed++
			if !keptID[c.ID] { // a read-only call not kept was served inline (allowAllAdj never drops).
				cc.Served++
				servedCount++
			}
			res.ByWinClass[cls] = cc
			res.ReadOnlyProposed++
		}
		res.ServedInline += servedCount
		if servedCount != servedHits {
			t.Fatalf("turn %d: served-by-diff %d != servedHits %d (kept/served accounting drifted)", ti, servedCount, servedHits)
		}

		// The client runs every KEPT read-only call and returns its result; feed those back
		// through admitInboundResults so the proxy-fill warms the vDSO for later turns —
		// exactly the loop f2d1bec5 closed. Served calls produced no execution (already warm).
		var inbound []agent.Message
		inbound = append(inbound, agent.Message{Role: agent.RoleUser, Content: "go"})
		var toolResults []agent.Message
		for _, c := range calls {
			pos := idToPos[c.ID]
			if _, isReadOnly := classes[pos]; !isReadOnly {
				continue
			}
			if !keptID[c.ID] {
				continue // served inline — not executed by the client, no new result.
			}
			inbound = append(inbound, agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{c}})
			toolResults = append(toolResults, agent.Message{Role: agent.RoleTool, ToolCallID: c.ID,
				Name: c.Function.Name, Content: `{"result":"bytes for ` + c.ID + `"}`, Witness: "witness-v1"})
		}
		inbound = append(inbound, toolResults...)
		if len(toolResults) > 0 {
			if _, err := srv.admitInboundResults(ctx, inbound, nil, "dedup-trace"); err != nil {
				t.Fatalf("turn %d admitInboundResults: %v", ti, err)
			}
		}
	}
	if res.ReadOnlyProposed > 0 {
		res.DedupRate = round4dedup(float64(res.ServedInline) / float64(res.ReadOnlyProposed))
	}
	return res
}

func itoaDedup(i int) string {
	if i == 0 {
		return "0"
	}
	var b [4]byte
	n := len(b)
	for i > 0 {
		n--
		b[n] = byte('0' + i%10)
		i /= 10
	}
	return string(b[n:])
}

// buildServedInlineDedupReport folds both regimes into the committed report.
func buildServedInlineDedupReport(t *testing.T) servedInlineDedupReport {
	t.Helper()
	classes := classifyTrace(representativeCodingTrace)

	eligible := replayDedupTrace(t, representativeCodingTrace, classes, nil)
	eligible.ToolNameShape = "read-only-prefixed (get_/read_/search_/list_/lookup_/find_/calc)"

	// Claude Code native tool names do NOT match readOnlyPrefix, so the serve gate rejects
	// every one — the honest reason the rate is 0 on the dominant path today.
	native := replayDedupTrace(t, representativeCodingTrace, classes, map[string]string{
		"read_file": "Read", "search_code": "Grep", "list_dir": "Glob", "get_config": "ReadConfig",
	})
	native.ToolNameShape = "Claude Code native (Read/Grep/Glob/...) — unmatched by readOnlyPrefix"

	rec := dedupRecommendation{
		DefaultOn: false,
		Rationale: "On a redundant read pattern with read-only-SHAPED tool names the mechanism " +
			"serves every cross-turn re-read (a real, fail-safe execution saving), but it fires ZERO " +
			"times on the dominant Claude Code path: the read-only NAME gate (get_/read_/search_/...) " +
			"does not recognize Read/Grep/Glob, so no native read is ever eligible. A miss is byte-" +
			"identical to today (TestServedInline_Miss), so the lever is safe to leave ON where it pays.",
		Gate: "Default --vdso-proxy-fill ON only after the serve eligibility classifier recognizes the " +
			"harness's real read-tool names (extend readOnlyPrefix / add a shape classifier for Read/Grep/" +
			"Glob), OR for MCP/snake_case tool deployments where reads are already get_/read_/search_ shaped. " +
			"Until then keep it opt-in.",
	}

	return servedInlineDedupReport{
		Schema: "served_inline_dedup.v1",
		Provenance: map[string]string{
			"serve":        "WITNESSED — fak's real adjudicateProposedServed seam served these calls in-process",
			"trace":        "MODELED — a named, reproducible representative coding transcript",
			"command":      "go test ./internal/gateway -run TestServedInlineDedup -count=1",
			"generated_by": "fak/internal/gateway.buildServedInlineDedupReport",
		},
		TraceName:      "representative-coding-session-v1",
		TraceTurns:     len(representativeCodingTrace),
		Saves:          "EXECUTIONS (the client tool round-trip of a duplicate/repeat read), NOT the surviving call's tokens",
		EligibleRegime: eligible,
		ClaudeNative:   native,
		Recommendation: rec,
		Assumptions: []string{
			"The trace's redundancy (files/searches re-read across turns) is representative of a real coding session; a session that never repeats a read measures near-zero (a valid result that down-ranks the lever).",
			"Tool NAMING regime drives eligibility: only get_/read_/search_/list_/lookup_/find_/calc-prefixed reads are served. Claude Code's native Read/Grep/Glob are not, so the native-name arm measures 0.",
			"Within-turn parallel/same-call-twice duplicates (W1/W2) are cold (no prior fill), so THIS mechanism does not dedup them — only cross-turn re-reads (W3) after the client returned a result are served.",
			"A force-fresh (_fak_fresh) re-read declines the serve by design (escape hatch), so it counts as read-only-proposed but never served.",
		},
		Promotion: "Feed the SAME harness a captured real /v1/messages session (fak guard -- claude, or a tau2/SWE run) " +
			"as the trace; if the real read pattern is redundant AND uses eligible tool names, the WITNESSED rate promotes the lever toward default-on.",
		Demotion: "If a captured real session shows near-zero cross-turn re-read (agents rarely re-read), or if the dominant " +
			"harness keeps native Read/Grep names unrecognized, the lever is demoted to opt-in and the value claim is retired for that path.",
		Invalidating: "The single assumption most likely to flip the result is the tool-NAMING regime: on real Claude Code traffic " +
			"the read tool is Read (not read_*), so served_inline is 0 regardless of redundancy until the name gate is widened.",
	}
}

// TestServedInlineDedupMeasurement is the re-runnable witness for #1350. It replays the
// named trace through the real seam, asserts the honest invariants the report rests on,
// and regenerates the committed golden report (UPDATE_GOLDEN=1).
func TestServedInlineDedupMeasurement(t *testing.T) {
	r := buildServedInlineDedupReport(t)

	// Eligible-name regime: the mechanism serves the cross-turn re-reads, and ONLY those.
	e := r.EligibleRegime
	if e.ReadOnlyProposed == 0 {
		t.Fatal("eligible regime measured zero read-only-proposed calls — trace is empty")
	}
	if e.ServedInline == 0 {
		t.Fatal("eligible regime served ZERO inline — the mechanism should serve cross-turn re-reads")
	}
	if got := e.ByWinClass[winWithinTurnDup]; got.Proposed == 0 || got.Served != 0 {
		t.Fatalf("within-turn dups: proposed=%d served=%d; want proposed>0 served=0 (cold intra-turn is NOT deduped)", got.Proposed, got.Served)
	}
	if got := e.ByWinClass[winCrossTurnReread]; got.Proposed == 0 || got.Served != got.Proposed {
		t.Fatalf("cross-turn re-reads: proposed=%d served=%d; want all served", got.Proposed, got.Served)
	}
	if got := e.ByWinClass[winForceFreshDecl]; got.Served != 0 {
		t.Fatalf("force-fresh re-read served=%d; want 0 (escape hatch declines the serve)", got.Served)
	}
	if got := e.ByWinClass[winFirstOccurrence]; got.Served != 0 {
		t.Fatalf("first-occurrence reads served=%d; want 0 (nothing warm to serve)", got.Served)
	}
	if e.ServedInline != e.ByWinClass[winCrossTurnReread].Served {
		t.Fatalf("served_inline %d != cross-turn re-reads served %d — an unexpected class was served", e.ServedInline, e.ByWinClass[winCrossTurnReread].Served)
	}

	// Claude-native-name regime: the read-only NAME gate rejects Read/Grep/Glob, so the SAME
	// redundant read pattern serves ZERO — the finding that answers "should it default on?".
	if r.ClaudeNative.ServedInline != 0 {
		t.Fatalf("claude-native regime served %d inline; want 0 (Read/Grep/Glob are not read-only-prefixed)", r.ClaudeNative.ServedInline)
	}

	// Net-true hygiene: the report names provenance, assumptions, and promotion/demotion.
	if r.Provenance["serve"] == "" || r.Provenance["trace"] == "" {
		t.Error("report must label serve (WITNESSED) and trace (MODELED) provenance")
	}
	if len(r.Assumptions) == 0 || r.Promotion == "" || r.Demotion == "" || r.Invalidating == "" {
		t.Error("report must name assumptions + promotion + demotion + invalidating assumption")
	}
	if r.Recommendation.Rationale == "" || r.Recommendation.Gate == "" {
		t.Error("report must carry a default-on recommendation backed by the measured rate")
	}

	t.Logf("eligible-name dedup rate = %.1f%% (%d/%d served); claude-native = %d served",
		e.DedupRate*100, e.ServedInline, e.ReadOnlyProposed, r.ClaudeNative.ServedInline)

	got, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		t.Fatalf("marshal report: %v", err)
	}
	golden := filepath.Join("testdata", "served_inline_dedup_report.json")
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

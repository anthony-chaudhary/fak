package gateway

// The #2889 first slice (Hermes-inspiration epic #2871): a subagent SCRIPT calls
// fak_syscall in a loop over the gateway's JSON-RPC wire, collapsing a 5-step
// pipeline into ONE model turn. Hermes gets the context saving by letting a
// script call tools via RPC (tools/delegate_tool.py + gateway JSON-RPCs); fak
// gets the SAME saving without dropping the floor, because the RPC target is the
// kernel — every scripted call is still individually adjudicated and journaled
// as its own tool process.
//
// The frames below are byte-identical to what an external script writes on the
// stdio wire: dispatchRPC is the same function ServeStdio feeds one line at a
// time, so this file IS the minimal script-calls-fak_syscall-in-a-loop example.
// The chaining (step i consumes step i-1's output) happens in the script, never
// in model context.
//
// gen/next bookkeeping (#2889):
//   - Promotion evidence (toward now): this contract green + the logged token
//     readout below; promotion needs a real subagent script against a live
//     gateway with provider-OBSERVED input tokens replacing the ESTIMATED
//     divisor.
//   - Demotion/retirement evidence: provider-observed accounting showing the
//     one-turn saving does not materialize, or a native MCP batch/pipeline verb
//     obsoleting the loop shape.
//   - Invalidating assumption: the baseline prices every model turn as a full
//     resend of the tool-schema floor plus prior exchanges with NO provider
//     prompt caching; strong prefix caching shrinks the baseline's marginal
//     cost and would overstate the saving measured here.

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

const scriptedPipelineSteps = 5

// scriptedPipelineRun is the wire evidence of one scripted loop: the exact bytes
// the script wrote and read per step, plus each step's decoded kernel response.
type scriptedPipelineRun struct {
	reqBytes  []int
	respBytes []int
	responses []SyscallResponse
}

// runScriptedSyscallPipeline plays the subagent script: five fak_syscall frames
// through the real dispatch path in one process turn, each step threading the
// previous step's executed output into its arguments.
func runScriptedSyscallPipeline(t *testing.T, srv *Server) scriptedPipelineRun {
	t.Helper()
	var run scriptedPipelineRun
	prev := ""
	for step := 1; step <= scriptedPipelineSteps; step++ {
		args := fmt.Sprintf(`{"step":%d,"input":%q}`, step, prev)
		frame := fmt.Sprintf(`{"jsonrpc":"2.0","id":%d,"method":"tools/call",`+
			`"params":{"name":"fak_syscall","arguments":{"tool":"allow_step_%d","arguments":%s,"trace_id":"s2889-step-%d"}}}`,
			step, step, args, step)
		resp := srv.dispatchRPC(context.Background(), []byte(frame))
		if resp == nil || resp.Error != nil {
			t.Fatalf("step %d: fak_syscall round-trip failed: %+v", step, resp)
		}
		wire, err := json.Marshal(resp.Result)
		if err != nil {
			t.Fatalf("step %d: marshal result: %v", step, err)
		}
		var wrap struct {
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
			IsError bool `json:"isError"`
		}
		if err := json.Unmarshal(wire, &wrap); err != nil || len(wrap.Content) != 1 || wrap.IsError {
			t.Fatalf("step %d: tool-result wrap = %s (err=%v)", step, wire, err)
		}
		var sr SyscallResponse
		if err := json.Unmarshal([]byte(wrap.Content[0].Text), &sr); err != nil {
			t.Fatalf("step %d: decode SyscallResponse: %v", step, err)
		}
		if sr.Result == nil {
			t.Fatalf("step %d: no executed result (verdict %+v) — the scripted call must run, not just adjudicate", step, sr.Verdict)
		}
		run.reqBytes = append(run.reqBytes, len(frame))
		run.respBytes = append(run.respBytes, len(wire))
		run.responses = append(run.responses, sr)
		prev = sr.Result.Content
	}
	return run
}

// Each scripted call is individually adjudicated AND journaled as its own tool
// process — the floor the script-calls-RPC shape must keep (the #2880-#2882
// bypass risk is exactly a scripted path that skips this).
func TestScriptedSyscallPipelineOneTurnEachCallAdjudicated(t *testing.T) {
	journal := mcpToolprocTestJournal(t)
	srv := newTestServer(t)
	run := runScriptedSyscallPipeline(t, srv)

	for i, sr := range run.responses {
		step := i + 1
		if sr.Verdict.Kind != "ALLOW" {
			t.Errorf("step %d: verdict = %+v, want its own ALLOW", step, sr.Verdict)
		}
		if want := fmt.Sprintf("s2889-step-%d", step); sr.TraceID != want {
			t.Errorf("step %d: trace = %q, want %q (per-call identity, not one blanket trace)", step, sr.TraceID, want)
		}
		if sr.Result.Status != "OK" {
			t.Errorf("step %d: result status = %q, want OK", step, sr.Result.Status)
		}
		var echoed struct {
			Step  int    `json:"step"`
			Input string `json:"input"`
		}
		if err := json.Unmarshal([]byte(sr.Result.Content), &echoed); err != nil {
			t.Fatalf("step %d: decode executed args: %v", step, err)
		}
		if echoed.Step != step {
			t.Errorf("step %d: executed args carry step %d — not this call's arguments", step, echoed.Step)
		}
		if step > 1 && !strings.Contains(echoed.Input, fmt.Sprintf(`"step":%d`, step-1)) {
			t.Errorf("step %d: input does not thread step %d's output — a loop, not a pipeline", step, step-1)
		}
	}

	tab := foldTestJournal(t, journal)
	if len(tab.Procs) != scriptedPipelineSteps {
		t.Fatalf("journal procs = %d, want %d (one witnessed tool process per scripted call)", len(tab.Procs), scriptedPipelineSteps)
	}
	for step := 1; step <= scriptedPipelineSteps; step++ {
		id, tool := fmt.Sprintf("s2889-step-%d", step), fmt.Sprintf("allow_step_%d", step)
		found := false
		for _, p := range tab.Procs {
			if p.CallID != id {
				continue
			}
			found = true
			if p.Tool != tool || p.State != toolproc.StateDone {
				t.Errorf("proc %s = %s/%s, want %s DONE", id, p.Tool, p.State, tool)
			}
		}
		if !found {
			t.Errorf("no journaled tool process for %s — a scripted call escaped the witness", id)
		}
	}
}

// The measured context-token saving of the one-turn scripted pipeline vs the
// turn-per-step baseline — the #2889 witness. Every byte in the comparison is
// wire-witnessed from THIS run (the script's own frames, plus the live
// tools/list schema block as the per-turn floor a harness re-sends every model
// turn); tokens use the house ~4-bytes/token ESTIMATED divisor and are labeled
// as such, never as observed usage.
//
// Baseline (turn-per-step): 5 model turns; turn i's input re-reads the floor
// plus every prior request/response exchange (they live in the transcript).
// Scripted (one turn): the floor once, plus — conservatively — all five
// exchanges charged into context once; in reality the script's RPC traffic
// never enters the model window at all.
func TestScriptedSyscallPipelineContextTokenSaving(t *testing.T) {
	mcpToolprocTestJournal(t) // keep the tool-process journal in a temp dir
	srv := newTestServer(t)
	run := runScriptedSyscallPipeline(t, srv)

	floorResp := rpcRoundTrip(t, srv, "tools/list", "")
	floorWire, err := json.Marshal(floorResp.Result)
	if err != nil {
		t.Fatalf("marshal tools/list: %v", err)
	}
	floor := len(floorWire)
	if floor == 0 {
		t.Fatal("tools/list floor is empty — nothing to price the per-turn resend with")
	}

	baseline, hist := 0, 0
	for i := 0; i < scriptedPipelineSteps; i++ {
		baseline += floor + hist
		hist += run.reqBytes[i] + run.respBytes[i]
	}
	scripted := floor + hist

	baseTok, scriptTok := baseline/estBytesPerToken, scripted/estBytesPerToken
	savedTok := baseTok - scriptTok
	pct := 100 * float64(savedTok) / float64(baseTok)

	t.Logf("#2889 scripted-RPC pipeline context readout (provenance %s, ~%d bytes/token):", agent.FootprintProvenance, estBytesPerToken)
	t.Logf("  per-turn floor (live tools/list schemas): %d bytes", floor)
	t.Logf("  turn-per-step baseline (%d turns):         %d bytes = %d tokens", scriptedPipelineSteps, baseline, baseTok)
	t.Logf("  scripted one-turn (conservative):          %d bytes = %d tokens", scripted, scriptTok)
	t.Logf("  saving:                                    %d tokens (%.1f%%)", savedTok, pct)

	if scripted >= baseline {
		t.Fatalf("scripted one-turn cost %d >= baseline %d — the collapse saved nothing", scripted, baseline)
	}
	// Demonstration floor, not physics: with the real registry floor the saving
	// lands near 80%; even a floor comparable to the exchange bytes stays >50%.
	if pct < 50 {
		t.Fatalf("saving = %.1f%%, below the 50%% demonstration floor (floor=%dB, exchanges=%dB)", pct, floor, hist)
	}
}

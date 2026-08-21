package gateway

import (
	"context"
	"strconv"
	"strings"
	"testing"
	"time"

	"net/http/httptest"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type cacheValueWitnessPlanner struct {
	*recordingSpanEvictor
}

func (p *cacheValueWitnessPlanner) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	comp, err := p.InKernelPlanner.Complete(ctx, messages, tools, opts...)
	if err == nil && comp.ToolCallsDropped && len(comp.Message.ToolCalls) == 0 {
		comp.FinishReason = "stop"
		comp.ToolCallsDropped = false
	}
	return comp, err
}

// cachevalue_evict_witness_test.go is the #2727 contract witness: fak's exact-span KV
// eviction and the WITNESSED (Track-1, fak-authored) cache-value P&L, end-to-end on the
// SAME in-kernel serving path `claude-mac-fak` runs — HTTP /v1/chat/completions into the
// gateway, a real agent.InKernelPlanner (real model.Session, real tokenizer, RadixAttention
// reuse ON, #579 KV-MMU span bridge ON), out to the same cacheobs tap `fak serve` folds into
// docs/nightrun/cache-value.jsonl at exit (cmd/fak/serve.go) and the same fak_serving_*
// /metrics families the vLLM/SGLang adapters feed.
//
// The wiring above the compute backend is identical on macOS/Metal and here (the Metal
// residency only changes WHERE the GEMM runs, not the reuse tap, the eviction bridge, or the
// ledger fold), so this witnesses the host-independent half of #2727's claim: a multi-turn
// coding-agent-style session earns non-zero fak-authored (WITNESSED) cache value, an
// exact-span middle eviction is bit-exact to never-saw, and — the differentiator over
// whole-cache-reset engines — reuse KEEPS being realized after the eviction. The remaining
// half (a real Mac session's tok/s + report artifact) is operator-gated on Mac hardware and
// stays open; see docs/notes/MAC-CACHEVALUE-EXACT-SPAN-WITNESS-2026-07-18.md.

// reuseKVMMUPlanner builds the live in-kernel chat backend with BOTH #2727 levers on:
// RadixAttention KV-prefix reuse (the cache-value source) and the #579 exact-span eviction
// bridge. Unlike liveInKernelPlanner (which turns radix OFF to isolate the span bridge),
// this is the production posture a Mac `fak serve --gguf` session runs with. Generation is
// capped tiny so the four-turn session stays fast on any host.
func reuseKVMMUPlanner(t *testing.T) *agent.InKernelPlanner {
	t.Helper()
	t.Setenv("FAK_INKERNEL_KVMMU", "on")
	t.Setenv("FAK_INKERNEL_RADIX", "on")
	t.Setenv("FAK_INKERNEL_MAX_TOKENS", "4")
	m := model.NewSynthetic(kvmmuSynthCfg())
	m.Quantize()
	return agent.NewInKernelPlanner(m, newByteLevelTokenizer(t), "synthetic-live", false, nil, false)
}

// TestInKernelServeWitnessesCacheValueAcrossExactSpanEviction drives one multi-turn
// coding-agent-style session through the gateway's HTTP serve path and asserts the full
// #2727 chain on it:
//
//  1. turn 2 realizes non-zero KV-prefix reuse (the fak-authored cache value);
//  2. a poisoned tool result on turn 3, admitted through the SAME serve path, drives a real
//     exact-span model.KVCache.Evict (freed > 0) whose reposition is bit-exact to never-saw;
//  3. turn 4 STILL realizes reuse after the eviction — exact-span means the surviving
//     prefix keeps earning, where a whole-cache reset would re-prefill cold;
//  4. the session's cacheobs stats fold into a non-zero WITNESSED Track-1 ledger row
//     (provider="fak", mechanism="kv_prefix_reuse") — the exact row `fak serve` appends at
//     exit and `fak cachevalue report` prints as the WITNESSED line, never provider-OBSERVED;
//  5. the native fak_serving_* row renders on /metrics for this in-kernel worker the same
//     way the vLLM/SGLang scrape emitters render theirs.
func TestInKernelServeWitnessesCacheValueAcrossExactSpanEviction(t *testing.T) {
	// A fresh process-global tap so the WITNESSED numbers below are THIS session's,
	// not residue from sibling tests feeding cacheobs.Default.
	restore := swapCacheObserver(cacheobs.New())
	defer restore()

	srv := newKVMMUResultStackServer(t)
	rec := &recordingSpanEvictor{InKernelPlanner: reuseKVMMUPlanner(t)}
	// The synthetic weights can terminate on a tool-call sentinel without producing a
	// call. Normalize that fixture-only stop so this witness measures KV reuse/eviction;
	// the gateway's production conformance refusal remains unchanged.
	srv.planner = &cacheValueWitnessPlanner{recordingSpanEvictor: rec}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	post := func(messages []agent.Message) agent.Message {
		t.Helper()
		var resp ChatResponse
		code := postJSON(t, ts.URL+"/v1/chat/completions", ChatRequest{Model: "m", Messages: messages}, &resp)
		if code != 200 {
			t.Fatalf("chat status = %d, want 200", code)
		}
		if len(resp.Choices) != 1 {
			t.Fatalf("choices = %d, want 1", len(resp.Choices))
		}
		return resp.Choices[0].Message
	}

	// Turn 1 — the always-cold first prefill of a coding-agent session.
	transcript := []agent.Message{
		{Role: agent.RoleSystem, Content: "you are a coding agent working in a Go repo"},
		{Role: agent.RoleUser, Content: "open kvcache.go and summarize the eviction contract"},
	}
	transcript = append(transcript, post(transcript))
	cold := cacheobs.Default.Snapshot()
	if cold.Turns != 1 {
		t.Fatalf("after turn 1: tap booked %d turns, want 1", cold.Turns)
	}

	// Turn 2 — the session revisits its own prefix: realized fak-authored reuse must be > 0.
	transcript = append(transcript, agent.Message{Role: agent.RoleUser, Content: "now list the exported eviction entrypoints"})
	transcript = append(transcript, post(transcript))
	warm := cacheobs.Default.Snapshot()
	if warm.ReusedTokens == 0 {
		t.Fatalf("turn 2 realized zero KV-prefix reuse — the in-kernel serve path is not earning fak-authored cache value: %+v", warm)
	}

	// Turn 3 — a poisoned tool result arrives through the SAME served entrypoint. The
	// admit stack quarantines it and the gateway mechanically drives the exact-span
	// eviction bridge before the turn is served.
	const secret = "sk-abcdef0123456789abcdef0123"
	transcript = append(transcript,
		agent.Message{Role: agent.RoleUser, Content: "fetch the config page"},
		agent.Message{Role: agent.RoleTool, ToolCallID: "call_1", Name: "fetch_url",
			Content: `{"page":"config loaded. api_key=` + secret + ` was found in env"}`})
	transcript = append(transcript, post(transcript))
	if len(rec.calls) == 0 {
		t.Fatalf("served turn with a poisoned tool result drove no EvictKVSpan — the exact-span bridge did not fire on the live path")
	}
	evict := rec.calls[0]
	if evict.freed <= 0 {
		t.Fatalf("exact-span eviction freed %d positions, want > 0", evict.freed)
	}
	if !evict.exact {
		t.Fatalf("post-eviction cache is NOT bit-identical to never-saw — the reposition invariant failed on the served path")
	}

	// Turn 4 — the differentiator: after the exact-span eviction the SURVIVING prefix
	// keeps realizing reuse. A whole-cache-reset engine would re-prefill this turn cold.
	postEvict := cacheobs.Default.Snapshot()
	transcript = append(transcript, agent.Message{Role: agent.RoleUser, Content: "summarize what we learned so far"})
	post(transcript)
	final := cacheobs.Default.Snapshot()
	if final.ReusedTokens <= postEvict.ReusedTokens {
		t.Fatalf("no reuse realized AFTER the exact-span eviction (reused %d -> %d) — eviction destroyed the session's cache value",
			postEvict.ReusedTokens, final.ReusedTokens)
	}
	if final.Turns != 4 {
		t.Fatalf("session booked %d turns, want 4", final.Turns)
	}

	// The Track-1 WITNESSED fold — the SAME cachevalueledger.Append serve-exit call
	// cmd/fak/serve.go makes, pointed at a scratch ledger, then scored the way
	// `fak cachevalue report` / the regression gate folds it.
	ledger := t.TempDir() + "/cache-value.jsonl"
	if err := cachevalueledger.Append("serve", "inkernel-exact-span-witness", ledger, final); err != nil {
		t.Fatalf("Append: %v", err)
	}
	rows := cachevalueledger.ReadLedgerFile(ledger)
	if len(rows) != 1 {
		t.Fatalf("ledger rows = %d, want 1", len(rows))
	}
	row := rows[0]
	if row.Provider != "fak" || row.Mechanism != "kv_prefix_reuse" {
		t.Fatalf("ledger row is not fak-authored: provider=%q mechanism=%q (a provider-OBSERVED row must never satisfy #2727)", row.Provider, row.Mechanism)
	}
	if row.ReusedTokens == 0 || row.ReuseRatio <= 0 {
		t.Fatalf("WITNESSED row carries zero cache value: %+v", row)
	}
	score, err := cachevalueledger.ScoreLedger(ledger)
	if err != nil {
		t.Fatalf("ScoreLedger: %v", err)
	}
	if score.MultiTurnSessions != 1 || score.RealizedReuseRatio <= 0 {
		t.Fatalf("report fold shows no WITNESSED value: multi_turn=%d realized_reuse=%.4f", score.MultiTurnSessions, score.RealizedReuseRatio)
	}
	if !score.VsNaiveMultipleExcluded {
		t.Fatalf("#1066 honesty fence missing from the fold: %+v", score)
	}

	// The native serving row: the in-kernel worker renders onto the SAME fak_serving_*
	// schema the vLLM/SGLang scrape emitters use, with the prefix-cache hit rate fed by
	// this session's realized reuse. Buffered (non-streaming) turns observe no first-token
	// boundary, so TTFT is honestly absent; the row fires on goodput + hit rate.
	text := srv.renderMetrics()
	labels := `{worker="local",engine="test",model="m"}`
	for _, want := range []string{
		"fak_serving_goodput_requests_per_second" + labels,
		"fak_serving_prefix_cache_hit_rate" + labels,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("native in-kernel serving row missing %q on /metrics\n--- metrics ---\n%s", want, text)
		}
	}
	hitLine := metricLine(text, "fak_serving_prefix_cache_hit_rate"+labels)
	if hitLine == "" {
		t.Fatalf("no fak_serving_prefix_cache_hit_rate sample line")
	}
	hit, err := strconv.ParseFloat(strings.TrimSpace(strings.TrimPrefix(hitLine, "fak_serving_prefix_cache_hit_rate"+labels)), 64)
	if err != nil {
		t.Fatalf("parse %q: %v", hitLine, err)
	}
	if hit <= 0 {
		t.Fatalf("fak_serving_prefix_cache_hit_rate = %v, want > 0 (the session realized reuse above)", hit)
	}

	// Pin the serve-exit row shape a Mac operator will land: same fold, non-zero value,
	// stamped within this run.
	if row.Date != time.Now().UTC().Format("2006-01-02") {
		t.Logf("note: ledger row date %s (UTC day boundary crossed mid-test)", row.Date)
	}
	t.Logf("WITNESS #2727 (host-independent half): turns=%d reused=%d/%d (ratio %.3f), exact-span evict freed=%d exact=%v, serving hit rate=%.3f",
		final.Turns, final.ReusedTokens, final.PromptTokens, final.ReuseRatio, evict.freed, evict.exact, hit)
}

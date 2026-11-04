package agent

import (
	"context"
	"os"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestPromptShrinkResolvedParityAndOverrides(t *testing.T) {
	messages := []Message{{Role: "user", Content: "one"}, {Role: "assistant", Content: "two"}, {Role: "user", Content: "three"}}
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "example"}}}
	zero, budget, no, yes := 0, 20, false, true
	for _, sp := range []SampleParams{
		{}, {CompactHistoryBudget: &budget}, {ElideStaleReads: &yes}, {DeferColdTools: &yes},
		{CompactHistoryBudget: &zero, ElideStaleReads: &no, DeferColdTools: &no},
	} {
		p := &InKernelPlanner{compactHistoryBudget: 30, elideStaleReads: true, deferColdTools: true}
		gotM, gotT, gotO := p.applyPromptShrinkResolved(context.Background(), messages, tools, sp)
		wantM, wantT, wantO := formerPromptShrinkOracle(p, context.Background(), messages, tools, func(out *SampleParams) { *out = sp })
		if !reflect.DeepEqual(gotM, wantM) || !reflect.DeepEqual(gotT, wantT) || !reflect.DeepEqual(gotO, wantO) {
			t.Fatal("resolved shrink differs from deterministic public wrapper")
		}
	}
	p := &InKernelPlanner{compactHistoryBudget: 30, elideStaleReads: true, deferColdTools: true}
	sp := SampleParams{CompactHistoryBudget: &zero, ElideStaleReads: &no, DeferColdTools: &no}
	gotM, gotT, gotO := p.applyPromptShrinkResolved(context.Background(), messages, tools, sp)
	if !reflect.DeepEqual(gotM, messages) || !reflect.DeepEqual(gotT, tools) || !reflect.DeepEqual(gotO, TypedPromptShrinkOutcome{}) {
		t.Fatal("explicit zero/false must override active planner defaults")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestPromptShrinkResolvedDoesNotReplayOptions(t *testing.T) {
	p := &InKernelPlanner{}
	calls := 0
	opt := func(sp *SampleParams) { calls++; budget := calls; sp.CompactHistoryBudget = &budget }
	sp := applySampleOpts(opt)
	p.applyPromptShrinkResolved(context.Background(), nil, nil, sp)
	if calls != 1 || *sp.CompactHistoryBudget != 1 {
		t.Fatalf("resolved controls changed: calls=%d sp=%+v", calls, sp)
	}
	p.ApplyPromptShrink(context.Background(), nil, nil, opt)
	if calls != 2 {
		t.Fatalf("public wrapper must resolve exactly once per invocation: %d", calls)
	}
}

// This syntactic guard covers the prompt entry path without claiming execution
// of tokenizer/admission fixtures or request-wide option evaluation counts.
// fak-test:runtime fast est=100ms lane=default
func TestPreparePromptResolvedShrinkSourceGuard(t *testing.T) {
	data, err := os.ReadFile("prompt_encoding.go")
	if err != nil {
		t.Fatal(err)
	}
	src := string(data)
	start := strings.Index(src, "func (p *InKernelPlanner) preparePrompt(")
	if start < 0 {
		t.Fatal("preparePrompt missing")
	}
	end := strings.Index(src[start:], "messages, tools, shrink := p.applyPromptShrinkResolved(ctx, messages, tools, sp)")
	if start < 0 || end < 0 {
		t.Fatal("resolved shrink call missing")
	}
	pre := src[start : start+end]
	for _, required := range []string{
		"if err := ctx.Err(); err != nil",
		"p.admitDeepSeekV41ChatProfile(messages, sp)",
		"p.admitNativeChatTemplate(messages, tools, sp)",
		"if p.tok != nil && p.tok.IsSPM()",
		"return p.prepareSPMRawPrompt(ctx, messages, tools, sp)",
	} {
		if !strings.Contains(pre, required) {
			t.Fatalf("early admission/SPM boundary missing: %s", required)
		}
	}
	tail := src[start:]
	if strings.Contains(tail, "applySampleOpts(") || strings.Contains(tail, "p.ApplyPromptShrink(") {
		t.Fatal("preparePrompt must not refold options for shrink")
	}
}

// Literal former public body at 75f9aed7, independent of the new helper.
func formerPromptShrinkOracle(p *InKernelPlanner, ctx context.Context, messages []Message, tools []ToolDef, opts ...SampleOpt) ([]Message, []ToolDef, TypedPromptShrinkOutcome) {
	sp := applySampleOpts(opts...)
	compactBudget := p.compactHistoryBudget
	if sp.CompactHistoryBudget != nil {
		compactBudget = *sp.CompactHistoryBudget
	}
	elideStale := p.elideStaleReads
	if sp.ElideStaleReads != nil {
		elideStale = *sp.ElideStaleReads
	}
	deferTools := p.deferColdTools
	if sp.DeferColdTools != nil {
		deferTools = *sp.DeferColdTools
	}

	if compactBudget <= 0 && !elideStale && !deferTools {
		return messages, tools, TypedPromptShrinkOutcome{}
	}

	var stash func(id, excerpt string, body []byte)
	if p.restoreStash != nil {
		traceID := nativePhaseTraceID(ctx)
		stash = func(id, excerpt string, body []byte) {
			p.restoreStash(traceID, id, excerpt, body)
		}
	}

	return ApplyTypedPromptShrinkLevers(messages, tools, TypedPromptShrinkConfig{
		CompactHistoryBudget:    compactBudget,
		ElideStaleReads:         elideStale,
		DeferColdTools:          deferTools,
		RestoreStash:            stash,
		observeNativeCompaction: nativeCompactionObserver(ctx) != nil,
	})
}

// fak-test:runtime fast est=500ms lane=default
func TestPreparePromptUsesResolvedShrinkWithoutReplay(t *testing.T) {
	p := reusePlanner(false, false, tinyCfg())
	p.tok = loadProbeTok(t)
	calls := 0
	opt := func(sp *SampleParams) {
		calls++
		if calls > 1 {
			panic("preparePrompt replayed option")
		}
		zero := 0
		sp.CompactHistoryBudget = &zero
	}
	sp := applySampleOpts(opt)
	prepared, err := p.preparePrompt(context.Background(), []Message{{Role: "user", Content: "hello"}}, nil, sp, opt)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 || len(prepared.ids) == 0 {
		t.Fatalf("calls=%d ids=%v", calls, prepared.ids)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err = p.preparePrompt(ctx, nil, nil, sp, opt); err != context.Canceled {
		t.Fatalf("cancellation=%v", err)
	}
	if calls != 1 {
		t.Fatalf("cancelled preparation replayed option: %d", calls)
	}
}

package gateway

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

type perfNativeTimingPlanner struct {
	mu   sync.Mutex
	next int
}

type perfOwnedLoopCachePlanner struct {
	usage agent.Usage
}

func (*perfOwnedLoopCachePlanner) Model() string { return "owned-loop-cache" }

func (p *perfOwnedLoopCachePlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	return p.completion(nil)
}

func (*perfOwnedLoopCachePlanner) StreamingSupported() bool { return true }

func (p *perfOwnedLoopCachePlanner) CompleteStream(_ context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	return p.completion(sink)
}

func (p *perfOwnedLoopCachePlanner) completion(sink agent.StreamSink) (*agent.Completion, error) {
	if sink != nil {
		if err := sink("done"); err != nil {
			return nil, err
		}
	}
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "done"},
		FinishReason: "stop",
		Model:        "owned-loop-cache",
		Usage:        p.usage,
	}, nil
}

func (*perfNativeTimingPlanner) Model() string { return "native-timing-model" }

func (p *perfNativeTimingPlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	return p.completion(nil)
}

func (*perfNativeTimingPlanner) StreamingSupported() bool { return true }

func (p *perfNativeTimingPlanner) CompleteStream(_ context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	return p.completion(sink)
}

func (p *perfNativeTimingPlanner) completion(sink agent.StreamSink) (*agent.Completion, error) {
	p.mu.Lock()
	i := p.next
	p.next++
	p.mu.Unlock()
	cases := []struct {
		prompt, cached, predicted int
		prefillMS, decodeMS       float64
	}{
		{prompt: 120, cached: 40, predicted: 4, prefillMS: 80, decodeMS: 4},
		{prompt: 240, cached: 60, predicted: 6, prefillMS: 90, decodeMS: 3},
		{prompt: 360, cached: 80, predicted: 8, prefillMS: 140, decodeMS: 2},
	}
	if i >= len(cases) {
		i = len(cases) - 1
	}
	c := cases[i]
	if sink != nil {
		if err := sink("done"); err != nil {
			return nil, err
		}
	}
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "done"},
		FinishReason: "stop",
		Model:        "native-timing-model",
		Usage: agent.Usage{
			PromptTokens:        c.prompt,
			CompletionTokens:    c.predicted,
			TotalTokens:         c.prompt + c.predicted,
			PromptTokensDetails: &agent.UsageTokenDetails{CachedTokens: c.cached},
		},
		Timings: agent.NewTimings(c.prompt, c.cached, c.predicted, c.prefillMS/1000, c.decodeMS/1000),
		NativeDecode: &agent.NativeDecodeSummary{
			Path: agent.NativeDecodePathSerial,
		},
	}, nil
}

// fak-test:runtime fast est=3s lane=default
func TestPerfLedgerNativeTimingSurvivesChatAndOwnedLoopHTTP(t *testing.T) {
	planner := &perfNativeTimingPlanner{}
	h := newPerfServedHarness(t, planner)
	h.srv.native = true
	h.srv.nativeMaxTurns = 2

	h.chat("native-timing-model", "ordinary chat", false)
	for _, stream := range []bool{false, true} {
		status, body := postNativeMessages(t, h.ts, map[string]any{
			"model":      "native-timing-model",
			"max_tokens": 16,
			"stream":     stream,
			"messages":   []map[string]any{{"role": "user", "content": "owned loop"}},
		})
		if status != http.StatusOK {
			t.Fatalf("native stream=%t status=%d body=%s", stream, status, body)
		}
	}

	rep := h.recent()
	if rep.Source != "live" || len(rep.Records) != 3 {
		t.Fatalf("live report source=%q rows=%d, want live/3: %+v", rep.Source, len(rep.Records), rep.Records)
	}
	wantPrefillMS := []float64{80, 90, 140}
	wantDecodeMS := []float64{4, 3, 2}
	wantCached := []int{40, 60, 80}
	for i, r := range rep.Records {
		if r.NativeTiming == nil {
			t.Fatalf("row %d has no native timing: %+v", i, r)
		}
		if r.NativeTiming.PrefillMS != wantPrefillMS[i] || r.NativeTiming.DecodeMS != wantDecodeMS[i] {
			t.Errorf("row %d phases = %+v, want prefill %.0fms decode %.0fms", i, r.NativeTiming, wantPrefillMS[i], wantDecodeMS[i])
		}
		if r.CachedTokens != wantCached[i] {
			t.Errorf("row %d cached_tokens=%d, want %d", i, r.CachedTokens, wantCached[i])
		}
	}
	// The streamed owned-loop row has a client-observed TTFT, but its full loop
	// wall time is not an engine prefill/decode estimate.
	if rep.Records[2].TTFTMS <= 0 || rep.Records[2].PrefillTPS != 0 || rep.Records[2].DecodeTPS != 0 {
		t.Errorf("streamed owned-loop client axes = %+v, want ttft measured and wall-derived rates absent", rep.Records[2])
	}

	n := rep.Summary.NativeTiming
	if n == nil || n.Count != 3 || n.PrefillMeasured != 3 || n.DecodeMeasured != 3 {
		t.Fatalf("live native timing coverage = %+v, want 3/3 prefill and decode", n)
	}
	if n.PrefillP50MS != 90 || n.DecodeP50MS != 3 || n.PrefillTPSP50 != 2000 || n.DecodeTPSP50 != 2000 {
		t.Errorf("live native timing medians = %+v, want 90ms/3ms and 2000/2000 tok/s", n)
	}
	if n.PrefillTPSWeighted != 1741.94 || n.DecodeTPSWeighted != 2000 {
		t.Errorf("live native timing weighted rates = %+v, want 1741.94/2000", n)
	}
	if rep.Summary.CacheHitShare != 0.25 {
		t.Errorf("cache hit share=%.4f, want 0.25", rep.Summary.CacheHitShare)
	}
	if line := h.compact(); !strings.Contains(line, "native phases=3") || !strings.Contains(line, "measured 3/3") {
		t.Fatalf("compact report does not expose native coverage: %q", line)
	}

	recs := h.durable()
	if len(recs) != 3 {
		t.Fatalf("durable rows=%d, want 3: %+v", len(recs), recs)
	}
	offline := perfledger.BuildReport(recs, 2, false, 0)
	if len(offline.Records) != 2 || offline.Summary.NativeTiming == nil || offline.Summary.NativeTiming.Count != 2 {
		t.Fatalf("bounded offline report = %+v, want newest 2 native rows", offline)
	}
	if offline.Summary.NativeTiming.PrefillP50MS != 90 || offline.Summary.NativeTiming.DecodeP50MS != 2 {
		t.Fatalf("offline native timing = %+v, want bounded p50 90ms/2ms", offline.Summary.NativeTiming)
	}
}

// fak-test:runtime fast est=5s lane=default
func TestPerfLedgerNativeTimingFromActualCPUReferencePath(t *testing.T) {
	t.Setenv("FAK_INKERNEL_MAX_TOKENS", "64")
	m := model.NewSynthetic(kvmmuSynthCfg())
	m.Quantize()
	planner := agent.NewInKernelPlanner(m, newByteLevelTokenizer(t), "native-synth", false, nil, false)
	h := newPerfServedHarness(t, planner)
	h.srv.native = true
	h.srv.nativeMaxTurns = 1

	content := strings.Repeat("actual native phase witness ", 64)
	chatBody := fmt.Sprintf(`{"model":"native-synth","stream":false,"max_tokens":64,"messages":[{"role":"user","content":%q}]}`, content)
	resp, err := http.Post(h.ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(chatBody))
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("actual native chat status=%d body=%s", resp.StatusCode, raw)
	}
	status, nativeBody := postNativeMessages(t, h.ts, map[string]any{
		"model":      "native-synth",
		"max_tokens": 64,
		"messages":   []map[string]any{{"role": "user", "content": content}},
	})
	if status != http.StatusOK {
		t.Fatalf("actual native owned loop status=%d body=%s", status, nativeBody)
	}

	rep := h.recent()
	if len(rep.Records) != 2 || rep.Summary.NativeTiming == nil || rep.Summary.NativeTiming.Count != 2 ||
		rep.Summary.NativeTiming.PrefillMeasured != 2 || rep.Summary.NativeTiming.DecodeMeasured != 2 {
		t.Fatalf("actual native live report = %+v, want two measured native rows", rep)
	}
	for i, r := range rep.Records {
		if r.NativeTiming == nil || r.NativeTiming.PrefillMS <= 0 || r.NativeTiming.DecodeMS <= 0 ||
			r.NativeTiming.PrefillTokens <= 0 || r.NativeTiming.DecodeTokens <= 0 {
			t.Fatalf("actual native row %d has no positive measured phases: timing=%+v row=%+v", i, r.NativeTiming, r)
		}
		t.Logf("actual CPU row %d native phases: prefill=%gms/%d tokens decode=%gms/%d tokens", i,
			r.NativeTiming.PrefillMS, r.NativeTiming.PrefillTokens, r.NativeTiming.DecodeMS, r.NativeTiming.DecodeTokens)
	}
	if line := h.compact(); !strings.Contains(line, "native phases=2") || !strings.Contains(line, "measured 2/2") {
		t.Fatalf("actual native compact report does not expose phase coverage: %q", line)
	}
	recs := h.durable()
	if len(recs) != 2 || recs[0].NativeTiming == nil || recs[1].NativeTiming == nil {
		t.Fatalf("actual native durable rows lost measured phases: %+v", recs)
	}
}

// fak-test:runtime fast est=2s lane=default
func TestPerfLedgerNativeTimingOwnedLoopCacheShapes(t *testing.T) {
	cases := []struct {
		name   string
		stream bool
		usage  agent.Usage
	}{
		{
			name: "OpenAI total includes cached buffered",
			usage: agent.Usage{
				PromptTokens:        80,
				CompletionTokens:    4,
				PromptTokensDetails: &agent.UsageTokenDetails{CachedTokens: 40},
			},
		},
		{
			name:   "Anthropic input already uncached streamed",
			stream: true,
			usage: agent.Usage{
				PromptTokens:         40,
				CompletionTokens:     4,
				CacheReadInputTokens: 40,
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := newPerfServedHarness(t, &perfOwnedLoopCachePlanner{usage: tc.usage})
			h.srv.native = true
			h.srv.nativeMaxTurns = 1
			status, body := postNativeMessages(t, h.ts, map[string]any{
				"model":      "owned-loop-cache",
				"max_tokens": 4,
				"stream":     tc.stream,
				"messages":   []map[string]any{{"role": "user", "content": "cache accounting"}},
			})
			if status != http.StatusOK {
				t.Fatalf("native stream=%t status=%d body=%s", tc.stream, status, body)
			}
			rep := h.recent()
			if len(rep.Records) != 1 {
				t.Fatalf("rows=%d, want 1: %+v", len(rep.Records), rep.Records)
			}
			r := rep.Records[0]
			if r.PromptTokens != 40 || r.CachedTokens != 40 {
				t.Fatalf("owned-loop cache row = prompt=%d cached=%d, want 40/40", r.PromptTokens, r.CachedTokens)
			}
			recs := h.durable()
			if len(recs) != 1 || recs[0].PromptTokens != 40 || recs[0].CachedTokens != 40 {
				t.Fatalf("durable owned-loop cache rows = %+v, want one 40/40 row", recs)
			}
		})
	}
}

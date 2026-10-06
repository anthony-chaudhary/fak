package gateway

// stream_itl_test.go — fak_serving_inter_token_latency_seconds must observe REAL
// gaps between consecutive output-token emissions (vLLM inter_token_latency_seconds
// semantics), not alias the per-turn TPOT mean. A 500ms mid-decode stall among 10ms
// gaps must land in the tail buckets and in the per-request max-ITL histogram.

import (
	"bytes"
	"context"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestStreamITLObservesRealGapsNotAverage(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	var tr streamITL
	base := time.Unix(1_700_000_000, 0)
	// Five deltas: gaps 10ms, 10ms, 10ms, then a 500ms stall (e.g. a peer prefill).
	for _, off := range []time.Duration{0, 10, 20, 30, 530} {
		tr.mark(m, base.Add(off*time.Millisecond))
	}
	tr.finish(m)

	itl, maxITL := m.itlSnapshot()
	if itl.count != 4 {
		t.Fatalf("ITL samples = %d, want 4 (one per gap, first delta is TTFT)", itl.count)
	}
	if math.Abs(itl.sum-0.53) > 1e-9 {
		t.Fatalf("ITL sum = %v, want 0.53", itl.sum)
	}
	bucket := func(s latencySnapshot, le float64) uint64 {
		for i, b := range gatewayLatencyBuckets {
			if b == le {
				return s.buckets[i]
			}
		}
		t.Fatalf("no bucket le=%v", le)
		return 0
	}
	// The per-turn mean (0.1325s) would put the single sample above 0.1; real gaps put
	// three samples at <=0.01 and exactly one in the (0.25, 0.5] tail.
	if got := bucket(itl, 0.01); got != 3 {
		t.Fatalf("ITL le=0.01 = %d, want 3", got)
	}
	if got := bucket(itl, 0.25); got != 3 {
		t.Fatalf("ITL le=0.25 = %d, want 3 (the stall must sit in the tail)", got)
	}
	if got := bucket(itl, 0.5); got != 4 {
		t.Fatalf("ITL le=0.5 = %d, want 4", got)
	}
	if maxITL.count != 1 || math.Abs(maxITL.sum-0.5) > 1e-9 {
		t.Fatalf("max ITL = count %d sum %v, want 1 sample of 0.5s", maxITL.count, maxITL.sum)
	}
	if tr != (streamITL{}) {
		t.Fatalf("finish must reset the tracker, got %+v", tr)
	}
	// A request with a single delta has no gap: no ITL and no max sample.
	tr.mark(m, base)
	tr.finish(m)
	if itl2, max2 := m.itlSnapshot(); itl2.count != 4 || max2.count != 1 {
		t.Fatalf("single-delta request added samples: itl=%d max=%d", itl2.count, max2.count)
	}
}

// TestServingITLNotAliasedFromTPOT: buffered/measured turns fill TPOT, but with no
// live stream gaps the ITL family stays absent instead of copying the TPOT mean.
func TestServingITLNotAliasedFromTPOT(t *testing.T) {
	srv := newTestServer(t)
	srv.metrics.observeInferenceTimed(100, 50, 0, 0, "stop", 2*time.Second, 500*time.Millisecond)
	out := srv.renderMetrics()
	if !strings.Contains(out, "fak_serving_time_per_output_token_seconds_count{") {
		t.Fatalf("TPOT family missing\n%s", out)
	}
	if strings.Contains(out, "fak_serving_inter_token_latency_seconds_count{") {
		t.Fatalf("ITL family published without any real inter-token gap (aliased from TPOT)")
	}
}

// gappedTokenPlanner streams tokens with a controlled sleep BEFORE each token.
type gappedTokenPlanner struct {
	tokens []string
	gaps   []time.Duration
}

func (p *gappedTokenPlanner) Model() string { return "test-model" }
func (p *gappedTokenPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return nil, agent.ErrStreamingUnsupported
}
func (p *gappedTokenPlanner) StreamingSupported() bool { return true }
func (p *gappedTokenPlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	var full strings.Builder
	for i, tok := range p.tokens {
		if d := p.gaps[i]; d > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(d):
			}
		}
		if err := sink(tok); err != nil {
			return nil, err
		}
		full.WriteString(tok)
	}
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: full.String()},
		FinishReason: "stop",
		Model:        "test-model",
		Usage:        agent.Usage{PromptTokens: 3, CompletionTokens: len(p.tokens)},
	}, nil
}

var _ agent.StreamingPlanner = (*gappedTokenPlanner)(nil)

// TestLiveStreamITLRecordsStallGap drives the real /v1/chat/completions live SSE path
// and asserts the serving ITL family counts each gap and the max-ITL family carries
// the injected stall.
func TestLiveStreamITLRecordsStallGap(t *testing.T) {
	const stall = 200 * time.Millisecond
	srv := newTestServer(t)
	srv.planner = &gappedTokenPlanner{
		tokens: []string{"alpha", " beta", " gamma", " delta"},
		gaps:   []time.Duration{0, 0, 0, stall},
	}
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)

	body := []byte(`{"model":"test-model","messages":[{"role":"user","content":"probe"}],"stream":true}`)
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK || !strings.Contains(string(raw), "[DONE]") {
		t.Fatalf("stream status=%d body=%s", resp.StatusCode, raw)
	}

	// The handler folds max-ITL on return, which can trail the client's read of [DONE].
	deadline := time.Now().Add(5 * time.Second)
	var itl, maxITL latencySnapshot
	for {
		itl, maxITL = srv.metrics.itlSnapshot()
		if maxITL.count == 1 || time.Now().After(deadline) {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if itl.count != 3 {
		t.Fatalf("ITL samples = %d, want 3 (4 content deltas)", itl.count)
	}
	if maxITL.count != 1 || maxITL.sum < stall.Seconds() {
		t.Fatalf("max ITL = count %d sum %v, want 1 sample >= %v", maxITL.count, maxITL.sum, stall)
	}
	if itl.sum < stall.Seconds() {
		t.Fatalf("ITL sum %v below the injected stall %v", itl.sum, stall)
	}

	out := srv.renderMetrics()
	for fam, want := range map[string]float64{
		"fak_serving_inter_token_latency_seconds_count":     3,
		"fak_serving_max_inter_token_latency_seconds_count": 1,
	} {
		re := regexp.MustCompile(regexp.QuoteMeta(fam) + `\{[^}]*\} (\S+)`)
		mm := re.FindStringSubmatch(out)
		if mm == nil {
			t.Fatalf("%s missing from /metrics", fam)
		}
		if got, _ := strconv.ParseFloat(mm[1], 64); got != want {
			t.Fatalf("%s = %v, want %v", fam, got, want)
		}
	}
}

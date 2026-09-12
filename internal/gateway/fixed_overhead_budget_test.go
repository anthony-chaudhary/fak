package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/sessionledger"
)

const (
	fixedOverheadSamples = 25
	fixedOverheadBudget  = 50 * time.Millisecond
	fixedOverheadChild   = "FAK_FIXED_OVERHEAD_CHILD"
)

type fixedOverheadPlanner struct {
	calls atomic.Int64
	nanos atomic.Int64
}

func (p *fixedOverheadPlanner) Model() string { return "fixed-overhead-probe" }

func (p *fixedOverheadPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (completion *agent.Completion, err error) {
	start := time.Now()
	defer func() { p.nanos.Add(time.Since(start).Nanoseconds()) }()
	p.calls.Add(1)
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "x"},
		FinishReason: "stop",
		Model:        p.Model(),
		Usage:        agent.Usage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
	}, nil
}

type fixedOverheadStats struct {
	p50 time.Duration
	p95 time.Duration
	max time.Duration
}

// TestFixedOverheadBudget measures the real HTTP handler with an instant planner.
// Its component probes are standalone and overlapping: they locate likely offenders,
// but are deliberately not added together or presented as an exact decomposition.
func TestFixedOverheadBudget(t *testing.T) {
	if os.Getenv(fixedOverheadChild) != "1" {
		ledgerDir := filepath.Join(t.TempDir(), "session-ledger")
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestFixedOverheadBudget$", "-test.v")
		cmd.Env = fixedOverheadSubprocessEnv(ledgerDir)
		out, err := cmd.CombinedOutput()
		t.Logf("isolated measurement subprocess:\n%s", out)
		if err != nil {
			t.Fatalf("isolated measurement subprocess: %v", err)
		}
		return
	}

	planner := &fixedOverheadPlanner{}
	srv, err := New(Config{
		EngineID:   "mock",
		Model:      planner.Model(),
		RequireKey: "budget-key",
		Logf:       func(string, ...any) {},
	})
	if err != nil {
		t.Fatalf("New gateway: %v", err)
	}
	t.Cleanup(srv.Close)
	srv.SetPlanner(planner)
	srv.SetAdmissionController(NewAdmissionController(AdmissionPolicy{
		MaxNumSeqs:      1,
		TokenBudget:     1 << 20,
		MaxWaiting:      1,
		PreallocCeiling: 1,
	}))
	handler := srv.Handler()

	smallBody := fixedOverheadChatBody(t, []agent.Message{{Role: agent.RoleUser, Content: "ping"}})
	longMessages := fixedOverheadLongTranscript()
	longBody := fixedOverheadChatBody(t, longMessages)

	// Warm route maps, JSON encoders, and the ledger file before collecting samples.
	for i := 0; i < 3; i++ {
		fixedOverheadServe(t, handler, planner, "/healthz", nil, false, i)
		fixedOverheadServe(t, handler, planner, "/v1/chat/completions", smallBody, true, i)
		fixedOverheadServe(t, handler, planner, "/v1/chat/completions", longBody, true, i)
	}

	healthTotal := fixedOverheadHealthSamples(t, handler, planner)
	smallTotal, _, smallFixed := fixedOverheadHandlerSamples(t, handler, planner, "/v1/chat/completions", smallBody, true)
	longTotal, _, longFixed := fixedOverheadHandlerSamples(t, handler, planner, "/v1/chat/completions", longBody, true)

	noop := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusNoContent) })
	middlewareStats := fixedOverheadMeasureBatched(t, 1024, func(i, _ int) func() error {
		req := httptest.NewRequest(http.MethodGet, "/probe", nil)
		rec := httptest.NewRecorder()
		return func() error {
			srv.withMetrics(noop).ServeHTTP(rec, req)
			if rec.Code != http.StatusNoContent {
				return fmt.Errorf("middleware status=%d", rec.Code)
			}
			return nil
		}
	})
	authStats := fixedOverheadMeasureBatched(t, 1024, func(i, _ int) func() error {
		req := httptest.NewRequest(http.MethodGet, "/probe", nil)
		req.Header.Set("Authorization", "Bearer budget-key")
		rec := httptest.NewRecorder()
		return func() error {
			srv.withAuth(noop).ServeHTTP(rec, req)
			if rec.Code != http.StatusNoContent {
				return fmt.Errorf("auth status=%d", rec.Code)
			}
			return nil
		}
	})

	ledger, err := sessionledger.Open(t.TempDir())
	if err != nil {
		t.Fatalf("open temp session ledger: %v", err)
	}
	ledgerStats := fixedOverheadMeasureBatched(t, 16, func(i, j int) func() error {
		trace := fmt.Sprintf("fixed-overhead-ledger-%d-%d", i, j)
		return func() error {
			_, err := ledger.Append(trace, "turn_begin", []byte(`{"sample":true}`))
			return err
		}
	})
	transcriptStats := fixedOverheadMeasureBatched(t, 1024, func(i, j int) func() error {
		trace := fmt.Sprintf("fixed-overhead-scan-%d-%d", i, j)
		return func() error {
			admissions, err := srv.admitInboundResults(context.Background(), longMessages, nil, trace)
			if err != nil {
				return err
			}
			if len(admissions) != 0 {
				return fmt.Errorf("tool-free transcript produced %d result admissions", len(admissions))
			}
			return nil
		}
	})
	admissionStats := fixedOverheadMeasureBatched(t, 1024, func(i, j int) func() error {
		turn := servedSessionTurn{traceID: fmt.Sprintf("fixed-overhead-admission-%d-%d", i, j)}
		return func() error {
			lease, err := srv.beginServedAdmission(context.Background(), turn, []agent.Message{{Role: agent.RoleUser, Content: "ping"}}, nil, 1)
			if err != nil {
				return err
			}
			lease.Release()
			return nil
		}
	})

	t.Logf("overlapping end-to-end Handler probes: health=%s small_chat_total=%s small_chat_fixed(total-planner)=%s long_~8k_chat_total=%s long_fixed(total-planner)=%s",
		fixedOverheadFormat(healthTotal), fixedOverheadFormat(smallTotal), fixedOverheadFormat(smallFixed), fixedOverheadFormat(longTotal), fixedOverheadFormat(longFixed))
	t.Logf("standalone non-additive probes (percentiles of per-operation batch averages): metrics_middleware=%s auth=%s real_temp_ledger_append=%s tool-free_~8k_inbound-result_scan=%s scheduler_admission_acquire_release=%s",
		fixedOverheadFormat(middlewareStats), fixedOverheadFormat(authStats), fixedOverheadFormat(ledgerStats), fixedOverheadFormat(transcriptStats), fixedOverheadFormat(admissionStats))
	t.Log("health-vs-chat includes route, decode, session ledger, transcript handling, admission, response encoding, and middleware; it is not a middleware-only delta; JSON request-event construction remains measured while the configured no-op logger excludes only logger formatting and output cost")

	if smallFixed.p95 > fixedOverheadBudget {
		components := map[string]fixedOverheadStats{
			"metrics middleware": middlewareStats,
			"auth":               authStats,
			"ledger append":      ledgerStats,
			"transcript scan":    transcriptStats,
			"admission":          admissionStats,
		}
		largestProbe := "unattributed full-handler work"
		worst := time.Duration(0)
		for name, stats := range components {
			if stats.p95 > worst {
				largestProbe, worst = name, stats.p95
			}
		}
		t.Fatalf("one-token fixed CPU overhead p95=%s exceeds %s budget; largest measured standalone probe=%s p95=%s (probes overlap, do not sum, and do not prove causality)", smallFixed.p95, fixedOverheadBudget, largestProbe, worst)
	}
}

func fixedOverheadSubprocessEnv(ledgerDir string) []string {
	env := make([]string, 0, len(os.Environ())+2)
	for _, entry := range os.Environ() {
		upper := strings.ToUpper(entry)
		if strings.HasPrefix(upper, "FAK_SESSION_LEDGER_DIR=") || strings.HasPrefix(upper, fixedOverheadChild+"=") {
			continue
		}
		env = append(env, entry)
	}
	return append(env, fixedOverheadChild+"=1", "FAK_SESSION_LEDGER_DIR="+ledgerDir)
}

func fixedOverheadLongTranscript() []agent.Message {
	const messages = 256
	const wordsPerMessage = 32
	result := make([]agent.Message, messages)
	content := bytes.Repeat([]byte("token "), wordsPerMessage)
	for i := range result {
		result[i] = agent.Message{Role: agent.RoleUser, Content: string(content)}
	}
	return result
}

func fixedOverheadChatBody(t *testing.T, messages []agent.Message) []byte {
	t.Helper()
	body, err := json.Marshal(ChatRequest{Model: "fixed-overhead-probe", Messages: messages, MaxTokens: 1})
	if err != nil {
		t.Fatalf("marshal chat request: %v", err)
	}
	return body
}

func fixedOverheadHandlerSamples(t *testing.T, handler http.Handler, planner *fixedOverheadPlanner, path string, body []byte, authenticated bool) (fixedOverheadStats, fixedOverheadStats, fixedOverheadStats) {
	t.Helper()
	totals := make([]time.Duration, 0, fixedOverheadSamples)
	plannerTimes := make([]time.Duration, 0, fixedOverheadSamples)
	fixed := make([]time.Duration, 0, fixedOverheadSamples)
	for i := 0; i < fixedOverheadSamples; i++ {
		total, plannerTime := fixedOverheadServe(t, handler, planner, path, body, authenticated, i+100)
		totals = append(totals, total)
		plannerTimes = append(plannerTimes, plannerTime)
		fixed = append(fixed, total-plannerTime)
	}
	return fixedOverheadSummarize(totals), fixedOverheadSummarize(plannerTimes), fixedOverheadSummarize(fixed)
}

func fixedOverheadHealthSamples(t *testing.T, handler http.Handler, planner *fixedOverheadPlanner) fixedOverheadStats {
	t.Helper()
	return fixedOverheadMeasureBatched(t, 256, func(i, j int) func() error {
		req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
		req.Header.Set("X-Trace-Id", fmt.Sprintf("fixed-overhead-health-%d-%d", i, j))
		rec := httptest.NewRecorder()
		beforeCalls := planner.calls.Load()
		return func() error {
			handler.ServeHTTP(rec, req)
			if rec.Code != http.StatusOK {
				return fmt.Errorf("health status=%d", rec.Code)
			}
			if planner.calls.Load() != beforeCalls {
				return fmt.Errorf("health called planner")
			}
			return nil
		}
	})
}

func fixedOverheadServe(t *testing.T, handler http.Handler, planner *fixedOverheadPlanner, path string, body []byte, authenticated bool, sample int) (time.Duration, time.Duration) {
	t.Helper()
	method := http.MethodGet
	var requestBody *bytes.Reader
	if body != nil {
		method = http.MethodPost
		requestBody = bytes.NewReader(body)
	} else {
		requestBody = bytes.NewReader(nil)
	}
	req := httptest.NewRequest(method, path, requestBody)
	req.Header.Set("X-Trace-Id", fmt.Sprintf("fixed-overhead-http-%s-%d", path, sample))
	if authenticated {
		req.Header.Set("Authorization", "Bearer budget-key")
		req.Header.Set("Content-Type", "application/json")
	}
	rec := httptest.NewRecorder()
	beforeCalls := planner.calls.Load()
	beforeNanos := planner.nanos.Load()
	start := time.Now()
	handler.ServeHTTP(rec, req)
	total := time.Since(start)
	plannerTime := time.Duration(planner.nanos.Load() - beforeNanos)
	wantCalls := beforeCalls
	if body != nil {
		wantCalls++
	}
	if rec.Code != http.StatusOK {
		t.Fatalf("%s status=%d body=%s", path, rec.Code, rec.Body.String())
	}
	if planner.calls.Load() != wantCalls {
		t.Fatalf("%s planner calls=%d after %d, want %d", path, planner.calls.Load(), beforeCalls, wantCalls)
	}
	return total, plannerTime
}

func fixedOverheadMeasureBatched(t *testing.T, iterations int, prepare func(int, int) func() error) fixedOverheadStats {
	t.Helper()
	samples := make([]time.Duration, 0, fixedOverheadSamples)
	for i := 0; i < fixedOverheadSamples; i++ {
		runs := make([]func() error, iterations)
		for j := range runs {
			runs[j] = prepare(i, j)
		}
		start := time.Now()
		for j, run := range runs {
			if err := run(); err != nil {
				t.Fatalf("sample %d iteration %d: %v", i, j, err)
			}
		}
		samples = append(samples, time.Since(start)/time.Duration(iterations))
	}
	return fixedOverheadSummarize(samples)
}

func fixedOverheadSummarize(samples []time.Duration) fixedOverheadStats {
	ordered := append([]time.Duration(nil), samples...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i] < ordered[j] })
	return fixedOverheadStats{
		p50: ordered[len(ordered)/2],
		p95: ordered[(len(ordered)*95+99)/100-1],
		max: ordered[len(ordered)-1],
	}
}

func fixedOverheadFormat(stats fixedOverheadStats) string {
	return fmt.Sprintf("p50=%s p95=%s max=%s", stats.p50, stats.p95, stats.max)
}

package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/pkg/deadlineadmit"
)

func nativeDeadlineFixture(t *testing.T) (*Server, *agent.InKernelPlanner) {
	t.Helper()
	t.Setenv("FAK_INKERNEL_RADIX", "on")
	m := model.NewSynthetic(kvmmuSynthCfg())
	m.Quantize()
	p := agent.NewInKernelPlanner(m, newByteLevelTokenizer(t), "synthetic-live", false, nil, false)
	s := newTestServer(t)
	s.planner, s.model = p, p.Model()
	s.metrics.deadlineEst = slowNativeDeadlineEstimator()
	return s, p
}

func slowNativeDeadlineEstimator() *deadlineadmit.Estimator {
	est := deadlineadmit.NewEstimator(deadlineadmit.Config{}, nil)
	// Deliberately slow prefill and fast decode separate cold and warm decisions.
	est.Observe(deadlineadmit.Observation{PromptTokens: 100, Prefill: 100 * time.Second, CompletionTokens: 100, Decode: time.Millisecond})
	return est
}

// fak-test:runtime fast est=200ms lane=default
func TestNativeDeadlineOwnedWarmCreditAndReservationLifetime(t *testing.T) {
	s, p := nativeDeadlineFixture(t)
	messages := []agent.Message{{Role: agent.RoleUser, Content: strings.Repeat("a", 80)}}
	zero := 0.0
	options := []agent.SampleOpt{agent.WithMaxTokens(1), agent.WithClientTemperature(&zero)}
	if _, err := p.Complete(context.Background(), messages, nil, options...); err != nil {
		t.Fatal(err)
	}
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	r.Header.Set(deadlineadmit.HeaderFakDeadlineMs, "10000")
	arrived := time.Now()
	r, cancel := bindClientDeadline(r, arrived)
	defer cancel()
	r, release, ok := s.admitRoutedClientDeadline(httptest.NewRecorder(), r, arrived, messages, 1)
	if !ok {
		t.Fatal("qualified native admission was rejected before acquiring its prefix")
	}
	defer release()
	if n := s.metrics.deadlineEst.InFlight(); n != 0 {
		t.Fatalf("reservation began before execution verdict: %d", n)
	}
	for attempt := 0; attempt < 2; attempt++ {
		comp, err := p.Complete(r.Context(), messages, nil, options...)
		if err != nil || comp == nil || comp.Usage.CachedPromptTokens() == 0 {
			t.Fatalf("warm attempt %d completion=%v error=%v", attempt, comp, err)
		}
		if n := s.metrics.deadlineEst.InFlight(); n != 1 {
			t.Fatalf("attempt %d in-flight=%d, want one reservation", attempt, n)
		}
	}
	cold := []agent.Message{{Role: agent.RoleUser, Content: strings.Repeat("z", 80)}}
	_, err := p.Complete(r.Context(), cold, nil, options...)
	var refusal *deadlineAdmissionError
	if !errors.As(err, &refusal) {
		t.Fatalf("new cold attempt reused warm credit: %v", err)
	}
	if status, code, _ := upstreamErrorStatus(err); status != http.StatusServiceUnavailable || code != deadlineadmit.CodeDeadlineInfeasible {
		t.Fatalf("refusal mapping=%d/%s", status, code)
	}
	release()
	release()
	if n := s.metrics.deadlineEst.InFlight(); n != 0 {
		t.Fatalf("released in-flight=%d", n)
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestNativeDeadlineColdStreamRefusesBeforeSSE(t *testing.T) {
	s, _ := nativeDeadlineFixture(t)
	zero := 0.0
	body, err := json.Marshal(ChatRequest{Model: "synthetic-live", Messages: []agent.Message{{Role: agent.RoleUser, Content: strings.Repeat("a", 80)}}, MaxTokens: 1, Temperature: &zero, Stream: true})
	if err != nil {
		t.Fatal(err)
	}
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set(deadlineadmit.HeaderFakDeadlineMs, "10000")
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, r)
	if w.Code != http.StatusServiceUnavailable || w.Header().Get("Retry-After") == "" || w.Header().Get(deadlineadmit.HeaderReject) != string(deadlineadmit.ReasonDeadlineInfeasible) {
		t.Fatalf("cold stream status=%d headers=%v body=%s", w.Code, w.Header(), w.Body.String())
	}
	if strings.Contains(w.Body.String(), "data:") || strings.Contains(w.Body.String(), "fak-heartbeat") {
		t.Fatalf("SSE committed before refusal: %s", w.Body.String())
	}
	if n := s.metrics.deadlineEst.InFlight(); n != 0 {
		t.Fatalf("refused stream leaked reservation: %d", n)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestNativeDeadlineEarlyBindingAndUnsupportedColdRoute(t *testing.T) {
	s := saturatedDeadlineServer(t)
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(time.Second))
	defer cancel()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil).WithContext(ctx)
	r.Header.Set(deadlineadmit.HeaderFakDeadlineMs, "10000")
	r, cancelBound := bindClientDeadline(r, time.Now())
	defer cancelBound()
	want, _ := ctx.Deadline()
	if got, _ := r.Context().Deadline(); !got.Equal(want) {
		t.Fatalf("early binding extended incoming deadline: %v, want %v", got, want)
	}
	w := httptest.NewRecorder()
	_, release, ok := s.admitRoutedClientDeadline(w, r, time.Now(), []agent.Message{{Role: agent.RoleUser, Content: strings.Repeat("a", 6400)}}, 1)
	if release != nil {
		defer release()
	}
	if ok || w.Code != http.StatusServiceUnavailable {
		t.Fatalf("unsupported route received credit: admitted=%v status=%d", ok, w.Code)
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestNativeDeadlineWarmHTTPUsesActualRouteCache(t *testing.T) {
	s, _ := nativeDeadlineFixture(t)
	body, err := json.Marshal(map[string]any{
		"model": "synthetic-live", "max_tokens": 1, "temperature": 0,
		"messages":   []agent.Message{{Role: agent.RoleUser, Content: strings.Repeat("a", 80)}},
		"logit_bias": map[string]int{"97": 100}, // deterministic ordinary text, not a synthetic tool sentinel
	})
	if err != nil {
		t.Fatal(err)
	}
	post := func(deadline bool) *httptest.ResponseRecorder {
		r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
		r.Header.Set("Content-Type", "application/json")
		if deadline {
			r.Header.Set(deadlineadmit.HeaderFakDeadlineMs, "10000")
		}
		w := httptest.NewRecorder()
		s.Handler().ServeHTTP(w, r)
		return w
	}
	if w := post(false); w.Code != http.StatusOK {
		t.Fatalf("warmup status=%d body=%s", w.Code, w.Body.String())
	}
	// Restore deterministic policy rates after the real warmup observation.
	s.metrics.deadlineEst = slowNativeDeadlineEstimator()
	w := post(true)
	if w.Code != http.StatusOK {
		t.Fatalf("warm route status=%d body=%s", w.Code, w.Body.String())
	}
	var response ChatResponse
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if response.Usage.CachedPromptTokens() == 0 {
		t.Fatal("HTTP positive did not consume an actual cached prefix")
	}
	if n := s.metrics.deadlineEst.InFlight(); n != 0 {
		t.Fatalf("warm HTTP leaked reservation: %d", n)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestNativeDeadlineExpiredCancellationAndWatchdog(t *testing.T) {
	for _, kind := range []string{"client-expired", "client-canceled", "watchdog-expired"} {
		t.Run(kind, func(t *testing.T) {
			s, p := nativeDeadlineFixture(t)
			r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
			r.Header.Set(deadlineadmit.HeaderFakDeadlineMs, "10000")
			arrived := time.Now()
			if kind == "client-expired" {
				arrived = arrived.Add(-time.Minute)
			}
			r, cancel := bindClientDeadline(r, arrived)
			defer cancel()
			r, release, ok := s.admitRoutedClientDeadline(httptest.NewRecorder(), r, arrived, nil, 1)
			if !ok {
				t.Fatal("native route did not install execution check")
			}
			defer release()
			ctx := r.Context()
			if kind == "client-canceled" {
				cancel()
			}
			if kind == "watchdog-expired" {
				var cancelWatchdog context.CancelFunc
				ctx, cancelWatchdog = context.WithDeadline(ctx, time.Now().Add(-time.Second))
				defer cancelWatchdog()
			}
			_, err := p.Complete(ctx, []agent.Message{{Role: agent.RoleUser, Content: "no work"}}, nil, agent.WithMaxTokens(1))
			var refusal *deadlineAdmissionError
			switch kind {
			case "client-expired":
				if !errors.As(err, &refusal) || refusal.verdict.Reason != deadlineadmit.ReasonDeadlineExpired {
					t.Fatalf("expired client error=%v", err)
				}
			case "client-canceled":
				if !errors.Is(err, context.Canceled) || errors.As(err, &refusal) {
					t.Fatalf("cancellation reclassified: %v", err)
				}
			case "watchdog-expired":
				if !errors.Is(err, context.DeadlineExceeded) || errors.As(err, &refusal) {
					t.Fatalf("watchdog reclassified as client deadline: %v", err)
				}
			}
			release()
			if n := s.metrics.deadlineEst.InFlight(); n != 0 {
				t.Fatalf("terminal request leaked reservation: %d", n)
			}
		})
	}
}

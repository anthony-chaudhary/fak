package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=1ms lane=default
func TestUpExitCodesAreSysexits(t *testing.T) {
	if upExitTransient != 75 || upExitStructural != 78 {
		t.Fatalf("exit codes transient=%d structural=%d, want 75/78", upExitTransient, upExitStructural)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestCheckMaxRSSAgainstResident(t *testing.T) {
	const gib = uint64(1 << 30)
	for _, tc := range []struct {
		name                         string
		ceiling, resident, sessionKV uint64
		refuse                       bool
	}{
		{"guard-disabled", 0, 30 * gib, gib, false},
		{"footprint-unknown", 8 * gib, 0, gib, false},
		{"room-for-a-session", 32 * gib, 20 * gib, gib, false},
		{"one-byte-short-of-ceiling", 21*gib + 1, 20 * gib, gib, false},
		{"session-exactly-fills", 21 * gib, 20 * gib, gib, true},
		{"session-overflows", 21 * gib, 20 * gib, 2 * gib, true},
		{"resident-alone-over", 16 * gib, 20 * gib, 0, true},
		{"resident-equals-ceiling", 20 * gib, 20 * gib, 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := checkMaxRSSAgainstResident(tc.ceiling, tc.resident, tc.sessionKV)
			if !tc.refuse {
				if err != nil {
					t.Fatalf("check = %v, want nil", err)
				}
				return
			}
			if !errors.Is(err, ErrMaxRSSBelowResident) {
				t.Fatalf("check = %v, want errors.Is ErrMaxRSSBelowResident", err)
			}
			wrapped := fmt.Errorf("boot: %w", err)
			var typed *MaxRSSBelowResidentError
			if !errors.As(wrapped, &typed) {
				t.Fatalf("errors.As(*MaxRSSBelowResidentError) failed on %v", wrapped)
			}
			if typed.Ceiling != tc.ceiling || typed.ResidentBytes != tc.resident || typed.SessionKV != tc.sessionKV {
				t.Fatalf("typed = %+v, want ceiling=%d resident=%d kv=%d", *typed, tc.ceiling, tc.resident, tc.sessionKV)
			}
			if !errors.Is(wrapped, ErrMaxRSSBelowResident) {
				t.Fatal("wrapped refusal lost the sentinel")
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestMaxRSSBelowResidentErrorIsOnlyItsSentinel(t *testing.T) {
	e := &MaxRSSBelowResidentError{Ceiling: 1, ResidentBytes: 2}
	if errors.Is(e, context.Canceled) {
		t.Fatal("MaxRSSBelowResidentError matched an unrelated sentinel")
	}
	if errors.Is(errors.New("other"), ErrMaxRSSBelowResident) {
		t.Fatal("unrelated error matched ErrMaxRSSBelowResident")
	}
}

// Compile-time pin, spelled independently of the server seam: the production planner
// reports host-budget stats, so readiness and /healthz see the real budget.
var _ interface {
	HostMemoryBudgetStats() agent.HostMemoryBudgetStats
} = (*agent.InKernelPlanner)(nil)

// hostBudgetStatsPlanner is an agent.Planner that also reports a scripted host budget.
type hostBudgetStatsPlanner struct {
	stats agent.HostMemoryBudgetStats
}

func (p *hostBudgetStatsPlanner) Model() string { return "host-budget-stats-fake" }

func (p *hostBudgetStatsPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return nil, errors.New("hostBudgetStatsPlanner: Complete not expected")
}

func (p *hostBudgetStatsPlanner) HostMemoryBudgetStats() agent.HostMemoryBudgetStats {
	return p.stats
}

func readyTurnkeyServer(planner agent.Planner) *turnkeyServer {
	s := &turnkeyServer{ready: &readinessGate{}, planner: planner}
	s.ready.markReady()
	return s
}

func structuralBudget() agent.HostMemoryBudgetStats {
	return agent.HostMemoryBudgetStats{
		Armed: true, UsedKnown: true, Ceiling: 16 << 30, Used: 19 << 30,
		AvailSigned: -(3 << 30), Structural: true,
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestTurnkeyReadinessAdmissionStarved(t *testing.T) {
	if readinessAdmissionStarved != "admission_starved" {
		t.Fatalf("readinessAdmissionStarved = %q, want admission_starved", readinessAdmissionStarved)
	}
	for _, tc := range []struct {
		name      string
		planner   agent.Planner
		wantReady bool
	}{
		{"structural", &hostBudgetStatsPlanner{stats: structuralBudget()}, false},
		{"unarmed", &hostBudgetStatsPlanner{}, true},
		{"usage-unknown", &hostBudgetStatsPlanner{stats: agent.HostMemoryBudgetStats{Armed: true, Ceiling: 1 << 30, Structural: true}}, true},
		{"headroom", &hostBudgetStatsPlanner{stats: agent.HostMemoryBudgetStats{Armed: true, UsedKnown: true, Ceiling: 16 << 30, Used: 8 << 30, AvailSigned: 8 << 30}}, true},
		{"planner-without-seam", &hostBudgetPlainPlanner{}, true},
		{"nil-planner", nil, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := readyTurnkeyServer(tc.planner)
			ready, state, reason := s.readiness()
			if ready != tc.wantReady {
				t.Fatalf("readiness ready=%v state=%q reason=%q, want ready=%v", ready, state, reason, tc.wantReady)
			}
			if !tc.wantReady && (state != readinessAdmissionStarved || reason != readinessAdmissionStarved) {
				t.Fatalf("starved readiness state=%q reason=%q, want %q", state, reason, readinessAdmissionStarved)
			}
			if tc.wantReady && state != "ok" {
				t.Fatalf("ready state = %q, want ok", state)
			}
		})
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestTurnkeyReadyzAdmissionStarvedIs503(t *testing.T) {
	s := readyTurnkeyServer(&hostBudgetStatsPlanner{stats: structuralBudget()})
	rec := httptest.NewRecorder()
	s.handleReadyz(rec, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("/readyz code = %d, want 503", rec.Code)
	}
	var body struct {
		Status string `json:"status"`
		Ready  bool   `json:"ready"`
		Reason string `json:"reason"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.Ready || body.Status != readinessAdmissionStarved || body.Reason != readinessAdmissionStarved {
		t.Fatalf("/readyz body = %+v, want ready=false status/reason %q", body, readinessAdmissionStarved)
	}
}

func decodeHealthz(t *testing.T, s *turnkeyServer) map[string]json.RawMessage {
	t.Helper()
	rec := httptest.NewRecorder()
	s.handleHealthz(rec, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("/healthz code = %d, want 200", rec.Code)
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	return body
}

// fak-test:runtime fast est=5ms lane=default
func TestTurnkeyHealthzHostBudgetBlock(t *testing.T) {
	existing := []string{"ok", "status", "ready", "mode", "engine", "tier", "model", "headroom_ratio", "native_startup", "live_residency", "agent_warm", "sessions"}

	t.Run("absent-when-unarmed", func(t *testing.T) {
		for _, p := range []agent.Planner{&hostBudgetStatsPlanner{}, &hostBudgetPlainPlanner{}} {
			body := decodeHealthz(t, readyTurnkeyServer(p))
			if _, ok := body["host_budget"]; ok {
				t.Fatalf("/healthz carries host_budget with no armed budget (%T)", p)
			}
			for _, k := range existing {
				if _, ok := body[k]; !ok {
					t.Fatalf("/healthz lost existing key %q", k)
				}
			}
		}
	})

	t.Run("structural-before-first-decline", func(t *testing.T) {
		st := structuralBudget()
		body := decodeHealthz(t, readyTurnkeyServer(&hostBudgetStatsPlanner{stats: st}))
		for _, k := range existing {
			if _, ok := body[k]; !ok {
				t.Fatalf("/healthz lost existing key %q", k)
			}
		}
		var ready bool
		var status string
		_ = json.Unmarshal(body["ready"], &ready)
		_ = json.Unmarshal(body["status"], &status)
		if ready || status != readinessAdmissionStarved {
			t.Fatalf("/healthz ready=%v status=%q, want false/%q", ready, status, readinessAdmissionStarved)
		}
		raw, ok := body["host_budget"]
		if !ok {
			t.Fatal("/healthz has no host_budget with an armed budget")
		}
		var hb map[string]json.RawMessage
		if err := json.Unmarshal(raw, &hb); err != nil {
			t.Fatal(err)
		}
		for _, k := range []string{"ceiling", "used", "avail_signed", "declined_total", "last_decline_at", "structural"} {
			if _, ok := hb[k]; !ok {
				t.Fatalf("host_budget missing key %q: %s", k, raw)
			}
		}
		if string(hb["last_decline_at"]) != "null" {
			t.Fatalf("last_decline_at = %s before any decline, want null", hb["last_decline_at"])
		}
		var got struct {
			Ceiling       int64 `json:"ceiling"`
			Used          int64 `json:"used"`
			AvailSigned   int64 `json:"avail_signed"`
			DeclinedTotal int64 `json:"declined_total"`
			Structural    bool  `json:"structural"`
		}
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Fatal(err)
		}
		if got.Ceiling != st.Ceiling || got.Used != st.Used || got.AvailSigned != st.AvailSigned || got.DeclinedTotal != 0 || !got.Structural {
			t.Fatalf("host_budget = %+v, want ceiling=%d used=%d avail_signed=%d declined=0 structural=true", got, st.Ceiling, st.Used, st.AvailSigned)
		}
	})

	t.Run("after-declines", func(t *testing.T) {
		st := structuralBudget()
		st.DeclinedTotal = 42
		st.LastDeclineAt = time.Date(2026, 10, 6, 12, 0, 0, 0, time.UTC)
		body := decodeHealthz(t, readyTurnkeyServer(&hostBudgetStatsPlanner{stats: st}))
		var got struct {
			DeclinedTotal int64      `json:"declined_total"`
			LastDeclineAt *time.Time `json:"last_decline_at"`
		}
		if err := json.Unmarshal(body["host_budget"], &got); err != nil {
			t.Fatal(err)
		}
		if got.DeclinedTotal != 42 || got.LastDeclineAt == nil || !got.LastDeclineAt.Equal(st.LastDeclineAt) {
			t.Fatalf("host_budget declines = %+v, want 42 at %v", got, st.LastDeclineAt)
		}
	})
}

func hostCapacityError(structural bool) *agent.InKernelCapacityError {
	e := &agent.InKernelCapacityError{
		Want: 2 << 30, Class: compute.MemoryKVCache, Scope: compute.MemoryScopeHost,
		Site: "host-memory-precheck", Structural: structural,
	}
	if structural {
		e.AvailSigned = -(3 << 30)
	} else {
		e.Avail = 1 << 30
		e.AvailSigned = 1 << 30
	}
	return e
}

// fak-test:runtime fast est=5ms lane=default
func TestWriteTurnkeyInferenceErrorStructuralHasNoRetryAfter(t *testing.T) {
	for _, tc := range []struct {
		name      string
		err       error
		wantRetry string
	}{
		{"structural", hostCapacityError(true), ""},
		{"wrapped-structural", fmt.Errorf("turn: %w", hostCapacityError(true)), ""},
		{"transient", hostCapacityError(false), turnkeyCapacityRetryAfterSeconds},
		{"oom", &agent.InKernelOOMError{Bytes: 1 << 30, Class: compute.MemoryKVCache, Site: "kv-grow"}, turnkeyCapacityRetryAfterSeconds},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			writeTurnkeyInferenceError(rec, tc.err)
			res := rec.Result()
			defer res.Body.Close()
			raw, _ := io.ReadAll(res.Body)
			if res.StatusCode != http.StatusServiceUnavailable {
				t.Fatalf("status = %d, want 503; body=%s", res.StatusCode, raw)
			}
			if got := res.Header.Get("Retry-After"); got != tc.wantRetry {
				t.Fatalf("Retry-After = %q, want %q", got, tc.wantRetry)
			}
			env := decodeTurnkeyInferenceErrorEnvelope(t, raw)
			if env.Error.Code != "in_kernel_oom" || env.Error.Type != "server_error" {
				t.Fatalf("envelope code/type = %q/%q, want in_kernel_oom/server_error", env.Error.Code, env.Error.Type)
			}
		})
	}
	if turnkeyCapacityRetryAfterSeconds != "5" {
		t.Fatalf("turnkeyCapacityRetryAfterSeconds = %q, want 5", turnkeyCapacityRetryAfterSeconds)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestTurnkeyWriteInferenceErrorCountsShed(t *testing.T) {
	s := &turnkeyServer{}
	for _, err := range []error{
		hostCapacityError(true),
		hostCapacityError(false),
		&agent.InKernelOOMError{Bytes: 1, Class: compute.MemoryKVCache, Site: "kv-grow"},
	} {
		rec := httptest.NewRecorder()
		s.writeInferenceError(rec, err)
		if rec.Code != http.StatusServiceUnavailable {
			t.Fatalf("writeInferenceError(%T) code = %d, want 503", err, rec.Code)
		}
	}
	if got := s.capacityStats().ShedTotal; got != 3 {
		t.Fatalf("shed_total after three capacity 503s = %d, want 3", got)
	}
	for _, err := range []error{
		&agent.InKernelContextLengthError{PromptTokens: 4000, MaxNewTokens: 200, MaxContext: 4096},
		errors.New("tokenizer exploded"),
	} {
		s.writeInferenceError(httptest.NewRecorder(), err)
	}
	if got := s.capacityStats().ShedTotal; got != 3 {
		t.Fatalf("shed_total after non-capacity errors = %d, want unchanged 3", got)
	}
	var nilServer *turnkeyServer
	rec := httptest.NewRecorder()
	nilServer.writeInferenceError(rec, hostCapacityError(true))
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("nil server writeInferenceError code = %d, want 503", rec.Code)
	}
}

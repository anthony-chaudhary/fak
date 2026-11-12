package gateway

// arm_admission_test.go — the served-boundary contract of the research-arm capacity
// refusal: a saturated arm answers with a TYPED, closed-vocabulary 429 (machine-readable
// code + reason token + structured counters + a measured Retry-After), not free-text
// exception prose; and the admission queue is observable on /metrics.
//
// fak-test:runtime medium est=6s lane=default

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/researcharm"
)

// TestArmAdmissionRefusalTypedBody pins the shed wire shape: 429, a wait-derived
// Retry-After, the machine-readable code, the closed reason token, and the arm's counters.
func TestArmAdmissionRefusalTypedBody(t *testing.T) {
	coord := researcharm.NewCoordinator(1)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("X-Fak-Project-Arm", "gw-bench")
	held, err := coord.Admit(context.Background(), req, "/v1/chat/completions", "trace-held")
	if err != nil {
		t.Fatalf("hold a slot: %v", err)
	}
	defer held.Done(0, nil)

	_, satErr := coord.AdmitWithWait(context.Background(), req, "/v1/chat/completions", "trace-shed", 40*time.Millisecond)
	if satErr == nil {
		t.Fatal("expected the second request to be refused")
	}

	rec := httptest.NewRecorder()
	if !writeArmAdmissionRefusal(rec, satErr) {
		t.Fatalf("refusal was not rendered as an arm capacity refusal: %v", satErr)
	}
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429: %s", rec.Code, rec.Body.String())
	}
	if ra := rec.Header().Get("Retry-After"); ra == "" || ra == "0" {
		t.Fatalf("Retry-After = %q, want a positive value", ra)
	}

	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode body: %v (%s)", err, rec.Body.String())
	}
	if got := body["reason"]; got != researcharm.ReasonArmConcurrencySaturated {
		t.Fatalf("top-level reason = %v, want %q", got, researcharm.ReasonArmConcurrencySaturated)
	}
	errObj, ok := body["error"].(map[string]any)
	if !ok {
		t.Fatalf("body has no error object: %s", rec.Body.String())
	}
	if got := errObj["code"]; got != armRefusalCode {
		t.Fatalf("error.code = %v, want %q (a client must be able to branch on it)", got, armRefusalCode)
	}
	for _, key := range []string{"arm_id", "active", "limit", "queued", "waited_ms"} {
		if _, ok := errObj[key]; !ok {
			t.Fatalf("error object missing structured field %q: %s", key, rec.Body.String())
		}
	}
	if got := errObj["arm_id"]; got != "gw-bench" {
		t.Fatalf("error.arm_id = %v, want gw-bench", got)
	}
}

// TestArmAdmissionRefusalIgnoresNonCapacityErrors pins that a non-capacity coordinator
// error is NOT dressed as a capacity signal — a held lease keeps its own rendering.
func TestArmAdmissionRefusalIgnoresNonCapacityErrors(t *testing.T) {
	rec := httptest.NewRecorder()
	if writeArmAdmissionRefusal(rec, researcharm.ErrExclusiveLeaseHeld) {
		t.Fatal("a lease conflict must not be rendered as a capacity refusal")
	}
	if rec.Body.Len() != 0 || rec.Code != 200 {
		t.Fatalf("non-capacity error should leave the response untouched, got %d %q", rec.Code, rec.Body.String())
	}
}

// TestRefuseArmAdmissionFallsBackToProse pins the fallback: a non-capacity error keeps the
// historical prose 429 + Retry-After: 1, so lease handling is unchanged.
func TestRefuseArmAdmissionFallsBackToProse(t *testing.T) {
	srv := &Server{}
	rec := httptest.NewRecorder()
	srv.refuseArmAdmission(rec, researcharm.ErrExclusiveLeaseHeld)
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429", rec.Code)
	}
	if got := rec.Header().Get("Retry-After"); got != "1" {
		t.Fatalf("Retry-After = %q, want the historical 1 on the prose path", got)
	}
}

// TestArmAdmissionServedPathAnswersTyped429 pins the behaviour end-to-end through the
// gateway handler: a saturated arm yields the typed 429 (not the old prose body).
func TestArmAdmissionServedPathAnswersTyped429(t *testing.T) {
	srv := newArmsTestServer(t)
	coord := researcharm.NewCoordinator(1)
	srv.SetResearchArmCoordinator(coord)
	srv.SetArmAdmitWait(40 * time.Millisecond)
	mux := srv.Handler()

	holdReq := httptest.NewRequest(http.MethodPost, "/manual", nil)
	holdReq.Header.Set("X-Fak-Project-Arm", "arm-typed")
	held, err := coord.Admit(context.Background(), holdReq, "/manual", "trace-held")
	if err != nil {
		t.Fatalf("hold slot: %v", err)
	}
	defer held.Done(0, nil)

	body, _ := json.Marshal(map[string]any{"model": "fak-mock", "messages": []map[string]string{{"role": "user", "content": "ping"}}})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(body)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Fak-Project-Arm", "arm-typed")
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429: %s", rec.Code, rec.Body.String())
	}
	var parsed map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &parsed); err != nil {
		t.Fatalf("served refusal is not JSON (%v): %s", err, rec.Body.String())
	}
	if got := parsed["reason"]; got != researcharm.ReasonArmConcurrencySaturated {
		t.Fatalf("reason = %v, want %q — the served path must carry the closed token", got, researcharm.ReasonArmConcurrencySaturated)
	}
	errObj, _ := parsed["error"].(map[string]any)
	if errObj == nil || errObj["code"] != armRefusalCode {
		t.Fatalf("error.code = %v, want %q", errObj["code"], armRefusalCode)
	}
}

// TestArmQueueMetrics pins the queue telemetry: per-arm depth/saturation/counters, and
// nothing at all until a coordinator is wired.
func TestArmQueueMetrics(t *testing.T) {
	var off strings.Builder
	(&Server{}).writeArmQueueMetrics(&off)
	if off.Len() != 0 {
		t.Fatalf("no coordinator must emit nothing, got %q", off.String())
	}

	coord := researcharm.NewCoordinator(2)
	srv := &Server{}
	srv.SetResearchArmCoordinator(coord)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("X-Fak-Project-Arm", "gw-bench")
	l, err := coord.Admit(context.Background(), req, "/v1/chat/completions", "t")
	if err != nil {
		t.Fatalf("admit: %v", err)
	}
	defer l.Done(0, nil)

	var on strings.Builder
	srv.writeArmQueueMetrics(&on)
	out := on.String()
	for _, want := range []string{
		"fak_gateway_arm_active_requests{arm=\"gw-bench\"} 1",
		"fak_gateway_arm_concurrency_limit{arm=\"gw-bench\"} 2",
		"fak_gateway_arm_queued_requests{arm=\"gw-bench\"} 0",
		"fak_gateway_arm_waited_total{arm=\"gw-bench\"} 0",
		"fak_gateway_arm_admitted_after_wait_total{arm=\"gw-bench\"} 0",
		"fak_gateway_arm_sheds_total{arm=\"gw-bench\"} 0",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("metrics output missing %q:\n%s", want, out)
		}
	}
}

// TestArmAdmissionReasonDeclaredInDosToml is the closed-vocabulary conformance witness:
// the token the runtime emits at the shed boundary MUST be a declared, refusable dos.toml
// [reasons] table, or dos_check_reason ARM_CONCURRENCY_SATURATED returns known=false.
func TestArmAdmissionReasonDeclaredInDosToml(t *testing.T) {
	content := readRepoDosTomlForReversibility(t)
	header := "[reasons." + researcharm.ReasonArmConcurrencySaturated + "]"
	if !strings.Contains(content, header) {
		t.Fatalf("gateway emits refusal token %q but dos.toml has no %s table — "+
			"dos_check_reason %s would return known=false (UNCLASSIFIED drift)",
			researcharm.ReasonArmConcurrencySaturated, header, researcharm.ReasonArmConcurrencySaturated)
	}
	block := reversibilityDosReasonBlock(content, header)
	if !reversibilityDosReasonFieldTrue(block, "refusal") {
		t.Fatalf("reason %q is declared but not marked refusal = true", researcharm.ReasonArmConcurrencySaturated)
	}
}

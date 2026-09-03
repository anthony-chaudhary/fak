package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/control"
)

func TestGatewayControlApply_DryRun(t *testing.T) {
	srv := newTestControlGateway(t, Config{})
	handler := srv.Handler()

	for _, path := range []string{"/v1/control/apply?dry_run=true", "/v1/fak/control/apply?dry_run=true"} {
		depth := uint32(5)
		batchTokens := uint32(32768)
		reqBody := ControlApplyRequest{
			SpeculativeDraftDepth: &depth,
			MaxBatchTokens:        &batchTokens,
		}
		bodyBytes, _ := json.Marshal(reqBody)

		req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		rr := httptest.NewRecorder()
		handler.ServeHTTP(rr, req)

		if rr.Code != http.StatusOK {
			t.Fatalf("%s returned status %d (body: %s)", path, rr.Code, rr.Body.String())
		}
		if wit := rr.Header().Get("X-Fak-Witness"); wit != "verified-dry-run" {
			t.Errorf("expected X-Fak-Witness: verified-dry-run, got %q", wit)
		}

		var res control.ApplyResult
		if err := json.Unmarshal(rr.Body.Bytes(), &res); err != nil {
			t.Fatalf("failed to decode response: %v", err)
		}
		if res.Status != "dry_run" {
			t.Errorf("status = %s, want dry_run", res.Status)
		}
		if !res.Valid {
			t.Errorf("expected res.Valid = true")
		}
		if len(res.Diff) != 2 {
			t.Errorf("expected 2 diff entries, got %d", len(res.Diff))
		}

		// Ensure live epoch remains 1
		if srv.ConfigEpoch() != 1 {
			t.Errorf("live epoch mutated after dry-run: %d", srv.ConfigEpoch())
		}
	}
}

func TestGatewayControlApply_RelationalInvariantRejection(t *testing.T) {
	srv := newTestControlGateway(t, Config{})
	handler := srv.Handler()

	// Invariant 1 violation: max_batch_tokens (2048) < max_model_len (8192)
	batchTokens := uint32(2048)
	modelLen := uint32(8192)
	reqBody := ControlApplyRequest{
		MaxBatchTokens: &batchTokens,
		MaxModelLen:    &modelLen,
	}
	bodyBytes, _ := json.Marshal(reqBody)

	req := httptest.NewRequest(http.MethodPost, "/v1/control/apply", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 Bad Request, got %d (body: %s)", rr.Code, rr.Body.String())
	}

	bodyStr := rr.Body.String()
	if !strings.Contains(bodyStr, control.ErrRelationalInvariantBatchTokens) {
		t.Errorf("expected %s in error response: %s", control.ErrRelationalInvariantBatchTokens, bodyStr)
	}

	// Invariant 2 violation: speculative_draft_depth (12) > max_preallocated_draft_slots (8)
	depth := uint32(12)
	slots := uint32(8)
	reqBody2 := ControlApplyRequest{
		SpeculativeDraftDepth:     &depth,
		MaxPreallocatedDraftLimit: &slots,
	}
	bodyBytes2, _ := json.Marshal(reqBody2)

	req2 := httptest.NewRequest(http.MethodPost, "/v1/control/apply", bytes.NewReader(bodyBytes2))
	req2.Header.Set("Content-Type", "application/json")
	rr2 := httptest.NewRecorder()
	handler.ServeHTTP(rr2, req2)

	if rr2.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 Bad Request, got %d (body: %s)", rr2.Code, rr2.Body.String())
	}
	if !strings.Contains(rr2.Body.String(), control.ErrRelationalInvariantDraftDepth) {
		t.Errorf("expected %s in error response: %s", control.ErrRelationalInvariantDraftDepth, rr2.Body.String())
	}
}

func TestGatewayControlApply_CommitAndCanaryRollback(t *testing.T) {
	srv := newTestControlGateway(t, Config{})
	handler := srv.Handler()

	// 1. Successfully apply a candidate configuration
	depth := uint32(6)
	reqBody := ControlApplyRequest{
		SpeculativeDraftDepth: &depth,
	}
	bodyBytes, _ := json.Marshal(reqBody)

	req := httptest.NewRequest(http.MethodPost, "/v1/control/apply", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("apply failed with status %d: %s", rr.Code, rr.Body.String())
	}
	if rr.Header().Get("X-Fak-Witness") != "verified-atomic-swap" {
		t.Errorf("expected X-Fak-Witness: verified-atomic-swap, got %q", rr.Header().Get("X-Fak-Witness"))
	}
	if rr.Header().Get("X-Fak-Config-Epoch") != "2" {
		t.Errorf("expected epoch 2, got %q", rr.Header().Get("X-Fak-Config-Epoch"))
	}

	// 2. Query event stream: should record SYSTEM_CONFIG_APPLIED
	reqEvents := httptest.NewRequest(http.MethodGet, "/v1/control/events", nil)
	rrEvents := httptest.NewRecorder()
	handler.ServeHTTP(rrEvents, reqEvents)

	if rrEvents.Code != http.StatusOK {
		t.Fatalf("events failed with status %d: %s", rrEvents.Code, rrEvents.Body.String())
	}
	if !strings.Contains(rrEvents.Body.String(), control.EventSystemConfigApplied) {
		t.Errorf("expected %s in events: %s", control.EventSystemConfigApplied, rrEvents.Body.String())
	}

	// 3. Ingest canary degradation telemetry (5xx error rate > 0.1%)
	telemetrySample := control.TelemetrySample{
		TotalRequests: 1000,
		Errors5xx:     5, // 0.5% > 0.1% threshold
	}
	tBytes, _ := json.Marshal(telemetrySample)

	reqTelemetry := httptest.NewRequest(http.MethodPost, "/v1/control/telemetry", bytes.NewReader(tBytes))
	reqTelemetry.Header.Set("Content-Type", "application/json")
	rrTelemetry := httptest.NewRecorder()
	handler.ServeHTTP(rrTelemetry, reqTelemetry)

	if rrTelemetry.Code != http.StatusOK {
		t.Fatalf("telemetry ingestion failed with status %d: %s", rrTelemetry.Code, rrTelemetry.Body.String())
	}

	var tResp struct {
		Triggered   bool                  `json:"triggered"`
		Trigger     string                `json:"trigger"`
		ConfigEpoch uint64                `json:"config_epoch"`
		Config      control.ServingConfig `json:"config"`
	}
	if err := json.Unmarshal(rrTelemetry.Body.Bytes(), &tResp); err != nil {
		t.Fatalf("failed to parse telemetry response: %v", err)
	}

	if !tResp.Triggered {
		t.Fatalf("expected rollback to be triggered")
	}
	if tResp.Trigger != control.Trigger5xxErrorRateExceeded {
		t.Errorf("trigger = %s, want %s", tResp.Trigger, control.Trigger5xxErrorRateExceeded)
	}
	if tResp.ConfigEpoch != 3 {
		t.Errorf("expected epoch 3 after automatic rollback, got %d", tResp.ConfigEpoch)
	}
	if tResp.Config.SpeculativeDraftDepth != 3 { // DefaultConfig had 3
		t.Errorf("config draft depth = %d, want LKG baseline 3", tResp.Config.SpeculativeDraftDepth)
	}

	// 4. Verify SYSTEM_CONFIG_AUTOMATIC_ROLLBACK in audit event stream
	rrEvents2 := httptest.NewRecorder()
	handler.ServeHTTP(rrEvents2, reqEvents)
	if !strings.Contains(rrEvents2.Body.String(), control.EventSystemConfigAutomaticRollback) {
		t.Errorf("expected %s in events after rollback: %s", control.EventSystemConfigAutomaticRollback, rrEvents2.Body.String())
	}
}

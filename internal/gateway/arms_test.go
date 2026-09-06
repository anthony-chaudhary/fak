package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/researcharm"
)

func newArmsTestServer(t *testing.T) *Server {
	t.Helper()
	abi.RegisterEngine("mock", engine.MockEngine)
	srv, err := New(Config{EngineID: "mock", Model: "fak-mock"})
	if err != nil {
		t.Fatalf("failed to create server: %v", err)
	}
	t.Cleanup(srv.Close)
	return srv
}

func TestGatewayArmsEndpoints(t *testing.T) {
	srv := newArmsTestServer(t)
	coord := researcharm.NewCoordinator(5)
	srv.SetResearchArmCoordinator(coord)

	mux := srv.Handler()

	// 1. GET /v1/fak/arms initially empty
	req := httptest.NewRequest("GET", "/v1/fak/arms", nil)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}

	var snap researcharm.Snapshot
	if err := json.NewDecoder(rec.Body).Decode(&snap); err != nil {
		t.Fatalf("failed to decode snapshot: %v", err)
	}
	if snap.TotalArms != 0 {
		t.Errorf("expected 0 arms, got %d", snap.TotalArms)
	}

	// 2. POST /v1/fak/arms/lease to acquire a lease
	leaseReqBody := researcharm.LeaseRequest{
		ArmID:       "bench-arm",
		Mode:        researcharm.LeaseModeShared,
		Concurrency: 2,
		TTL:         10 * time.Minute,
	}
	bodyBytes, _ := json.Marshal(leaseReqBody)
	req = httptest.NewRequest("POST", "/v1/fak/arms/lease", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusCreated {
		t.Fatalf("expected 201 Created, got %d: %s", rec.Code, rec.Body.String())
	}

	var acquiredLease researcharm.LeaseInfo
	if err := json.NewDecoder(rec.Body).Decode(&acquiredLease); err != nil {
		t.Fatalf("failed to decode lease: %v", err)
	}
	if acquiredLease.ArmID != "bench-arm" {
		t.Errorf("expected ArmID bench-arm, got %s", acquiredLease.ArmID)
	}
	if acquiredLease.Token == "" {
		t.Errorf("expected token in lease response")
	}

	// 3. GET /v1/fak/arms/lease to list active leases
	req = httptest.NewRequest("GET", "/v1/fak/arms/lease", nil)
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	var leases []researcharm.LeaseInfo
	if err := json.NewDecoder(rec.Body).Decode(&leases); err != nil {
		t.Fatalf("failed to decode leases: %v", err)
	}
	if len(leases) != 1 || leases[0].ArmID != "bench-arm" {
		t.Errorf("expected 1 lease for bench-arm, got %+v", leases)
	}

	// 4. POST /v1/fak/arms/limits
	limReqBody := researcharm.LimitRequest{
		ArmID:          "bench-arm",
		MaxConcurrency: 4,
	}
	bodyBytes, _ = json.Marshal(limReqBody)
	req = httptest.NewRequest("POST", "/v1/fak/arms/limits", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}

	// 5. DELETE /v1/fak/arms/lease to release
	req = httptest.NewRequest("DELETE", "/v1/fak/arms/lease?id="+acquiredLease.ID+"&token="+acquiredLease.Token, nil)
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestGatewayArmsThrottling(t *testing.T) {
	srv := newArmsTestServer(t)
	coord := researcharm.NewCoordinator(1) // Limit = 1 request per arm
	srv.SetResearchArmCoordinator(coord)

	mux := srv.Handler()

	// Acquire and occupy the 1 concurrency slot for arm "arm-throttle"
	reqManual := httptest.NewRequest("POST", "/manual", nil)
	reqManual.Header.Set("X-Fak-Project-Arm", "arm-throttle")
	l1, err := coord.Admit(context.Background(), reqManual, "/manual", "trace-manual")
	if err != nil {
		t.Fatalf("manual admit failed: %v", err)
	}
	defer l1.Done(0, nil)

	// Send request with same arm "arm-throttle" -> should fail with 429 Too Many Requests
	chatBody := map[string]any{
		"model": "fak-mock",
		"messages": []map[string]string{
			{"role": "user", "content": "ping"},
		},
	}
	bodyBytes, _ := json.Marshal(chatBody)
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Fak-Project-Arm", "arm-throttle")
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("expected 429 Too Many Requests, got %d: %s", rec.Code, rec.Body.String())
	}
	if rec.Header().Get("Retry-After") != "1" {
		t.Errorf("expected Retry-After header 1, got %s", rec.Header().Get("Retry-After"))
	}
}

func TestGatewayArmsExclusiveLease(t *testing.T) {
	srv := newArmsTestServer(t)
	coord := researcharm.NewCoordinator(10)
	srv.SetResearchArmCoordinator(coord)

	// Acquire exclusive lease for arm "owner-arm"
	_, err := coord.AcquireLease(researcharm.LeaseRequest{
		ArmID: "owner-arm",
		Mode:  researcharm.LeaseModeExclusive,
		TTL:   5 * time.Minute,
	})
	if err != nil {
		t.Fatalf("acquire exclusive lease failed: %v", err)
	}

	mux := srv.Handler()

	// Request from "other-arm" should fail with 429
	chatBody := map[string]any{
		"model": "fak-mock",
		"messages": []map[string]string{
			{"role": "user", "content": "ping"},
		},
	}
	bodyBytes, _ := json.Marshal(chatBody)
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Fak-Project-Arm", "other-arm")
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("expected 429 Too Many Requests on exclusive hold, got %d: %s", rec.Code, rec.Body.String())
	}

	// Request from "owner-arm" should succeed
	reqOwner := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(bodyBytes))
	reqOwner.Header.Set("Content-Type", "application/json")
	reqOwner.Header.Set("X-Fak-Project-Arm", "owner-arm")
	recOwner := httptest.NewRecorder()
	mux.ServeHTTP(recOwner, reqOwner)

	if recOwner.Code != http.StatusOK {
		t.Fatalf("expected 200 OK for lease owner, got %d: %s", recOwner.Code, recOwner.Body.String())
	}
}

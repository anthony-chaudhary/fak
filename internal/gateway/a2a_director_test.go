package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/director"
)

// TestA2ADirectorDigestReturns200OK verifies GET /a2a/v1/director/digest returns 200 OK
// with valid JSON matching director.DirectorDigest schema.
func TestA2ADirectorDigestReturns200OK(t *testing.T) {
	srv, err := New(Config{
		EngineID: "mock",
		Model:    "test-model",
	})
	if err != nil {
		t.Fatalf("failed to create gateway server: %v", err)
	}
	defer srv.Close()

	req := httptest.NewRequest(http.MethodGet, "/a2a/v1/director/digest", nil)
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected HTTP 200 OK, got %d (body: %s)", resp.StatusCode, w.Body.String())
	}

	contentType := resp.Header.Get("Content-Type")
	if !strings.Contains(contentType, "application/json") {
		t.Fatalf("expected application/json Content-Type, got %q", contentType)
	}

	var digest director.DirectorDigest
	if err := json.Unmarshal(w.Body.Bytes(), &digest); err != nil {
		t.Fatalf("failed to unmarshal DirectorDigest JSON: %v (raw: %s)", err, w.Body.String())
	}

	if digest.Schema != director.DigestSchema {
		t.Errorf("expected schema %q, got %q", director.DigestSchema, digest.Schema)
	}
	if digest.Timestamp <= 0 {
		t.Errorf("expected positive timestamp, got %d", digest.Timestamp)
	}
	if !strings.HasPrefix(digest.RollupHash, "sha256:") {
		t.Errorf("expected RollupHash with sha256: prefix, got %q", digest.RollupHash)
	}
}

// TestA2ADirectorDigestWithRecordedState verifies that tracked workers and leases
// in RollupEngine are correctly compiled and served over GET /a2a/v1/director/digest.
func TestA2ADirectorDigestWithRecordedState(t *testing.T) {
	srv, err := New(Config{
		EngineID: "mock",
		Model:    "test-model",
	})
	if err != nil {
		t.Fatalf("failed to create gateway server: %v", err)
	}
	defer srv.Close()

	engine := director.NewRollupEngine()
	engine.RecordWorker(director.WorkerDigestRow{
		RunID:           "RID-DIR-001",
		Lane:            "gateway",
		Issue:           "#11411",
		State:           director.WorkerHealthy,
		StepCount:       15,
		VerifiedCommits: 3,
		TreeTouches:     6,
		VelocityScore:   2.5,
		LastWitnessMs:   time.Now().UnixMilli(),
	})
	engine.RecordLease(director.LeaseSnapshot{
		Lane:     "gateway",
		LaneKind: director.LaneKindCluster,
		Tree:     []string{"internal/gateway/**"},
		Holder:   "RID-DIR-001",
		Mode:     director.LeaseModeExclusive,
	})

	SetDirectorEngine(engine)

	req := httptest.NewRequest(http.MethodGet, "/a2a/v1/director/digest", nil)
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected HTTP 200 OK, got %d (body: %s)", resp.StatusCode, w.Body.String())
	}

	var digest director.DirectorDigest
	if err := json.Unmarshal(w.Body.Bytes(), &digest); err != nil {
		t.Fatalf("failed to unmarshal DirectorDigest: %v", err)
	}

	if digest.TotalWorkers != 1 || digest.ActiveWorkers != 1 {
		t.Fatalf("expected 1 active worker, got total=%d active=%d", digest.TotalWorkers, digest.ActiveWorkers)
	}
	if len(digest.Workers) != 1 || digest.Workers[0].RunID != "RID-DIR-001" {
		t.Fatalf("unexpected workers list: %+v", digest.Workers)
	}
	if len(digest.Leases) != 1 || digest.Leases[0].Lane != "gateway" {
		t.Fatalf("unexpected leases list: %+v", digest.Leases)
	}

	// Verify zero-prose invariant at HTTP boundary
	var rawMap map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &rawMap); err != nil {
		t.Fatalf("failed to unmarshal to raw map: %v", err)
	}

	prohibited := []string{"message", "prose", "text", "summary", "description", "comment", "claim", "narrative"}
	for _, p := range prohibited {
		if _, exists := rawMap[p]; exists {
			t.Fatalf("HTTP payload leaked forbidden prose key %q", p)
		}
	}
}

// TestA2ADirectorDigestMethodNotAllowed verifies non-GET requests are rejected with 405.
func TestA2ADirectorDigestMethodNotAllowed(t *testing.T) {
	srv, err := New(Config{
		EngineID: "mock",
		Model:    "test-model",
	})
	if err != nil {
		t.Fatalf("failed to create gateway server: %v", err)
	}
	defer srv.Close()

	req := httptest.NewRequest(http.MethodPost, "/a2a/v1/director/digest", strings.NewReader(`{}`))
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("expected HTTP 405 Method Not Allowed, got %d (body: %s)", resp.StatusCode, w.Body.String())
	}
}

package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/internal/guardvars"
)

const observationEndpointPath = "/v1/fak/observation"

type observationSnapshotWire struct {
	Schema     string `json:"schema"`
	ObservedAt string `json:"observed_at"`
	Sources    struct {
		Sessions         guardvars.ObservationEnvelope `json:"sessions"`
		CacheAttribution guardvars.ObservationEnvelope `json:"cache_attribution"`
		ManagedCache     guardvars.ObservationEnvelope `json:"managed_cache"`
		Harness          guardvars.ObservationEnvelope `json:"harness"`
	} `json:"sources"`
}

type observationSessionData struct {
	Sessions []guardvars.SessionVars `json:"sessions"`
	Count    int                     `json:"count"`
}

func TestObservationSnapshotEndpointStateAndLegacyParity(t *testing.T) {
	t.Run("observed zero stays data", func(t *testing.T) {
		restoreCacheObserver := swapCacheObserver(cacheobs.New())
		t.Cleanup(restoreCacheObserver)

		srv := newObservationTestServer(t)
		srv.listSessions = func(context.Context) []SessionState { return []SessionState{} }
		srv.cacheTTL1H = true
		srv.SetSessionHarnessProvider(func() SessionHarness { return SessionHarness{} })

		snapshot, raw := getObservationSnapshot(t, srv)
		assertObservationBoundary(t, snapshot)

		if got := snapshot.Sources.Sessions.Availability; got != guardvars.AvailabilityObserved {
			t.Fatalf("sessions availability = %q, want OBSERVED for a clean zero", got)
		}
		var sessions observationSessionData
		decodeObservationData(t, snapshot.Sources.Sessions, &sessions)
		if sessions.Count != 0 || sessions.Sessions == nil {
			t.Fatalf("sessions zero = count %d rows %#v, want explicit count=0 and []", sessions.Count, sessions.Sessions)
		}

		if got := snapshot.Sources.CacheAttribution.Availability; got != guardvars.AvailabilityObserved {
			t.Fatalf("cache availability = %q, want OBSERVED for measured zero counters", got)
		}
		var cache guardvars.CacheAttributionVars
		decodeObservationData(t, snapshot.Sources.CacheAttribution, &cache)
		if !reflect.DeepEqual(cache, guardvars.CacheAttributionVars{}) {
			t.Fatalf("cache zero = %+v, want the zero value", cache)
		}

		if got := snapshot.Sources.ManagedCache.Availability; got != guardvars.AvailabilityObserved {
			t.Fatalf("managed-cache availability = %q, want OBSERVED for active zero", got)
		}
		var managed guardvars.ManagedCacheVars
		decodeObservationData(t, snapshot.Sources.ManagedCache, &managed)
		if !managed.Active || managed.Upgraded != 0 {
			t.Fatalf("managed-cache zero = %+v, want active=true upgraded=0", managed)
		}

		if got := snapshot.Sources.Harness.Availability; got != guardvars.AvailabilityEmpty {
			t.Fatalf("harness availability = %q, want EMPTY before the first sample", got)
		}
		if len(snapshot.Sources.Harness.Data) != 0 {
			t.Fatalf("EMPTY harness carried data: %s", snapshot.Sources.Harness.Data)
		}
		assertObservationPayloadFree(t, raw)
		t.Logf("observed-zero snapshot: %s", raw)
	})

	t.Run("nonzero values match legacy debug vars", func(t *testing.T) {
		srv := newObservationTestServer(t)
		srv.listSessions = func(context.Context) []SessionState {
			return []SessionState{{
				TraceID:  "trace-1",
				Run:      "running",
				Priority: 7,
				Budget:   SessionBudget{TurnsLeft: 9, TokensLeft: 800},
				Time:     SessionTime{ElapsedSeconds: 12},
			}}
		}
		srv.cacheTTL1H = true
		srv.metrics.observeInference(1000, 50, 600, 200, "end_turn", time.Second)
		srv.metrics.observeCacheTTLUpgrade("")
		srv.SetSessionHarnessProvider(func() SessionHarness {
			return SessionHarness{Samples: 3, KernelCPUPercent: 12.5, KernelRSSBytes: 4096}
		})

		snapshot, raw := getObservationSnapshot(t, srv)
		assertObservationBoundary(t, snapshot)
		legacy := getLegacyDebugVars(t, srv)

		var sessions observationSessionData
		decodeObservationData(t, snapshot.Sources.Sessions, &sessions)
		if sessions.Count != 1 || len(sessions.Sessions) != 1 {
			t.Fatalf("snapshot sessions = %+v, want one live row", sessions)
		}
		if len(legacy.Sessions) != 1 || !reflect.DeepEqual(legacy.Sessions[0], sessions.Sessions[0]) {
			t.Fatalf("legacy sessions = %+v, snapshot = %+v", legacy.Sessions, sessions.Sessions)
		}

		var cache guardvars.CacheAttributionVars
		decodeObservationData(t, snapshot.Sources.CacheAttribution, &cache)
		if legacy.CacheAttribution == nil || !reflect.DeepEqual(*legacy.CacheAttribution, cache) {
			t.Fatalf("legacy cache = %+v, snapshot = %+v", legacy.CacheAttribution, cache)
		}

		var managed guardvars.ManagedCacheVars
		decodeObservationData(t, snapshot.Sources.ManagedCache, &managed)
		if legacy.ManagedCache == nil || !reflect.DeepEqual(*legacy.ManagedCache, managed) {
			t.Fatalf("legacy managed-cache = %+v, snapshot = %+v", legacy.ManagedCache, managed)
		}

		var harness SessionHarness
		decodeObservationData(t, snapshot.Sources.Harness, &harness)
		if legacy.Harness == nil || *legacy.Harness != harness {
			t.Fatalf("legacy harness = %+v, snapshot = %+v", legacy.Harness, harness)
		}

		assertObservationPayloadFree(t, raw)
		t.Logf("nonzero snapshot: %s", raw)
	})

	t.Run("unavailable sources carry closed reasons", func(t *testing.T) {
		srv := newObservationTestServer(t)
		snapshot, raw := getObservationSnapshot(t, srv)
		assertObservationBoundary(t, snapshot)

		if got := snapshot.Sources.Sessions.Availability; got != guardvars.AvailabilityUnavailable {
			t.Fatalf("sessions availability = %q, want UNAVAILABLE", got)
		}
		if got := snapshot.Sources.Sessions.Reason; got != "SESSION_SOURCE_NOT_CONFIGURED" {
			t.Fatalf("sessions reason = %q, want SESSION_SOURCE_NOT_CONFIGURED", got)
		}
		if got := snapshot.Sources.Harness.Availability; got != guardvars.AvailabilityUnavailable {
			t.Fatalf("harness availability = %q, want UNAVAILABLE", got)
		}
		if got := snapshot.Sources.Harness.Reason; got != "HARNESS_SOURCE_NOT_CONFIGURED" {
			t.Fatalf("harness reason = %q, want HARNESS_SOURCE_NOT_CONFIGURED", got)
		}
		if got := snapshot.Sources.ManagedCache.Availability; got != guardvars.AvailabilityNotApplicable {
			t.Fatalf("managed-cache availability = %q, want NOT_APPLICABLE while disabled", got)
		}
		if got := snapshot.Sources.ManagedCache.Reason; got != "MANAGED_CACHE_DISABLED" {
			t.Fatalf("managed-cache reason = %q, want MANAGED_CACHE_DISABLED", got)
		}
		assertObservationPayloadFree(t, raw)
		t.Logf("unavailable snapshot: %s", raw)
	})

	t.Run("source failures become typed unavailable states", func(t *testing.T) {
		srv := newObservationTestServer(t)
		srv.listSessions = func(context.Context) []SessionState {
			panic("session source failure must not escape")
		}
		srv.SetSessionHarnessProvider(func() SessionHarness {
			panic("harness source failure must not escape")
		})

		snapshot, raw := getObservationSnapshot(t, srv)
		assertObservationBoundary(t, snapshot)
		if got := snapshot.Sources.Sessions.Availability; got != guardvars.AvailabilityUnavailable {
			t.Fatalf("failed sessions availability = %q, want UNAVAILABLE", got)
		}
		if got := snapshot.Sources.Sessions.Reason; got != "SESSION_SOURCE_READ_FAILED" {
			t.Fatalf("failed sessions reason = %q, want SESSION_SOURCE_READ_FAILED", got)
		}
		if got := snapshot.Sources.Harness.Availability; got != guardvars.AvailabilityUnavailable {
			t.Fatalf("failed harness availability = %q, want UNAVAILABLE", got)
		}
		if got := snapshot.Sources.Harness.Reason; got != "HARNESS_SOURCE_READ_FAILED" {
			t.Fatalf("failed harness reason = %q, want HARNESS_SOURCE_READ_FAILED", got)
		}
		assertObservationPayloadFree(t, raw)
		t.Logf("source-failure snapshot: %s", raw)
	})

	t.Run("stale source keeps its own observation clock", func(t *testing.T) {
		srv := newObservationTestServer(t)
		sampledAt := time.Date(2026, 8, 20, 18, 0, 0, 0, time.UTC)
		srv.SetSessionHarnessObservationProvider(func() SessionHarnessObservation {
			return SessionHarnessObservation{
				Availability: guardvars.AvailabilityStale,
				ObservedAt:   sampledAt,
				Revision:     "harness-sample-7",
			}
		})

		snapshot, raw := getObservationSnapshot(t, srv)
		harness := snapshot.Sources.Harness
		if got := harness.Availability; got != guardvars.AvailabilityStale {
			t.Fatalf("stale harness availability = %q, want STALE", got)
		}
		if got := harness.Reason; got != "HARNESS_SAMPLE_STALE" {
			t.Fatalf("stale harness reason = %q, want HARNESS_SAMPLE_STALE", got)
		}
		if harness.ObservedAt != sampledAt.Format(time.RFC3339Nano) || harness.Revision != "harness-sample-7" {
			t.Fatalf("stale harness metadata = observed_at %q revision %q", harness.ObservedAt, harness.Revision)
		}
		if len(harness.Data) != 0 {
			t.Fatalf("stale harness carried data: %s", harness.Data)
		}
		if err := harness.Validate(); err != nil {
			t.Fatalf("stale harness envelope invalid: %v", err)
		}
		assertObservationPayloadFree(t, raw)
		t.Logf("stale snapshot: %s", raw)
	})
}

func newObservationTestServer(t *testing.T) *Server {
	t.Helper()
	srv := newTestServer(t)
	srv.requireKey = "snapshot-admin-secret"
	srv.readBearer = "snapshot-read-secret"
	srv.model = "https://private.example/model?credential=must-not-leak"
	return srv
}

func getObservationSnapshot(t *testing.T, srv *Server) (observationSnapshotWire, []byte) {
	t.Helper()
	h := srv.Handler()

	unauthorized := httptest.NewRequest(http.MethodGet, observationEndpointPath, nil)
	unauthorized.RemoteAddr = "203.0.113.8:40000"
	unauthorizedRec := httptest.NewRecorder()
	h.ServeHTTP(unauthorizedRec, unauthorized)
	if unauthorizedRec.Code != http.StatusUnauthorized {
		t.Fatalf("remote observation without credentials = %d, want 401", unauthorizedRec.Code)
	}

	req := httptest.NewRequest(http.MethodGet, observationEndpointPath, nil)
	req.RemoteAddr = "203.0.113.8:40001"
	req.Header.Set("Authorization", "Bearer snapshot-read-secret")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("GET %s with read bearer = %d, want 200; body=%s", observationEndpointPath, rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("Content-Type"); !strings.HasPrefix(got, "application/json") {
		t.Fatalf("content-type = %q, want application/json", got)
	}
	raw := append([]byte(nil), rec.Body.Bytes()...)
	var snapshot observationSnapshotWire
	if err := json.Unmarshal(raw, &snapshot); err != nil {
		t.Fatalf("decode observation snapshot: %v\n%s", err, raw)
	}
	return snapshot, raw
}

func getLegacyDebugVars(t *testing.T, srv *Server) debugVarsResponse {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, "/debug/vars", nil)
	req.RemoteAddr = "203.0.113.8:40002"
	req.Header.Set("Authorization", "Bearer snapshot-read-secret")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /debug/vars = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	var out debugVarsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode /debug/vars: %v", err)
	}
	return out
}

func decodeObservationData(t *testing.T, envelope guardvars.ObservationEnvelope, out any) {
	t.Helper()
	if err := envelope.Validate(); err != nil {
		t.Fatalf("%s envelope invalid: %v", envelope.Source, err)
	}
	if err := json.Unmarshal(envelope.Data, out); err != nil {
		t.Fatalf("decode %s data: %v; data=%s", envelope.Source, err, envelope.Data)
	}
}

func assertObservationBoundary(t *testing.T, snapshot observationSnapshotWire) {
	t.Helper()
	if snapshot.Schema != "fak-observation-snapshot/1" {
		t.Fatalf("snapshot schema = %q, want fak-observation-snapshot/1", snapshot.Schema)
	}
	if _, err := time.Parse(time.RFC3339Nano, snapshot.ObservedAt); err != nil {
		t.Fatalf("snapshot observed_at = %q: %v", snapshot.ObservedAt, err)
	}
	for name, envelope := range map[string]guardvars.ObservationEnvelope{
		"sessions":          snapshot.Sources.Sessions,
		"cache_attribution": snapshot.Sources.CacheAttribution,
		"managed_cache":     snapshot.Sources.ManagedCache,
		"harness":           snapshot.Sources.Harness,
	} {
		if err := envelope.Validate(); err != nil {
			t.Errorf("%s envelope invalid: %v", name, err)
		}
		if envelope.Source != name {
			t.Errorf("%s envelope source = %q", name, envelope.Source)
		}
		if envelope.ObservedAt != snapshot.ObservedAt {
			t.Errorf("%s observed_at = %q, want shared boundary %q", name, envelope.ObservedAt, snapshot.ObservedAt)
		}
	}
}

func assertObservationPayloadFree(t *testing.T, raw []byte) {
	t.Helper()
	body := string(raw)
	for _, secret := range []string{
		"snapshot-admin-secret",
		"snapshot-read-secret",
		"private.example",
		"must-not-leak",
	} {
		if strings.Contains(body, secret) {
			t.Errorf("observation leaked forbidden payload %q: %s", secret, body)
		}
	}
	for _, key := range []string{
		`"prompt"`,
		`"arguments"`,
		`"result"`,
		`"transcript"`,
		`"credential"`,
		`"url"`,
	} {
		if strings.Contains(strings.ToLower(body), key) {
			t.Errorf("observation contains forbidden payload key %s: %s", key, body)
		}
	}
}

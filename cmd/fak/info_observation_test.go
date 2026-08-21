package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/guardvars"
)

func TestInfoObservationStateRenderMatrix(t *testing.T) {
	tests := []struct {
		name         string
		availability guardvars.Availability
		value        *float64
		reason       string
	}{
		{name: "observed-zero", availability: guardvars.AvailabilityObserved, value: guardInfoObservationNumber(0)},
		{name: "observed-nonzero", availability: guardvars.AvailabilityObserved, value: guardInfoObservationNumber(3)},
		{name: "empty", availability: guardvars.AvailabilityEmpty},
		{name: "unavailable", availability: guardvars.AvailabilityUnavailable, reason: "SOURCE_READ_FAILED"},
		{name: "stale", availability: guardvars.AvailabilityStale, reason: "SAMPLE_STALE"},
		{name: "not-applicable", availability: guardvars.AvailabilityNotApplicable, reason: "WIRE_UNSUPPORTED"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			view := &guardInfoObservationView{
				Schema:         guardInfoObservationViewSchema,
				Transport:      guardInfoObservationTransportTyped,
				SnapshotSchema: gateway.ObservationSnapshotSchemaV1,
				Sessions: guardInfoObservationMetric{
					Availability: tc.availability,
					Value:        tc.value,
					Unit:         "live sessions",
					ObservedAt:   "2026-08-21T00:00:00Z",
					Revision:     "sessions-r7",
					Provenance:   "gateway.session_registry",
					Reason:       tc.reason,
					NextCheck:    guardInfoObservationNextCheck("sessions", tc.availability),
				},
				CacheAttribution: guardInfoObservationMetric{
					Availability: tc.availability,
					Value:        tc.value,
					Unit:         "token-equivalent",
					ObservedAt:   "2026-08-21T00:00:00Z",
					Revision:     "cache-r9",
					Provenance:   "gateway.cache_accounting",
					Reason:       tc.reason,
					NextCheck:    guardInfoObservationNextCheck("cache", tc.availability),
				},
			}
			v := guardInfoVars{Observation: view}
			if tc.availability == guardvars.AvailabilityObserved && tc.value != nil {
				v.Sessions = make([]guardInfoSession, int(*tc.value))
				v.CacheAttribution = &guardInfoCacheAttribution{TotalTokenEquiv: *tc.value}
			}

			tr := newGuardInfoTrend(guardInfoTrendCap)
			tr.push(v)
			wantTransport := "source   typed"
			wantSessions := guardInfoObservationMetricText("sessions", view.Sessions)
			wantCache := guardInfoObservationMetricText("cache", view.CacheAttribution)
			surfaces := map[string]string{
				"compact": renderGuardInfoLine(v),
				"visual":  renderGuardInfoVisualBlock(v, tr, 240, 0),
				"tabbed":  renderGuardInfoInteractiveBlock(infoViewState{active: viewCache}, v, tr, 240, 0),
			}
			for surface, captured := range surfaces {
				for _, want := range []string{wantSessions, wantCache} {
					if !strings.Contains(captured, want) {
						t.Fatalf("%s did not consume shared semantic text %q:\n%s", surface, want, captured)
					}
				}
				for _, freshness := range []string{"at=2026-08-21T00:00:00Z", "rev=sessions-r7", "rev=cache-r9"} {
					if !strings.Contains(captured, freshness) {
						t.Fatalf("%s omitted shared freshness %q:\n%s", surface, freshness, captured)
					}
				}
				if surface != "compact" && !strings.Contains(captured, wantTransport) {
					t.Fatalf("%s omitted typed transport provenance:\n%s", surface, captured)
				}
				if strings.Contains(captured, "SECRET_PAYLOAD") {
					t.Fatalf("%s leaked payload text:\n%s", surface, captured)
				}
				if tc.availability != guardvars.AvailabilityObserved &&
					(strings.Contains(captured, "nothing yet") || strings.Contains(captured, "no cache yet")) {
					t.Fatalf("%s collapsed %s into an ambiguous cold-cache phrase:\n%s", surface, tc.availability, captured)
				}
				if tc.name == "observed-zero" && strings.Contains(captured, "source unavailable") {
					t.Fatalf("%s turned measured zero into source failure:\n%s", surface, captured)
				}
				t.Logf("%s capture:\n%s", surface, captured)
			}

			raw, err := json.Marshal(v)
			if err != nil {
				t.Fatal(err)
			}
			var captured guardInfoVars
			if err := json.Unmarshal(raw, &captured); err != nil {
				t.Fatal(err)
			}
			if captured.Observation == nil || captured.Observation.Sessions.Availability != tc.availability ||
				captured.Observation.CacheAttribution.Availability != tc.availability {
				t.Fatalf("json semantic divergence: %s", raw)
			}
			if tc.value == nil {
				if captured.Observation.Sessions.Value != nil || captured.Observation.CacheAttribution.Value != nil {
					t.Fatalf("json invented a value for %s: %s", tc.name, raw)
				}
			} else if captured.Observation.Sessions.Value == nil || captured.Observation.CacheAttribution.Value == nil ||
				*captured.Observation.Sessions.Value != *tc.value || *captured.Observation.CacheAttribution.Value != *tc.value {
				t.Fatalf("json lost measured value %v: %s", *tc.value, raw)
			}
			t.Logf("json capture:\n%s", raw)

			for _, block := range []string{
				fitGuardInfoStatus(renderGuardInfoLine(v), 36),
				renderGuardInfoVisualBlock(v, tr, 36, 4),
				renderGuardInfoInteractiveBlock(infoViewState{active: viewCache}, v, tr, 36, 4),
			} {
				for _, row := range strings.Split(block, "\n") {
					if got := dispWidthTUI(row); got > 36 {
						t.Fatalf("narrow render escaped its 36-cell bound (%d): %q", got, row)
					}
				}
			}
		})
	}
}

func TestInfoObservationClientPrefersTypedAndMarksLegacyFallback(t *testing.T) {
	t.Run("typed snapshot overrides legacy overlap", func(t *testing.T) {
		var debugHits, typedHits int
		snapshot := gateway.ObservationSnapshot{
			Schema:     gateway.ObservationSnapshotSchemaV1,
			ObservedAt: "2026-08-21T00:00:00Z",
			Sources: gateway.ObservationSnapshotSources{
				Sessions: infoObservationEnvelope(t, "sessions", "gateway.session_registry", guardvars.AvailabilityObserved,
					gateway.ObservationSessionData{Sessions: make([]guardvars.SessionVars, 2), Count: 2}, ""),
				CacheAttribution: infoObservationEnvelope(t, "cache_attribution", "gateway.cache_accounting", guardvars.AvailabilityObserved,
					guardvars.CacheAttributionVars{TotalTokenEquiv: 42}, ""),
				ManagedCache: infoObservationEnvelope(t, "managed_cache", "gateway.managed_cache", guardvars.AvailabilityNotApplicable,
					nil, "MANAGED_CACHE_DISABLED"),
				Harness: infoObservationEnvelope(t, "harness", "host.harness_sampler", guardvars.AvailabilityEmpty,
					nil, ""),
			},
		}
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/debug/vars":
				debugHits++
				_, _ = w.Write([]byte(`{"sessions":[{}],"cache_attribution":{"total_token_equiv":1},"inference":{"turns":7}}`))
			case "/v1/fak/observation":
				typedHits++
				_ = json.NewEncoder(w).Encode(snapshot)
			default:
				http.NotFound(w, r)
			}
		}))
		defer srv.Close()

		v, err := readGuardInfoVars(&claudeMacDebugClient{base: srv.URL, hc: srv.Client()})
		if err != nil {
			t.Fatal(err)
		}
		if debugHits != 1 || typedHits != 1 || v.Observation == nil || v.Observation.Transport != guardInfoObservationTransportTyped {
			t.Fatalf("typed preference not witnessed: debug=%d typed=%d view=%+v", debugHits, typedHits, v.Observation)
		}
		if len(v.Sessions) != 2 || v.CacheAttribution == nil || v.CacheAttribution.TotalTokenEquiv != 42 || v.Inference.Turns != 7 {
			t.Fatalf("typed overlap did not override legacy while retaining legacy-only fields: %+v", v)
		}
	})

	t.Run("legacy fallback is explicit and cached", func(t *testing.T) {
		var typedHits int
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/v1/fak/observation" {
				typedHits++
				http.NotFound(w, r)
				return
			}
			_, _ = w.Write([]byte(`{"sessions":[],"cache_attribution":{"total_token_equiv":0}}`))
		}))
		defer srv.Close()
		client := &claudeMacDebugClient{base: srv.URL, hc: srv.Client()}
		for i := 0; i < 2; i++ {
			v, err := readGuardInfoVars(client)
			if err != nil {
				t.Fatal(err)
			}
			if v.Observation == nil || v.Observation.Transport != guardInfoObservationTransportLegacy ||
				v.Observation.FallbackReason != "VERSIONED_SNAPSHOT_UNSUPPORTED" ||
				v.Observation.Sessions.Availability != guardvars.AvailabilityEmpty ||
				v.Observation.CacheAttribution.Availability != guardvars.AvailabilityObserved ||
				v.Observation.CacheAttribution.Value == nil || *v.Observation.CacheAttribution.Value != 0 ||
				v.Observation.CacheAttribution.Provenance != "legacy.debug_vars/unknown" {
				t.Fatalf("legacy fallback semantics lost: %+v", v.Observation)
			}
		}
		if typedHits != 1 {
			t.Fatalf("definitive legacy gateway should be probed once, got %d typed hits", typedHits)
		}
	})
}

func infoObservationEnvelope(t *testing.T, source, provenance string, availability guardvars.Availability, data any, reason string) guardvars.ObservationEnvelope {
	t.Helper()
	envelope := guardvars.ObservationEnvelope{
		Schema:       guardvars.ObservationSchemaV1,
		Source:       source,
		ObservedAt:   "2026-08-21T00:00:00Z",
		Provenance:   provenance,
		Availability: availability,
		Reason:       reason,
	}
	if availability == guardvars.AvailabilityObserved {
		raw, err := json.Marshal(data)
		if err != nil {
			t.Fatal(err)
		}
		envelope.Data = raw
	}
	if err := envelope.Validate(); err != nil {
		t.Fatalf("invalid test envelope: %v", err)
	}
	return envelope
}

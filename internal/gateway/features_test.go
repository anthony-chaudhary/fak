package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/engine"
)

func TestFeaturesEndpoint_EnablementMatrix(t *testing.T) {
	abi.RegisterEngine("mock", engine.MockEngine)
	s, err := New(Config{EngineID: "mock", Model: "test-model", Provider: "openai"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Close)
	want, err := NewFeatureCatalog([]FeatureStatus{
		{Feature: FeatureRemoteKV, State: FeatureConfiguredStandby, Provenance: FeatureCLIFlag, Description: "remote cache requested; endpoint unresolved"},
		{Feature: FeaturePolicyFloor, State: FeatureConfiguredActive, Provenance: FeatureAutoDefault, Description: "capability floor installed"},
		{Feature: FeatureMetal, State: FeatureDisabled, Provenance: FeatureAutoDefault, Description: "Metal not requested"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SetFeatureCatalog(want); err != nil {
		t.Fatal(err)
	}

	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/fak/features", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("GET status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("Content-Type"); got != "application/json" {
		t.Fatalf("Content-Type = %q, want application/json", got)
	}
	var got FeatureCatalog
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("response = %#v, want %#v", got, want)
	}

	rec = httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/fak/features", nil))
	if rec.Code != http.StatusMethodNotAllowed || rec.Header().Get("Allow") != http.MethodGet {
		t.Fatalf("POST status/Allow = %d/%q, want 405/GET", rec.Code, rec.Header().Get("Allow"))
	}

	protected, err := New(Config{EngineID: "mock", Model: "test-model", Provider: "openai", RequireKey: "secret"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(protected.Close)
	rec = httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/fak/features", nil)
	req.RemoteAddr = "203.0.113.10:1234"
	protected.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated off-host GET status = %d, want 401", rec.Code)
	}
}

func TestFeatureCatalogValidationAndIsolation(t *testing.T) {
	input := []FeatureStatus{
		{Feature: FeatureVDSO, State: FeatureConfiguredStandby, Provenance: FeatureEnvVar, Description: "requested"},
		{Feature: FeatureMetal, State: FeatureDisabled, Provenance: FeatureAutoDefault, Description: "off"},
	}
	catalog, err := NewFeatureCatalog(input)
	if err != nil {
		t.Fatal(err)
	}
	input[0].Description = "mutated"
	if catalog.Features[1].Description != "requested" {
		t.Fatal("catalog aliases caller input")
	}
	if catalog.Features[0].Feature != FeatureMetal || catalog.Features[1].Feature != FeatureVDSO {
		t.Fatalf("features not sorted: %#v", catalog.Features)
	}
	empty, err := NewFeatureCatalog(nil)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(empty)
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != `{"schema":"fak-serve-features/1","features":[]}` {
		t.Fatalf("empty JSON = %s", raw)
	}

	invalid := []struct {
		name string
		rows []FeatureStatus
	}{
		{"unknown feature", []FeatureStatus{{Feature: "surprise", State: FeatureDisabled, Provenance: FeatureAutoDefault}}},
		{"unknown state", []FeatureStatus{{Feature: FeatureVDSO, State: "maybe", Provenance: FeatureAutoDefault}}},
		{"unknown provenance", []FeatureStatus{{Feature: FeatureVDSO, State: FeatureDisabled, Provenance: "guess"}}},
		{"duplicate including disabled", []FeatureStatus{{Feature: FeatureVDSO, State: FeatureDisabled, Provenance: FeatureAutoDefault}, {Feature: FeatureVDSO, State: FeatureConfiguredStandby, Provenance: FeatureCLIFlag}}},
	}
	for _, tc := range invalid {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NewFeatureCatalog(tc.rows); err == nil {
				t.Fatal("NewFeatureCatalog accepted invalid input")
			}
		})
	}

	s := &Server{}
	if err := s.SetFeatureCatalog(FeatureCatalog{Schema: "future/2", Features: catalog.Features}); err == nil {
		t.Fatal("SetFeatureCatalog accepted unknown schema")
	}
	if err := s.SetFeatureCatalog(catalog); err != nil {
		t.Fatal(err)
	}
	catalog.Features[0].Description = "caller mutation"
	first := s.FeatureSnapshot()
	first.Features[0].Description = "snapshot mutation"
	second := s.FeatureSnapshot()
	if second.Features[0].Description == "caller mutation" || second.Features[0].Description == "snapshot mutation" {
		t.Fatalf("published catalog aliases mutable storage: %#v", second)
	}
}

func TestFeatureCatalogConcurrentPublication(t *testing.T) {
	s := &Server{}
	active, _ := NewFeatureCatalog([]FeatureStatus{{Feature: FeaturePolicyFloor, State: FeatureConfiguredActive, Provenance: FeatureAutoDefault}})
	standby, _ := NewFeatureCatalog([]FeatureStatus{{Feature: FeatureRemoteKV, State: FeatureConfiguredStandby, Provenance: FeatureCLIFlag}})
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(2)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				catalog := active
				if (i+j)%2 != 0 {
					catalog = standby
				}
				if err := s.SetFeatureCatalog(catalog); err != nil {
					t.Errorf("publish: %v", err)
					return
				}
			}
		}(i)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				snapshot := s.FeatureSnapshot()
				if snapshot.Schema != FeatureSchema || len(snapshot.Features) > 1 {
					t.Errorf("torn snapshot: %#v", snapshot)
					return
				}
			}
		}()
	}
	wg.Wait()
}

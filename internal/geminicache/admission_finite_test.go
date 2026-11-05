package geminicache

import (
	"context"
	"io"
	"math"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"
)

type finiteAdmissionTransport func(*http.Request) (*http.Response, error)

func (f finiteAdmissionTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

// fak-test:runtime fast est=1ms lane=default
func TestCreateRefusesNonfiniteEconomicsBeforeTransport(t *testing.T) {
	t.Parallel()
	now := time.Date(2026, 11, 5, 12, 0, 0, 0, time.UTC)
	id := NewIdentity("test-account", "test-project", "test-location", "models/test-model", []byte("test prefix"))
	for _, tc := range []struct {
		name        string
		value, cost float64
		allowed     bool
	}{
		{"value NaN", math.NaN(), 1, false},
		{"cost NaN", 2, math.NaN(), false},
		{"value positive infinity", math.Inf(1), 1, false},
		{"value negative infinity", math.Inf(-1), 1, false},
		{"cost positive infinity", 2, math.Inf(1), false},
		{"cost negative infinity", 2, math.Inf(-1), false},
		{"finite profitable", 2, 1, true},
		{"finite break-even", 1, 1, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			client := Client{
				BaseURL: "https://cache.example.invalid/v1beta", Now: func() time.Time { return now },
				Capabilities: Capabilities{GenerateContent: true},
				HTTPClient: &http.Client{Transport: finiteAdmissionTransport(func(req *http.Request) (*http.Response, error) {
					calls++
					if req.Method != http.MethodPost || req.URL.Path != "/v1beta/cachedContents" {
						t.Fatalf("unexpected request: %s %s", req.Method, req.URL.Path)
					}
					return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(`{"name":"cachedContents/test-cache","model":"models/test-model"}`)), Header: make(http.Header)}, nil
				})},
			}
			admission := Admission{PredictedReuseValueUSD: tc.value, CreationStorageCostUSD: tc.cost, TTL: time.Hour, MaxTTL: 2 * time.Hour, PrivacyAllowed: true}
			if err := admission.Check(); (err == nil) != tc.allowed {
				t.Fatalf("Check error = %v, allowed = %v", err, tc.allowed)
			}
			got, err := client.Create(context.Background(), RouteGenerateContent, id, admission, CreateConfig{})
			if !tc.allowed {
				if err == nil || calls != 0 || !reflect.DeepEqual(got, Receipt{}) {
					t.Fatalf("invalid economics reached transport or returned a receipt: calls=%d receipt=%+v err=%v", calls, got, err)
				}
				return
			}
			if err != nil || calls != 1 || got.Object.Name != "cachedContents/test-cache" || got.Identity != id || got.State != StateActive || !got.Observed.Equal(now) {
				t.Fatalf("finite admission changed: calls=%d receipt=%+v err=%v", calls, got, err)
			}
		})
	}
}

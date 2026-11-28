package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// fak-test:runtime fast est=30ms
func TestRouteStatusHealthyAndJSON(t *testing.T) {
	payload := routeStatusPayload{
		Schema:         "fak-router-status/v1",
		RoutingEnabled: true,
		DefaultModel:   "default",
		Routes: []routeStatusEntry{
			{
				Model:          "default",
				ResidencyFloor: "",
				Primary: routeStatusTarget{
					Account:             "local-gpu",
					Kind:                "local",
					UpstreamModel:       "qwen-local",
					Locality:            "self_hosted",
					State:               "healthy",
					CooldownRemainingMs: 0,
				},
				Fallbacks: []routeStatusTarget{
					{
						Account:             "cloud-fb",
						Kind:                "openai",
						UpstreamModel:       "gpt-4o-mini",
						Locality:            "vendor",
						State:               "healthy",
						CooldownRemainingMs: 0,
					},
				},
			},
			{
				Model:          "private",
				ResidencyFloor: "local_only",
				Primary: routeStatusTarget{
					Account:             "local-gpu",
					Kind:                "local",
					UpstreamModel:       "qwen-local",
					Locality:            "self_hosted",
					State:               "healthy",
					CooldownRemainingMs: 0,
				},
			},
		},
		Totals: routeStatusTotals{
			Attempts:  10,
			Successes: 9,
			Failovers: 1,
		},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/router/status" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(payload)
	}))
	defer srv.Close()

	code, out, errb := runRT("status", "--addr", srv.URL)
	if code != 0 {
		t.Fatalf("expected exit 0, got %d (stderr=%s)", code, errb)
	}
	for _, want := range []string{
		"routing_enabled: true",
		"attempts=10 successes=9 failovers=1",
		"default",
		"private",
		"residency_floor=local_only",
		"locality=self_hosted",
		"locality=vendor",
		"upstream=qwen-local",
		"upstream=gpt-4o-mini",
		"state=healthy",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q in human output:\n%s", want, out)
		}
	}

	code, jsonOut, errb := runRT("status", "--addr", srv.URL, "--json")
	if code != 0 {
		t.Fatalf("expected exit 0 for --json, got %d (stderr=%s)", code, errb)
	}
	var decoded routeStatusPayload
	if err := json.Unmarshal([]byte(jsonOut), &decoded); err != nil {
		t.Fatalf("decode --json output: %v\n%s", err, jsonOut)
	}
	if decoded.Schema != "fak-router-status/v1" || len(decoded.Routes) != 2 || decoded.Totals.Failovers != 1 {
		t.Fatalf("unexpected decoded payload: %+v", decoded)
	}
}

// fak-test:runtime fast est=30ms
func TestRouteStatusNonLocalOnlyCooldownExitsZero(t *testing.T) {
	payload := routeStatusPayload{
		Schema:         "fak-router-status/v1",
		RoutingEnabled: true,
		DefaultModel:   "default",
		Routes: []routeStatusEntry{
			{
				Model: "default",
				Primary: routeStatusTarget{
					Account:             "local-gpu",
					Kind:                "local",
					UpstreamModel:       "qwen-local",
					Locality:            "self_hosted",
					State:               "cooldown",
					CooldownRemainingMs: 25000,
				},
				Fallbacks: []routeStatusTarget{
					{
						Account:             "cloud-fb",
						Kind:                "openai",
						UpstreamModel:       "gpt-4o-mini",
						Locality:            "vendor",
						State:               "healthy",
						CooldownRemainingMs: 0,
					},
				},
			},
		},
		Totals: routeStatusTotals{Attempts: 2, Successes: 1, Failovers: 1},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(payload)
	}))
	defer srv.Close()

	code, out, errb := runRT("status", "--addr", srv.URL)
	if code != 0 {
		t.Fatalf("expected exit 0 when non-local_only primary is cooling down, got %d (stderr=%s)", code, errb)
	}
	if !strings.Contains(out, "state=cooldown cooldown=25000ms") {
		t.Fatalf("expected cooldown details in output:\n%s", out)
	}
}

// fak-test:runtime fast est=30ms
func TestRouteStatusLocalOnlyPrimaryCooldownExitsOne(t *testing.T) {
	payload := routeStatusPayload{
		Schema:         "fak-router-status/v1",
		RoutingEnabled: true,
		DefaultModel:   "private",
		Routes: []routeStatusEntry{
			{
				Model:          "private",
				ResidencyFloor: "local_only",
				Primary: routeStatusTarget{
					Account:             "local-gpu",
					Kind:                "local",
					UpstreamModel:       "qwen-local",
					Locality:            "self_hosted",
					State:               "cooldown",
					CooldownRemainingMs: 12000,
				},
			},
		},
		Totals: routeStatusTotals{Attempts: 3, Successes: 2, Failovers: 0},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(payload)
	}))
	defer srv.Close()

	code, out, errb := runRT("status", "--addr", srv.URL)
	if code != 1 {
		t.Fatalf("expected exit 1 when local_only primary is cooling down, got %d (stderr=%s)", code, errb)
	}
	if !strings.Contains(out, "cooldown=12000ms") || !strings.Contains(out, "DEGRADED") {
		t.Fatalf("expected cooldown + DEGRADED marker in output:\n%s", out)
	}

	code, jsonOut, errb := runRT("status", "--addr", srv.URL, "--json")
	if code != 1 {
		t.Fatalf("expected exit 1 with --json when local_only primary is cooling down, got %d (stderr=%s)", code, errb)
	}
	var decoded routeStatusPayload
	if err := json.Unmarshal([]byte(jsonOut), &decoded); err != nil {
		t.Fatalf("decode --json output: %v\n%s", err, jsonOut)
	}
	if decoded.Routes[0].Primary.CooldownRemainingMs != 12000 {
		t.Fatalf("unexpected cooldown_remaining_ms: %+v", decoded.Routes[0].Primary)
	}
}

// fak-test:runtime fast est=30ms
func TestRouteStatusUsageAndServerErrors(t *testing.T) {
	code, _, errb := runRT("status", "extra-arg")
	if code != 2 {
		t.Fatalf("expected exit 2 for unexpected arg, got %d (stderr=%s)", code, errb)
	}

	code, _, errb = runRT("status", "--addr", "   ")
	if code != 2 {
		t.Fatalf("expected exit 2 for empty --addr, got %d (stderr=%s)", code, errb)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "nope", http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	code, _, errb = runRT("status", "--addr", srv.URL)
	if code != 1 {
		t.Fatalf("expected exit 1 on HTTP 503, got %d (stderr=%s)", code, errb)
	}
	if !strings.Contains(errb, "HTTP 503") {
		t.Fatalf("expected HTTP 503 in stderr, got: %s", errb)
	}
}

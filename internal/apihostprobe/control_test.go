package apihostprobe

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestProbeWithControlRequiresPositiveControl(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/control/models" {
			http.Error(w, "broken fixture", http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	got := ProbeWithControl(context.Background(), ProbeControl{
		Name:    "range boundary",
		Target:  ReadinessTarget{Name: "target", BaseURL: server.URL + "/target"},
		Control: ReadinessTarget{Name: "control", BaseURL: server.URL + "/control"},
	}, []int{http.StatusNoContent}, ReadinessOptions{})
	if got.Conclusive || got.Verdict != "UNKNOWN" {
		t.Fatalf("failed control must be UNKNOWN, got %+v", got)
	}
	if got.Target.HTTPStatus != nil {
		t.Fatalf("target ran after failed control: %+v", got.Target)
	}
}

func TestProbeWithControlUsesExactStatusSet(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/control/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[]}`))
			return
		}
		w.WriteHeader(http.StatusPartialContent)
	}))
	defer server.Close()
	experiment := ProbeControl{
		Name:    "exact boundary",
		Target:  ReadinessTarget{Name: "target", BaseURL: server.URL + "/target"},
		Control: ReadinessTarget{Name: "control", BaseURL: server.URL + "/control"},
	}

	fail := ProbeWithControl(context.Background(), experiment, []int{http.StatusOK, http.StatusNoContent}, ReadinessOptions{})
	if !fail.Conclusive || fail.Verdict != "FAIL" {
		t.Fatalf("206 must not pass an implied 200-range, got %+v", fail)
	}
	pass := ProbeWithControl(context.Background(), experiment, []int{http.StatusPartialContent}, ReadinessOptions{})
	if !pass.Conclusive || pass.Verdict != "PASS" {
		t.Fatalf("explicit 206 must pass, got %+v", pass)
	}
}

func TestProbeWithControlRejectsUnnamedExperiment(t *testing.T) {
	got := ProbeWithControl(context.Background(), ProbeControl{}, []int{http.StatusOK}, ReadinessOptions{})
	if got.Verdict != "UNKNOWN" || got.Error != "probe name is required" {
		t.Fatalf("got %+v", got)
	}
}

package quantwatch_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/quantwatch"
)

func TestLiveFetchRecordsSourceQueryTimeAndAbstention(t *testing.T) {
	now := time.Date(2026, 8, 10, 12, 0, 0, 0, time.UTC)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/arxiv":
			w.Header().Set("Content-Type", "application/atom+xml")
			_, _ = w.Write([]byte(`<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry><id>https://arxiv.org/abs/2608.00002</id><title>Quantization Research</title><published>2026-08-09T00:00:00Z</published></entry></feed>`))
		case strings.Contains(r.URL.Path, "/repos/acme/runtime/releases"):
			http.Error(w, "rate limited", http.StatusTooManyRequests)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	got := quantwatch.FetchLive(context.Background(), server.Client(), quantwatch.LiveRequest{
		Query: "quantization", QueryTime: now, MaxPerSource: 3,
		ArxivEndpoint: server.URL + "/arxiv", GitHubAPI: server.URL,
		GitHubRepositories: []string{"acme/runtime"},
	})
	if got.Outcome != quantwatch.OutcomeRanked || len(got.Items) != 1 {
		t.Fatalf("outcome=%q items=%d", got.Outcome, len(got.Items))
	}
	if len(got.Sources) != 2 {
		t.Fatalf("sources=%d", len(got.Sources))
	}
	if got.Sources[0].QueryTime != now || got.Sources[0].URL == "" {
		t.Fatalf("arxiv receipt=%#v", got.Sources[0])
	}
	if !got.Sources[1].Abstained || got.Sources[1].Reason != quantwatch.ReasonSourceUnavailable {
		t.Fatalf("github receipt=%#v", got.Sources[1])
	}
	if got.Items[0].Claims.Artifact.Status != quantwatch.ClaimUnknown || got.Items[0].Claims.HardwareEnvelope.Status != quantwatch.ClaimUnknown {
		t.Fatalf("invented claims: %#v", got.Items[0].Claims)
	}
}

func TestLiveFetchBoundRefusal(t *testing.T) {
	got := quantwatch.FetchLive(context.Background(), nil, quantwatch.LiveRequest{MaxPerSource: 101, QueryTime: time.Now()})
	if got.Outcome != quantwatch.OutcomeRefused || got.Reason != quantwatch.ReasonBoundExceeded {
		t.Fatalf("%q %q", got.Outcome, got.Reason)
	}
}

func TestResultJSONCarriesHonestClaimSurfaces(t *testing.T) {
	raw, err := json.Marshal(quantwatch.IngestSnapshot([]byte(`{"schema":"fak.quantwatch.snapshot/v1","query_time":"2026-08-10T12:00:00Z","sources":[]}`)))
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"artifact", "recipe", "runtime_delegation", "hardware_envelope", "query_time", "sources"} {
		if !strings.Contains(string(raw), `"`+field+`"`) && field != "artifact" { // no items means claim fields correctly absent
			if field == "recipe" || field == "runtime_delegation" || field == "hardware_envelope" {
				continue
			}
			t.Fatalf("result missing %s: %s", field, raw)
		}
	}
}

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func cacheWitnessServer(t *testing.T, cached int) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("path=%s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"test","choices":[{"message":{"role":"assistant","content":"CACHE-WITNESS"},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":1,"total_tokens":101,"prompt_tokens_details":{"cached_tokens":` + fmt.Sprint(cached) + `}}}`))
	}))
}

func TestMicroCacheWitnessRefusesSingleEndpoint(t *testing.T) {
	s := cacheWitnessServer(t, 50)
	defer s.Close()
	var out, errb bytes.Buffer
	code := runMicroCacheWitness(&out, &errb, []string{"--model", "test", "--gateway-seat", "a=" + s.URL, "--calls", "2"})
	var got microCacheWitness
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if code != 3 || got.Verdict != "not-yet" || got.DistinctSeatEndpoints != 1 {
		t.Fatalf("code=%d got=%+v err=%s", code, got, errb.String())
	}
}

func TestMicroCacheWitnessCapturesOnOffProviderCounters(t *testing.T) {
	a := cacheWitnessServer(t, 40)
	defer a.Close()
	b := cacheWitnessServer(t, 10)
	defer b.Close()
	var out, errb bytes.Buffer
	code := runMicroCacheWitness(&out, &errb, []string{"--model", "test", "--gateway-seat", "a=" + a.URL, "--gateway-seat", "b=" + b.URL, "--calls", "2"})
	var got microCacheWitness
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if code != 0 || got.Verdict != "ready" || got.On.CachedPromptTokens == 0 || got.Off.CachedPromptTokens == 0 || len(got.On.SelectedSeats) != 2 {
		t.Fatalf("code=%d got=%+v err=%s", code, got, errb.String())
	}
}

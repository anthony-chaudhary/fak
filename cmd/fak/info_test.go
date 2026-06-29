package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestGroupThousands(t *testing.T) {
	cases := map[int64]string{0: "0", 12: "12", 123: "123", 1234: "1,234", 12345: "12,345", 1234567: "1,234,567"}
	for in, want := range cases {
		if got := groupThousands(in); got != want {
			t.Fatalf("groupThousands(%d) = %q, want %q", in, got, want)
		}
	}
}

func TestSignedTokens(t *testing.T) {
	if got := signedTokens(12345); got != "+12,345" {
		t.Fatalf("positive saving = %q, want +12,345", got)
	}
	// Negative is the load-bearing case: net saving is below zero until reads repay writes.
	if got := signedTokens(-1234); got != "-1,234" {
		t.Fatalf("negative saving = %q, want -1,234", got)
	}
}

func TestRenderGuardInfoLineNoCacheCleanFloor(t *testing.T) {
	var v guardInfoVars
	v.Inference.Turns = 4
	v.Gateway.InflightRequests = 0
	line := renderGuardInfoLine(v)
	if !strings.Contains(line, "cache —") {
		t.Fatalf("no-cache should read 'cache —', got: %s", line)
	}
	if !strings.Contains(line, "floor clean") {
		t.Fatalf("zero refusals should read 'floor clean', got: %s", line)
	}
	if !strings.Contains(line, "turns 4") {
		t.Fatalf("missing turns: %s", line)
	}
}

func TestRenderGuardInfoLineProvenCacheAndSafety(t *testing.T) {
	var v guardInfoVars
	v.Kernel.Denies = 1
	v.Kernel.Transforms = 2
	v.Kernel.Quarantines = 1
	v.Kernel.ResultDenies = 1 // folds into quarantined => 2
	v.VCache = &struct {
		CacheReadTokens int64   `json:"cache_read_tokens"`
		SavedTokenEquiv float64 `json:"saved_token_equiv"`
		HitRate         float64 `json:"hit_rate"`
		Multiplier      float64 `json:"multiplier"`
		Status          string  `json:"status"`
	}{CacheReadTokens: 1000, SavedTokenEquiv: 12345, HitRate: 0.88, Multiplier: 2.1, Status: "PROVEN"}

	line := renderGuardInfoLine(v)
	for _, want := range []string{"PROVEN", "×2.10", "saved +12,345 tok", "hit 88%", "blocked 1", "repaired 2", "quarantined 2"} {
		if !strings.Contains(line, want) {
			t.Fatalf("line missing %q:\n%s", want, line)
		}
	}
}

// debugVarsStub returns a gateway whose /debug/vars matches the guardInfoVars shape.
func debugVarsStub(t *testing.T) *httptest.Server {
	t.Helper()
	const body = `{
		"gateway":{"uptime_seconds":42,"inflight_requests":1,"vdso":true},
		"kernel":{"submits":3,"admitted":2,"denies":1,"transforms":0,"quarantines":1,"result_denies":0},
		"inference":{"turns":5},
		"vcache":{"cache_read_tokens":1000,"saved_token_equiv":12345,"hit_rate":0.88,"multiplier":2.1,"status":"PROVEN"}
	}`
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/debug/vars" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
}

func TestRunInfoOnceRendersLine(t *testing.T) {
	srv := debugVarsStub(t)
	defer srv.Close()

	var stdout, stderr bytes.Buffer
	code := runInfo(&stdout, &stderr, []string{"--gateway-url", srv.URL, "--once"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "fak info · "+srv.URL) {
		t.Fatalf("missing header:\n%s", out)
	}
	for _, want := range []string{"PROVEN", "×2.10", "blocked 1", "quarantined 1", "turns 5", "inflight 1", "up 42s"} {
		if !strings.Contains(out, want) {
			t.Fatalf("output missing %q:\n%s", want, out)
		}
	}
}

func TestRunInfoJSONSnapshot(t *testing.T) {
	srv := debugVarsStub(t)
	defer srv.Close()

	var stdout, stderr bytes.Buffer
	code := runInfo(&stdout, &stderr, []string{"--gateway-url", srv.URL, "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
	}
	var v guardInfoVars
	if err := json.Unmarshal(stdout.Bytes(), &v); err != nil {
		t.Fatalf("json output did not round-trip: %v\n%s", err, stdout.String())
	}
	if v.VCache == nil || v.VCache.Status != "PROVEN" {
		t.Fatalf("vcache not decoded: %+v", v.VCache)
	}
	if v.Kernel.Denies != 1 || v.Inference.Turns != 5 {
		t.Fatalf("kernel/inference not decoded: %+v / %+v", v.Kernel, v.Inference)
	}
}

func TestRunInfoRejectsBadInterval(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runInfo(&stdout, &stderr, []string{"--gateway-url", "http://127.0.0.1:1", "--interval", "0s"}); code != 2 {
		t.Fatalf("bad interval exit = %d, want 2", code)
	}
}

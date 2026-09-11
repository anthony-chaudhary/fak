package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func cacheAdminTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/fak/cacheprt/summary", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok","entries":123,"hit_rate":0.42}`))
	})
	mux.HandleFunc("/v1/fak/cacheprt/namespaces", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"namespaces":[{"namespace":"default","nodes":12,"bytes":4096}],"degraded":true,"degraded_reason":"per-namespace stats not wired"}`))
	})
	mux.HandleFunc("/v1/fak/cacheprt/coldcliff", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"cliff_tokens":4096,"risk":"low","rebuild_mult":1.5}`))
	})
	return httptest.NewServer(mux)
}

func TestCacheAdminStatusParsesSummary(t *testing.T) {
	srv := cacheAdminTestServer(t)
	defer srv.Close()
	var out, errb bytes.Buffer
	if code := runCacheStatus(&out, &errb, []string{"--addr", srv.URL}); code != 0 {
		t.Fatalf("exit = %d, stderr: %s", code, errb.String())
	}
	for _, want := range []string{"status", "ok", "entries", "123", "hit_rate", "0.42"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("stdout missing %q: %s", want, out.String())
		}
	}
	if errb.Len() != 0 {
		t.Errorf("unexpected stderr: %s", errb.String())
	}
}

func TestCacheAdminStatusJSONVerbatim(t *testing.T) {
	srv := cacheAdminTestServer(t)
	defer srv.Close()
	var out, errb bytes.Buffer
	if code := runCacheStatus(&out, &errb, []string{"--addr", srv.URL, "--json"}); code != 0 {
		t.Fatalf("exit = %d, stderr: %s", code, errb.String())
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(out.Bytes(), &obj); err != nil {
		t.Fatalf("stdout not JSON: %v\n%s", err, out.String())
	}
	for _, key := range []string{"status", "entries", "hit_rate"} {
		if _, ok := obj[key]; !ok {
			t.Errorf("JSON output missing key %q: %s", key, out.String())
		}
	}
}

func TestCacheAdminStatusCompactOneLine(t *testing.T) {
	srv := cacheAdminTestServer(t)
	defer srv.Close()
	var out, errb bytes.Buffer
	if code := runCacheStatus(&out, &errb, []string{"--addr", srv.URL, "--compact"}); code != 0 {
		t.Fatalf("exit = %d, stderr: %s", code, errb.String())
	}
	lines := strings.Split(strings.TrimRight(out.String(), "\n"), "\n")
	if len(lines) != 1 {
		t.Errorf("compact output = %d lines, want 1: %q", len(lines), out.String())
	}
	for _, want := range []string{"status=ok", "entries=123", "hit_rate=0.42"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("compact output missing %q: %s", want, out.String())
		}
	}
}

func TestCacheAdminNamespacesTable(t *testing.T) {
	srv := cacheAdminTestServer(t)
	defer srv.Close()
	var out, errb bytes.Buffer
	if code := runCacheNamespaces(&out, &errb, []string{"--addr", srv.URL}); code != 0 {
		t.Fatalf("exit = %d, stderr: %s", code, errb.String())
	}
	for _, want := range []string{"default", "nodes=12", "bytes=4096", "degraded: true", "per-namespace stats not wired"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("namespaces table missing %q: %s", want, out.String())
		}
	}
}

func TestCacheAdminNamespacesCompact(t *testing.T) {
	srv := cacheAdminTestServer(t)
	defer srv.Close()
	var out, errb bytes.Buffer
	if code := runCacheNamespaces(&out, &errb, []string{"--addr", srv.URL, "--compact"}); code != 0 {
		t.Fatalf("exit = %d, stderr: %s", code, errb.String())
	}
	lines := strings.Split(strings.TrimRight(out.String(), "\n"), "\n")
	if len(lines) != 1 || !strings.Contains(out.String(), "namespaces(1)") || !strings.Contains(out.String(), "degraded=true") {
		t.Errorf("compact namespaces = %q", out.String())
	}
}

func TestCacheAdminColdcliffParses(t *testing.T) {
	srv := cacheAdminTestServer(t)
	defer srv.Close()
	var out, errb bytes.Buffer
	if code := runCacheColdcliff(&out, &errb, []string{"--addr", srv.URL}); code != 0 {
		t.Fatalf("exit = %d, stderr: %s", code, errb.String())
	}
	for _, want := range []string{"cliff_tokens", "4096", "risk", "low"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("coldcliff output missing %q: %s", want, out.String())
		}
	}
}

func TestCacheAdminUnreachableExit1(t *testing.T) {
	// 127.0.0.1:1 is the house convention for a guaranteed-refused dial.
	unreach := "http://127.0.0.1:1"
	for _, fn := range []func(io.Writer, io.Writer, []string) int{
		runCacheStatus,
		runCacheNamespaces,
		runCacheColdcliff,
	} {
		var out, errb bytes.Buffer
		code := fn(&out, &errb, []string{"--addr", unreach})
		if code != 1 {
			t.Errorf("unreachable exit = %d, want 1 (stderr: %s)", code, errb.String())
		}
		if !strings.Contains(errb.String(), "gateway unreachable") {
			t.Errorf("stderr missing %q: %s", "gateway unreachable", errb.String())
		}
		if out.Len() != 0 {
			t.Errorf("unreachable run fabricated stdout: %s", out.String())
		}
	}
}

func TestCacheAdminGetRejectsNonJSONAndHTTPErrors(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/fak/cacheprt/summary", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte(`<html>proxy error</html>`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	if _, err := cacheAdminGet(srv.URL, "/v1/fak/cacheprt/summary"); err == nil {
		t.Fatal("expected error for HTTP 403")
	} else if !strings.Contains(err.Error(), "HTTP 403") {
		t.Errorf("error should name the HTTP status, got: %v", err)
	}

	mux2 := http.NewServeMux()
	mux2.HandleFunc("/v1/fak/cacheprt/summary", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`not json at all`))
	})
	srv2 := httptest.NewServer(mux2)
	defer srv2.Close()
	if _, err := cacheAdminGet(srv2.URL, "/v1/fak/cacheprt/summary"); err == nil {
		t.Fatal("expected error for non-JSON body")
	}
}

func TestCacheAdminUnknownSubcommand(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runCache(&out, &errb, []string{"frobnicate"}); code != 2 {
		t.Errorf("unknown subcommand exit = %d, want 2", code)
	}
	var out2, errb2 bytes.Buffer
	if code := runCache(&out2, &errb2, nil); code != 2 {
		t.Errorf("no-arg exit = %d, want 2", code)
	}
}

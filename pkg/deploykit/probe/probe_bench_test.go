package probe

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/deploykit/stamp"
)

// benchServe serves h on a local httptest server and closes it when the
// benchmark finishes. It is the *testing.B counterpart of serve.
func benchServe(b *testing.B, h http.Handler) string {
	b.Helper()
	srv := httptest.NewServer(h)
	b.Cleanup(srv.Close)
	return srv.URL
}

// mountBenchServer mounts the probe routes on a fresh mux and serves them on a
// local httptest server, returning the base URL.
func mountBenchServer(b *testing.B, h Hooks) string {
	b.Helper()
	mux := http.NewServeMux()
	if err := Mount(mux, h); err != nil {
		b.Fatalf("Mount: %v", err)
	}
	return benchServe(b, mux)
}

// BenchmarkMountLiveness measures a GET /healthz round trip through the
// liveness handler Mount registers on a fresh mux, served by a local
// httptest server.
func BenchmarkMountLiveness(b *testing.B) {
	base := mountBenchServer(b, Hooks{Ready: readyOK, Version: func() stamp.Identity { return identityA }})
	client := &http.Client{}
	status := func() (int, error) {
		resp, err := client.Get(base + PathHealth)
		if err != nil {
			return 0, err
		}
		defer resp.Body.Close()
		_, _ = io.Copy(io.Discard, resp.Body)
		return resp.StatusCode, nil
	}
	if got, err := status(); err != nil || got != http.StatusOK {
		b.Fatalf("fixture GET /healthz = %d, %v; want %d", got, err, http.StatusOK)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if got, err := status(); err != nil || got != http.StatusOK {
			b.Fatalf("GET /healthz = %d, %v; want %d", got, err, http.StatusOK)
		}
	}
}

// BenchmarkMountVersion measures a GET /version round trip through Mount's
// identity handler, including parsing the reply back with the stamp parser
// and comparing it to the expected build identity.
func BenchmarkMountVersion(b *testing.B) {
	base := mountBenchServer(b, Hooks{Ready: readyOK, Version: func() stamp.Identity { return identityA }})
	client := &http.Client{}
	fetch := func() (stamp.Identity, error) {
		resp, err := client.Get(base + PathVersion)
		if err != nil {
			return stamp.Identity{}, err
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return stamp.Identity{}, fmt.Errorf("GET /version = %d", resp.StatusCode)
		}
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return stamp.Identity{}, err
		}
		return stamp.ParseVersionJSON(body)
	}
	id, err := fetch()
	if err != nil || id != identityA {
		b.Fatalf("fixture GET /version = %+v, %v; want %+v", id, err, identityA)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		got, err := fetch()
		if err != nil || got != identityA {
			b.Fatalf("GET /version = %+v, %v; want %+v", got, err, identityA)
		}
	}
}

// BenchmarkMountDrain measures an authorized POST /v1/admin/drain round trip
// through Mount's drain handler: BearerAuth admission, the one-at-a-time
// lock, and the live-shape JSON reply.
func BenchmarkMountDrain(b *testing.B) {
	base := mountBenchServer(b, Hooks{
		Ready: readyOK,
		Drain: func(context.Context, time.Duration) DrainResponse {
			return DrainResponse{Status: DrainStatusDrained, ElapsedMS: 12}
		},
		AdminAuth: BearerAuth("admin-secret"),
	})
	client := &http.Client{}
	post := func() (*http.Response, error) {
		req, err := http.NewRequest(http.MethodPost, base+PathDrain+"?timeout=5s", nil)
		if err != nil {
			return nil, err
		}
		req.Header.Set("Authorization", "Bearer admin-secret")
		return client.Do(req)
	}
	resp, err := post()
	if err != nil {
		b.Fatalf("fixture POST /v1/admin/drain: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK || !strings.Contains(string(body), `"status":"drained"`) {
		b.Fatalf("fixture drain = %d %q, want 200 and a drained body", resp.StatusCode, body)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		resp, err := post()
		if err != nil {
			b.Fatalf("POST /v1/admin/drain: %v", err)
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			b.Fatalf("drain = %d, want %d", resp.StatusCode, http.StatusOK)
		}
	}
}

// BenchmarkClientProbe measures Client.Probe's four-check happy path
// (liveness, readiness, catalog, identity) against an in-process httptest
// server that is ready immediately and reports the expected commit, so no
// sample ever retries or sleeps.
func BenchmarkClientProbe(b *testing.B) {
	mux := http.NewServeMux()
	mux.HandleFunc(PathModels, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"object":"list","data":[{"id":"fak","object":"model"}]}`)
	})
	if err := Mount(mux, Hooks{Ready: readyOK, Version: func() stamp.Identity { return identityA }}); err != nil {
		b.Fatalf("Mount: %v", err)
	}
	base := benchServe(b, mux)
	c := fastClient()
	ctx := context.Background()
	target := Target{BaseURL: base}
	expect := Expect{Commit: commitA}
	res, err := c.Probe(ctx, target, expect)
	if err != nil || !res.Ready || res.Models != 1 || res.Identity == nil || *res.Identity != identityA {
		b.Fatalf("fixture Probe = %+v, %v; want ready with one model and identity %s", res, err, commitA)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := c.Probe(ctx, target, expect)
		if err != nil || !res.Ready {
			b.Fatalf("Probe = %+v, %v; want ready", res, err)
		}
	}
}

// BenchmarkDecodeDrainResponse measures DecodeDrainResponse over a realistic
// drained body carrying a 64-row observation snapshot, the shape fak-server's
// drain endpoint writes.
func BenchmarkDecodeDrainResponse(b *testing.B) {
	entry := `{"detail":"` + strings.Repeat("x", 300) + `"}`
	entries := make([]string, 64)
	for i := range entries {
		entries[i] = entry
	}
	body := `{"status":"drained","active_connections":0,"elapsed_ms":12,"observation_snapshot":[` + strings.Join(entries, ",") + `]}`
	resp, err := DecodeDrainResponse(http.StatusOK, strings.NewReader(body))
	if err != nil || !resp.Drained() || resp.ElapsedMS != 12 {
		b.Fatalf("fixture DecodeDrainResponse = %+v, %v; want a drained 12ms response", resp, err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		resp, err := DecodeDrainResponse(http.StatusOK, strings.NewReader(body))
		if err != nil || !resp.Drained() {
			b.Fatalf("DecodeDrainResponse = %+v, %v; want drained", resp, err)
		}
	}
}

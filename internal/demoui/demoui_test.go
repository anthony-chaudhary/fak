package demoui

import (
	"bytes"
	"flag"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// Probe must report a usable machine surface: at least one core, at least one matmul
// worker, at least the reference backend, and a non-empty human summary. This is the
// witness that a demo always has something honest to render about the hardware.
func TestProbeReportsMachine(t *testing.T) {
	hw := Probe()
	if hw.LogicalCores < 1 {
		t.Fatalf("logical cores = %d, want >= 1", hw.LogicalCores)
	}
	if hw.Workers < 1 {
		t.Fatalf("workers = %d, want >= 1", hw.Workers)
	}
	if len(hw.Backends) == 0 {
		t.Fatal("no compute backends registered (cpu-ref should always be present)")
	}
	if hw.Summary == "" {
		t.Fatal("empty hardware summary")
	}
	// On a default (untagged) build the only backend is the reference floor, so there
	// is no accelerator and the summary must say so rather than imply a GPU.
	if hw.Accelerator == "" && !strings.Contains(hw.Summary, "CPU") {
		t.Fatalf("CPU-only summary should mention CPU, got %q", hw.Summary)
	}
}

// Beat must tick repeatedly while work runs: a 350ms job with a 100ms cadence has to
// produce at least two heartbeats, which is the property the demos rely on to keep the
// screen alive (~1×/s) during a long blocking phase.
func TestBeatTicksDuringWork(t *testing.T) {
	var ticks int32
	Beat(80*time.Millisecond,
		func(time.Duration) { atomic.AddInt32(&ticks, 1) },
		func() { time.Sleep(320 * time.Millisecond) },
	)
	if got := atomic.LoadInt32(&ticks); got < 2 {
		t.Fatalf("ticks = %d, want >= 2 over a 320ms job at 80ms cadence", got)
	}
}

// Beat must not return until work is finished, even with a fast cadence — a caller
// that reads work's result right after Beat returns depends on this barrier.
func TestBeatWaitsForWork(t *testing.T) {
	var finished int32
	Beat(5*time.Millisecond,
		func(time.Duration) {},
		func() { time.Sleep(60 * time.Millisecond); atomic.StoreInt32(&finished, 1) },
	)
	if atomic.LoadInt32(&finished) != 1 {
		t.Fatal("Beat returned before work() finished")
	}
}

// A zero cadence opts out of ticking but still runs and waits on work.
func TestBeatZeroCadenceNoTicks(t *testing.T) {
	var ticks, finished int32
	Beat(0,
		func(time.Duration) { atomic.AddInt32(&ticks, 1) },
		func() { time.Sleep(20 * time.Millisecond); atomic.StoreInt32(&finished, 1) },
	)
	if atomic.LoadInt32(&ticks) != 0 {
		t.Fatalf("ticks = %d, want 0 at zero cadence", atomic.LoadInt32(&ticks))
	}
	if atomic.LoadInt32(&finished) != 1 {
		t.Fatal("Beat with zero cadence did not wait on work()")
	}
}

// Spinner must animate then leave a clean line (stop is idempotent).
func TestSpinnerAnimatesAndClears(t *testing.T) {
	var buf bytes.Buffer
	stop := Spinner(&buf, "Loading model")
	time.Sleep(300 * time.Millisecond)
	stop()
	stop() // idempotent — second call must not panic or write garbage
	out := buf.String()
	if !strings.Contains(out, "Loading model") {
		t.Fatalf("spinner output missing label: %q", out)
	}
	if !strings.Contains(out, "\r") {
		t.Fatal("spinner never rewrote its line (no carriage return)")
	}
}

func TestNormalizeBasePath(t *testing.T) {
	tests := []struct {
		raw  string
		want string
	}{
		{"", ""},
		{"/", ""},
		{"guarddemo", "/guarddemo"},
		{"/guarddemo/", "/guarddemo"},
		{"  /demo/nested/  ", "/demo/nested"},
		{"///", ""},
	}
	for _, tc := range tests {
		if got := NormalizeBasePath(tc.raw); got != tc.want {
			t.Errorf("NormalizeBasePath(%q) = %q, want %q", tc.raw, got, tc.want)
		}
	}
}

func TestBasePathFlagUsesSharedEnvDefaultAndHelp(t *testing.T) {
	t.Setenv(DemoBasePathEnv, "/from-env")
	fs := flag.NewFlagSet("demo", flag.ContinueOnError)
	got := BasePathFlag(fs, "/guarddemo")
	if *got != "/from-env" {
		t.Fatalf("BasePathFlag default = %q, want /from-env", *got)
	}
	f := fs.Lookup("base-path")
	if f == nil {
		t.Fatal("base-path flag was not registered")
	}
	if f.DefValue != "/from-env" {
		t.Fatalf("base-path DefValue = %q, want /from-env", f.DefValue)
	}
	if !strings.Contains(f.Usage, "/guarddemo") || !strings.Contains(f.Usage, DemoBasePathEnv) {
		t.Fatalf("base-path help %q should mention example path and %s", f.Usage, DemoBasePathEnv)
	}
}

func TestLocalURLUsesLoopbackForWildcardBinds(t *testing.T) {
	if got := LocalURL("0.0.0.0:8151", "/guarddemo"); got != "http://127.0.0.1:8151/guarddemo/" {
		t.Fatalf("LocalURL(wildcard) = %q, want loopback URL with base path", got)
	}
	if got := LocalURL("127.0.0.1:8151", ""); got != "http://127.0.0.1:8151/" {
		t.Fatalf("LocalURL(loopback) = %q, want loopback root URL", got)
	}
}

func TestMountWithBasePath(t *testing.T) {
	app := http.NewServeMux()
	app.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte("index"))
	})
	app.HandleFunc("/api/ping", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("pong"))
	})

	root := http.NewServeMux()
	if base := MountWithBasePath(root, "", app); base != "" {
		t.Fatalf("root base = %q, want empty", base)
	}
	rr := httptest.NewRecorder()
	root.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/api/ping", nil))
	if rr.Code != http.StatusOK || rr.Body.String() != "pong" {
		t.Fatalf("root /api/ping = status %d body %q, want 200 pong", rr.Code, rr.Body.String())
	}

	prefixed := http.NewServeMux()
	if base := MountWithBasePath(prefixed, "/guarddemo/", app); base != "/guarddemo" {
		t.Fatalf("prefixed base = %q, want /guarddemo", base)
	}
	rr = httptest.NewRecorder()
	prefixed.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/guarddemo/api/ping", nil))
	if rr.Code != http.StatusOK || rr.Body.String() != "pong" {
		t.Fatalf("prefixed /guarddemo/api/ping = status %d body %q, want 200 pong", rr.Code, rr.Body.String())
	}
	rr = httptest.NewRecorder()
	prefixed.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/guarddemo", nil))
	if rr.Code != http.StatusMovedPermanently || rr.Header().Get("Location") != "/guarddemo/" {
		t.Fatalf("/guarddemo redirect = status %d location %q, want 301 /guarddemo/", rr.Code, rr.Header().Get("Location"))
	}
	rr = httptest.NewRecorder()
	prefixed.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/", nil))
	if rr.Code != http.StatusFound || rr.Header().Get("Location") != "/guarddemo/" {
		t.Fatalf("root redirect = status %d location %q, want 302 /guarddemo/", rr.Code, rr.Header().Get("Location"))
	}
}

func TestListenAddrHonorsPortOnlyForDefaultAddr(t *testing.T) {
	t.Setenv("PORT", "18151")
	const def = "127.0.0.1:8151"
	if got := ListenAddr(def, def); got != "0.0.0.0:18151" {
		t.Fatalf("ListenAddr(default, default) = %q, want 0.0.0.0:18151", got)
	}
	if got := ListenAddr("127.0.0.1:9000", def); got != "127.0.0.1:9000" {
		t.Fatalf("ListenAddr(explicit, default) = %q, want explicit address", got)
	}
}

// TestBrowserDemoPagesUseRelativeAPIPaths pins the HTTPS/path-prefix deployment
// contract for the browser demos. The pages may be served under /guarddemo/,
// /turntax/, etc. behind one HTTPS reverse proxy, so browser API calls must stay
// relative to the page path instead of jumping back to the domain root.
func TestBrowserDemoPagesUseRelativeAPIPaths(t *testing.T) {
	pages := []string{
		filepath.Join("..", "..", "cmd", "guarddemo", "page.html"),
		filepath.Join("..", "..", "cmd", "turntaxdemo", "page.html"),
		filepath.Join("..", "..", "cmd", "ctxdemo", "page.html"),
		filepath.Join("..", "..", "cmd", "demorace", "page.html"),
		filepath.Join("..", "..", "cmd", "unseedemo", "page.html"),
	}
	bad := []string{
		`fetch("/api`, `fetch('/api`, "fetch(`/api",
		`EventSource("/api`, `EventSource('/api`, "EventSource(`/api",
	}
	for _, page := range pages {
		b, err := os.ReadFile(page)
		if err != nil {
			t.Fatalf("read %s: %v", page, err)
		}
		s := string(b)
		for _, needle := range bad {
			if strings.Contains(s, needle) {
				t.Errorf("%s contains %q; use a relative api/... URL so reverse-proxy path prefixes work", page, needle)
			}
		}
	}
}

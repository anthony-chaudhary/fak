package main

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"strings"
	"testing"
)

func TestClaudeMacFakDryRunDefaultsToInteractive(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "super-secret-test-key")
	t.Setenv("API_TIMEOUT_MS", "")
	dir := t.TempDir()

	var stdout, stderr bytes.Buffer
	code := runClaudeMacFak(&stdout, &stderr, []string{
		"--dry-run",
		"--claude-config-dir", dir,
		"--gateway-url", "http://node.example:8080/v1",
		"--model", "qwen-local",
	})
	if code != 0 {
		t.Fatalf("runClaudeMacFak code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{
		"provider=existing-fak-gateway",
		"gateway=http://node.example:8080",
		"<redacted from FAK_GATEWAY_KEY>",
		"API_TIMEOUT_MS",
		"1800000",
		"Launch\n  claude",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("dry-run missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, " -p ") || strings.Contains(out, "--output-format json") {
		t.Fatalf("default dry-run should be interactive, not a probe:\n%s", out)
	}
	if strings.Contains(out, "super-secret-test-key") {
		t.Fatalf("dry-run leaked bearer:\n%s", out)
	}
}

func TestClaudeMacFakProbeAddsPromptAndJSONOutput(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "super-secret-test-key")
	dir := t.TempDir()

	var stdout, stderr bytes.Buffer
	code := runClaudeMacFak(&stdout, &stderr, []string{
		"--dry-run",
		"--probe",
		"--claude-config-dir", dir,
		"--gateway-url", "http://node.example:8080",
		"--model", "qwen-local",
	})
	if code != 0 {
		t.Fatalf("runClaudeMacFak code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"claude", "-p", "Reply with exactly: OK", "--output-format json"} {
		if !strings.Contains(out, want) {
			t.Fatalf("probe dry-run missing %q:\n%s", want, out)
		}
	}
}

func TestClaudeMacFakProbeInteractiveConflict(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "super-secret-test-key")
	dir := t.TempDir()

	var stdout, stderr bytes.Buffer
	code := runClaudeMacFak(&stdout, &stderr, []string{
		"--dry-run",
		"--probe",
		"--interactive",
		"--claude-config-dir", dir,
		"--gateway-url", "http://node.example:8080",
		"--model", "qwen-local",
	})
	if code != 2 {
		t.Fatalf("runClaudeMacFak code=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "either --probe or --interactive") {
		t.Fatalf("stderr = %q", stderr.String())
	}
}

func TestClaudeMacFakRequiresKeyWhenFetchDisabled(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "")
	dir := t.TempDir()

	var stdout, stderr bytes.Buffer
	code := runClaudeMacFak(&stdout, &stderr, []string{
		"--dry-run",
		"--fetch-key=false",
		"--claude-config-dir", dir,
		"--gateway-url", "http://node.example:8080",
		"--model", "qwen-local",
	})
	if code != 2 {
		t.Fatalf("runClaudeMacFak code=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "FAK_GATEWAY_KEY is empty") {
		t.Fatalf("stderr = %q", stderr.String())
	}
}

// TestEnsureClaudeMacGatewayKeySurfacesSSHStderr is the regression for the
// opaque "fetch gateway key over ssh: exit status 255" report: ssh's exit
// status is meaningless on its own, so the wrapped error must carry the
// stderr text (the real cause) plus the override hint. The ssh invocation is
// replaced by a helper process that mimics ssh: prints a resolve error to
// stderr and exits 255.
func TestEnsureClaudeMacGatewayKeySurfacesSSHStderr(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "")
	orig := execCommand
	t.Cleanup(func() { execCommand = orig })
	execCommand = func(name string, args ...string) *exec.Cmd {
		cs := append([]string{"-test.run=TestClaudeMacFakSSHHelperProcess", "--", name}, args...)
		cmd := exec.Command(os.Args[0], cs...)
		cmd.Env = append(os.Environ(), "GO_WANT_SSH_HELPER_PROCESS=1")
		return cmd
	}

	err := ensureClaudeMacGatewayKey("FAK_GATEWAY_KEY", true, "user@node-macos-a.local", "")
	if err == nil {
		t.Fatal("expected an error when the ssh fetch fails")
	}
	msg := err.Error()
	for _, want := range []string{
		"node-macos-a.local",           // which host failed
		"Could not resolve hostname",   // the real cause from stderr, not a bare 255
		"set FAK_GATEWAY_KEY directly", // the actionable override hint
	} {
		if !strings.Contains(msg, want) {
			t.Fatalf("error missing %q:\n%s", want, msg)
		}
	}
	if strings.Contains(msg, "exit status 255") && !strings.Contains(msg, "Could not resolve hostname") {
		t.Fatalf("error fell back to the opaque exit status:\n%s", msg)
	}
}

// TestClaudeMacFakSSHHelperProcess is not a real test: it is the fake `ssh`
// the test above execs. It writes a resolve error to stderr and exits 255,
// reproducing the connection-level failure shape of the original bug report.
func TestClaudeMacFakSSHHelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_SSH_HELPER_PROCESS") != "1" {
		return
	}
	_, _ = os.Stderr.WriteString("ssh: Could not resolve hostname node-macos-a.local: Name or service not known\n")
	os.Exit(255)
}

func TestClaudeMacFakDryRunDoesNotProbeDebugGateway(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "super-secret-test-key")
	dir := t.TempDir()

	var stdout, stderr bytes.Buffer
	code := runClaudeMacFak(&stdout, &stderr, []string{
		"--dry-run",
		"--claude-config-dir", dir,
		"--gateway-url", "http://127.0.0.1:1",
		"--model", "qwen-local",
	})
	if code != 0 {
		t.Fatalf("runClaudeMacFak code=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	if strings.Contains(stderr.String(), "gateway unreachable") {
		t.Fatalf("dry-run should not probe the gateway: %s", stderr.String())
	}
}

func TestClaudeMacFakRejectsNonPositiveOverlayInterval(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runClaudeMacFak(&stdout, &stderr, []string{
		"--overlay",
		"--overlay-interval", "0s",
	})
	if code != 2 {
		t.Fatalf("runClaudeMacFak code=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "--overlay-interval must be positive") {
		t.Fatalf("stderr = %q", stderr.String())
	}
}

func TestClaudeMacDebugClientProbeUsesBearer(t *testing.T) {
	var sawHealth, sawVars bool
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer super-secret-test-key" {
			t.Fatalf("Authorization = %q", got)
		}
		switch r.URL.Path {
		case "/healthz":
			sawHealth = true
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"ok":true,"engine":"fak","model":"qwen-local","planner":"inkernel"}`))
		case "/debug/vars":
			sawVars = true
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"gateway":{"up":true,"vdso":true},"kernel":{"submits":9,"vdso_hits":3,"engine_calls":6,"vdso_hit_ratio":0.3333333333},"runtime":{"num_goroutine":7,"memory":{"heap_alloc_bytes":2048}}}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()

	c := &claudeMacDebugClient{base: ts.URL, key: "super-secret-test-key", hc: ts.Client()}
	h, v, err := c.probe()
	if err != nil {
		t.Fatalf("probe: %v", err)
	}
	if !sawHealth || !sawVars {
		t.Fatalf("probe did not call both endpoints: health=%v vars=%v", sawHealth, sawVars)
	}
	if h.Planner != "inkernel" || v.Kernel.Submits != 9 || v.Kernel.VDSOHits != 3 {
		t.Fatalf("unexpected probe data: health=%+v vars=%+v", h, v.Kernel)
	}
}

func TestRenderClaudeMacPreflightWarnsOnMockWithoutBearerLeak(t *testing.T) {
	var v claudeMacDebugVars
	v.Gateway.VDSO = true
	v.Gateway.UptimeSeconds = 3725
	v.Gateway.InflightRequests = 2
	v.Kernel.VDSOHitRatio = 0.875

	out := renderClaudeMacPreflight(
		claudeMacHealth{OK: true, Engine: "fak", Model: "qwen-local", Planner: "mock"},
		v,
		"http://node.example:8080",
		"qwen-local",
		"gateway-bearer",
		"http://grafana.example",
	)
	for _, want := range []string{
		"fak debug",
		"planner(live)=mock",
		"vdso=on",
		"cache-hit 0.88",
		"inflight 2",
		"auth gateway-bearer",
		"grafana http://grafana.example",
		"WARN: planner=mock",
		"-> launching claude ...",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("preflight missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "super-secret-test-key") || strings.Contains(out, "Bearer ") {
		t.Fatalf("preflight leaked bearer material:\n%s", out)
	}
}

func TestRenderClaudeMacOverlayLine(t *testing.T) {
	var v claudeMacDebugVars
	v.Kernel.Submits = 10
	v.Kernel.VDSOHits = 4
	v.Kernel.EngineCalls = 6
	v.Gateway.InflightRequests = 2
	v.Runtime.Memory.HeapAllocBytes = 2048
	v.Runtime.NumGoroutine = 7

	out := renderClaudeMacOverlayLine(v)
	for _, want := range []string{
		"submits 10",
		"hits 4 (40.0%)",
		"engine 6",
		"inflight 2",
		"gor 7",
		// Throughput leads the line now; with no measured turns the rates read "-".
		"prefill -",
		"decode -",
		"turns 0",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("overlay line missing %q: %s", want, out)
		}
	}
}

// TestRenderClaudeMacOverlayLineShowsThroughput proves the overlay surfaces real
// model-generation throughput even when the kernel counters are 0 (the exact
// proxy/chat case the user hit: submits 0 / engine 0 while the box decodes tokens).
func TestRenderClaudeMacOverlayLineShowsThroughput(t *testing.T) {
	var v claudeMacDebugVars
	v.Gateway.InflightRequests = 1
	v.Inference.Turns = 3
	v.Inference.PrefillTokensPerSecond = 250
	v.Inference.DecodeTokensPerSecond = 200
	v.Inference.InflightMaxAgeSeconds = 42

	out := renderClaudeMacOverlayLine(v)
	for _, want := range []string{
		"prefill 250 tok/s",
		"decode 200 tok/s",
		"turns 3",
		"oldest 42s",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("throughput overlay line missing %q: %s", want, out)
		}
	}
}

// TestPanelLegendExpandsAcronyms proves the preflight panel carries a legend that
// expands every acronym/term it (and the overlay) print — so an operator never has to
// leave the terminal to decode vDSO/TTFT/prefill/decode/engine/planner. A regression
// that drops a term re-opens the "what does this mean" confusion.
func TestPanelLegendExpandsAcronyms(t *testing.T) {
	var v claudeMacDebugVars
	v.Gateway.VDSO = true
	out := renderClaudeMacPreflight(
		claudeMacHealth{OK: true, Engine: "inkernel", Model: "qwen-local", Planner: "proxy"},
		v, "http://node.example:8080", "qwen-local", "gateway-bearer", "",
	)
	for _, want := range []string{
		"legend:",
		"engine(build) =",
		"planner(live) =",
		"vDSO =",
		"prefill =",
		"decode =",
		"TTFT = time-to-first-token",
		"tok/s = tokens per second",
		"inflight =",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("panel legend missing %q:\n%s", want, out)
		}
	}
}

// TestOverlayLegendCoversOverlayOnlyFields proves the overlay header legend expands the
// terms unique to the overlay line (turns/submits/hits/engine/heap/gor) on top of the
// shared panel legend, and explicitly tells the operator the kernel counters reading 0
// on a proxy/chat workload is expected — the exact confusion the user hit.
func TestOverlayLegendCoversOverlayOnlyFields(t *testing.T) {
	out := claudeMacOverlayLegend()
	for _, want := range []string{
		// shared panel terms are included...
		"vDSO =",
		"TTFT = time-to-first-token",
		// ...plus the overlay-only fields.
		"submits = kernel adjudications",
		"hits = vDSO fast-path hits",
		"engine = submits that reached the model",
		"heap = Go heap in use",
		"gor = live goroutines",
		"stay 0 on a proxy/chat workload — that is expected",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("overlay legend missing %q:\n%s", want, out)
		}
	}
}

// TestRenderClaudeMacPreflightProxyClarity proves the panel disambiguates the
// engine(build)/planner(live) labels and annotates a 0.00 cache-hit on a proxy planner
// as expected rather than a fault — the confusion the user flagged.
func TestRenderClaudeMacPreflightProxyClarity(t *testing.T) {
	var v claudeMacDebugVars
	v.Gateway.VDSO = true
	v.Gateway.InflightRequests = 1
	v.Kernel.VDSOHitRatio = 0.0
	v.Inference.InflightMaxAgeSeconds = 45 // a slow first request

	out := renderClaudeMacPreflight(
		claudeMacHealth{OK: true, Engine: "inkernel", Model: "qwen-local", Planner: "proxy"},
		v, "http://node.example:8080", "qwen-local", "gateway-bearer", "",
	)
	for _, want := range []string{
		"engine(build)=inkernel",
		"planner(live)=proxy",
		"proxy: kernel fast-path not exercised",
		"SLOW: cold upstream load or a wedged request",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("proxy-clarity preflight missing %q:\n%s", want, out)
		}
	}
}

// TestClaudeMacFakInteractiveEmitsDebugPanel proves --interactive now carries real,
// asserted behavior (not a dead flag): against a live gateway the preflight debug
// panel is printed BEFORE the launch. The launch itself targets a non-existent
// command so exec fails fast; the panel is emitted regardless, which is what we
// assert. The bearer must never appear in the panel.
func TestClaudeMacFakInteractiveEmitsDebugPanel(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "super-secret-test-key")
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/healthz":
			_, _ = w.Write([]byte(`{"ok":true,"engine":"metal","model":"qwen-local","planner":"inkernel"}`))
		case "/debug/vars":
			_, _ = w.Write([]byte(`{"gateway":{"up":true,"vdso":true,"uptime_seconds":11520,"inflight_requests":1},"kernel":{"submits":1240,"vdso_hits":1101,"engine_calls":139,"vdso_hit_ratio":0.888},"runtime":{"num_goroutine":47,"memory":{"heap_alloc_bytes":432013312}}}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()
	dir := t.TempDir()

	var stdout, stderr bytes.Buffer
	runClaudeMacFak(&stdout, &stderr, []string{
		"--interactive",
		"--claude-config-dir", dir,
		"--gateway-url", ts.URL,
		"--model", "qwen-local",
		"--command", "fak-no-such-claude-binary-xyz",
	})
	out := stdout.String()
	for _, want := range []string{"fak debug", "planner(live)=inkernel", "cache-hit 0.89", "-> launching claude ..."} {
		if !strings.Contains(out, want) {
			t.Fatalf("interactive --debug should emit the preflight panel; missing %q\nstdout=%s\nstderr=%s", want, out, stderr.String())
		}
	}
	if strings.Contains(out, "super-secret-test-key") {
		t.Fatalf("preflight leaked the bearer:\n%s", out)
	}
}

// TestClaudeMacFakInteractiveAbortsOnUnreachableGateway is the "better info"
// guarantee: an interactive launch whose gateway is unreachable returns 1 and
// never reaches the launch, instead of starting Claude against a dead backend.
func TestClaudeMacFakInteractiveAbortsOnUnreachableGateway(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "super-secret-test-key")
	dir := t.TempDir()

	var stdout, stderr bytes.Buffer
	code := runClaudeMacFak(&stdout, &stderr, []string{
		"--interactive",
		"--claude-config-dir", dir,
		// 127.0.0.1:1 is a closed reserved port: connection refused, fast.
		"--gateway-url", "http://127.0.0.1:1",
		"--model", "qwen-local",
		"--command", "fak-no-such-claude-binary-xyz",
	})
	if code != 1 {
		t.Fatalf("unreachable interactive launch must abort with code 1, got %d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "gateway unreachable") {
		t.Fatalf("expected a gateway-unreachable error, stderr=%q", stderr.String())
	}
	if strings.Contains(stdout.String(), "launching claude") {
		t.Fatalf("must not launch claude after an unreachable-gateway abort:\n%s", stdout.String())
	}
}

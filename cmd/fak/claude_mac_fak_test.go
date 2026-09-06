package main

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/appversion"
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
		"--width", "1000",
	})
	if code != 0 {
		t.Fatalf("runClaudeMacFak code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{
		"claude",
		"-p",
		"Reply with exactly: OK",
		"--output-format json",
		"--safe-mode",
		`--tools ""`,
		"--disable-slash-commands",
		"--no-session-persistence",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("probe dry-run missing %q:\n%s", want, out)
		}
	}
	wantCommand := `claude --permission-mode bypassPermissions -p "Reply with exactly: OK" --output-format json --safe-mode --tools "" --disable-slash-commands --no-session-persistence`
	if !strings.Contains(out, wantCommand) {
		t.Fatalf("probe dry-run command has wrong prompt/output order; want %q in:\n%s", wantCommand, out)
	}
}

func TestClaudeMacFakProbePrependsPromptBeforePassthrough(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "super-secret-test-key")
	dir := t.TempDir()

	var stdout, stderr bytes.Buffer
	code := runClaudeMacFak(&stdout, &stderr, []string{
		"--dry-run",
		"--probe",
		"--claude-config-dir", dir,
		"--gateway-url", "http://node.example:8080",
		"--model", "qwen-local",
		"--width", "1000",
		"--",
		"--output-format", "stream-json",
	})
	if code != 0 {
		t.Fatalf("runClaudeMacFak code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	wantCommand := `claude --permission-mode bypassPermissions -p "Reply with exactly: OK" --output-format stream-json --safe-mode --tools "" --disable-slash-commands --no-session-persistence`
	if !strings.Contains(out, wantCommand) {
		t.Fatalf("probe passthrough command has wrong prompt/output order; want %q in:\n%s", wantCommand, out)
	}
	if strings.Contains(out, "--output-format stream-json --output-format json") {
		t.Fatalf("probe passthrough should not append the default JSON output flag:\n%s", out)
	}
}

func TestClaudeMacFakProbeKeepsExplicitIsolationOverrides(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "super-secret-test-key")
	dir := t.TempDir()

	var stdout, stderr bytes.Buffer
	code := runClaudeMacFak(&stdout, &stderr, []string{
		"--dry-run",
		"--probe",
		"--claude-config-dir", dir,
		"--gateway-url", "http://node.example:8080",
		"--model", "qwen-local",
		"--width", "1000",
		"--",
		"--output-format", "json",
		"--safe-mode",
		"--tools", "Read",
		"--disable-slash-commands",
		"--no-session-persistence",
	})
	if code != 0 {
		t.Fatalf("runClaudeMacFak code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, `claude --permission-mode bypassPermissions -p "Reply with exactly: OK" --output-format json --safe-mode --tools Read --disable-slash-commands --no-session-persistence`) {
		t.Fatalf("probe passthrough command did not preserve explicit isolation flags:\n%s", out)
	}
	if strings.Contains(out, `--tools Read --tools ""`) || strings.Count(out, "--tools") != 2 {
		t.Fatalf("probe passthrough duplicated --tools or added the empty default:\n%s", out)
	}
	if strings.Count(out, "--safe-mode") != 2 || strings.Count(out, "--disable-slash-commands") != 2 || strings.Count(out, "--no-session-persistence") != 2 {
		t.Fatalf("probe passthrough duplicated boolean isolation defaults:\n%s", out)
	}
}

func TestClaudeMacFakProbeAddsJSONOutputWithPassthrough(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "super-secret-test-key")
	dir := t.TempDir()

	var stdout, stderr bytes.Buffer
	code := runClaudeMacFak(&stdout, &stderr, []string{
		"--dry-run",
		"--probe",
		"--claude-config-dir", dir,
		"--gateway-url", "http://node.example:8080",
		"--model", "qwen-local",
		"--width", "1000",
		"--",
		"--max-turns", "1",
	})
	if code != 0 {
		t.Fatalf("runClaudeMacFak code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	wantCommand := `claude --permission-mode bypassPermissions -p "Reply with exactly: OK" --max-turns 1 --output-format json --safe-mode --tools "" --disable-slash-commands --no-session-persistence`
	if !strings.Contains(out, wantCommand) {
		t.Fatalf("probe passthrough command did not add default JSON output; want %q in:\n%s", wantCommand, out)
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

	// A remote (non-loopback) gateway: the ssh fetch must run and fail loudly.
	err := ensureClaudeMacGatewayKey("FAK_GATEWAY_KEY", true, "user@node-macos-a.local", "", "http://node-macos-a.local:8080")
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

func TestEnsureClaudeMacGatewayKeyUsesBoundedNonInteractiveSSH(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "")
	orig := execCommand
	t.Cleanup(func() { execCommand = orig })
	var sawArgs []string
	execCommand = func(name string, args ...string) *exec.Cmd {
		if name != "ssh" {
			t.Fatalf("execCommand name = %q, want ssh", name)
		}
		sawArgs = append([]string(nil), args...)
		cs := append([]string{"-test.run=TestClaudeMacFakSSHHelperProcess", "--", name}, args...)
		cmd := exec.Command(os.Args[0], cs...)
		cmd.Env = append(os.Environ(), "GO_WANT_SSH_HELPER_PROCESS=1")
		return cmd
	}

	_ = ensureClaudeMacGatewayKey("FAK_GATEWAY_KEY", true, "user@node-macos-a.local", "", "http://node-macos-a.local:8080")
	got := strings.Join(sawArgs, " ")
	for _, want := range []string{
		"-o BatchMode=yes",
		"-o ConnectTimeout=5",
		"-o ConnectionAttempts=1",
		"user@node-macos-a.local cat ~/.fak-gateway-key",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("ssh args missing %q:\n%v", want, sawArgs)
		}
	}
}

// TestEnsureClaudeMacGatewayKeyLocalSkipsSSH is the easy-local-default
// guarantee: when the gateway is loopback there is no Mac to ssh into and a
// local fak serve without --require-key-env needs no bearer, so the ssh fetch
// must be skipped entirely and an empty key tolerated — no error, no exec.
func TestEnsureClaudeMacGatewayKeyLocalSkipsSSH(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "")
	orig := execCommand
	t.Cleanup(func() { execCommand = orig })
	execCommand = func(name string, args ...string) *exec.Cmd {
		t.Fatalf("local gateway must not invoke ssh; got %s %v", name, args)
		return nil
	}

	for _, gw := range []string{
		"http://127.0.0.1:8080",
		"http://localhost:8080",
		"http://[::1]:8080",
		"http://127.0.0.1:8080/v1",
	} {
		if err := ensureClaudeMacGatewayKey("FAK_GATEWAY_KEY", true, "user@node-macos-a.local", "", gw); err != nil {
			t.Fatalf("local gateway %q should skip the ssh fetch, got %v", gw, err)
		}
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

// TestClaudeMacFakRefusesTheUnconfiguredPlaceholderGateway is the regression for
// the first-run experience of the README's Mac showcase (#5457): a reader who
// runs `fak mac` having configured nothing used to wait out an ssh connect
// timeout and then get a resolve error naming `node-macos-a.local` — a host from
// someone else's lab. Nothing in that told them the host was a placeholder, or
// that `fak mac` needs a gateway to already exist. Refuse UP FRONT instead, name
// the placeholder as a placeholder, and hand over the setup path.
func TestClaudeMacFakRefusesTheUnconfiguredPlaceholderGateway(t *testing.T) {
	t.Setenv("FAK_MAC_GATEWAY", "")
	t.Setenv("FAK_GATEWAY_KEY", "")
	orig := execCommand
	t.Cleanup(func() { execCommand = orig })
	execCommand = func(name string, args ...string) *exec.Cmd {
		t.Fatalf("unconfigured gateway must refuse before any ssh fetch; got %s %v", name, args)
		return nil
	}

	var stdout, stderr bytes.Buffer
	code := runClaudeMacFak(&stdout, &stderr, []string{
		"--dry-run",
		"--claude-config-dir", t.TempDir(),
	})
	if code != 2 {
		t.Fatalf("runClaudeMacFak code=%d, want 2 (usage refusal)\nstdout=%s\nstderr=%s", code, stdout.String(), stderr.String())
	}
	msg := stderr.String()
	for _, want := range []string{
		"no Mac gateway configured",       // the actual problem, in the first line
		"placeholder",                     // the host is ours, not theirs
		defaultClaudeMacGateway,           // ...and which value is the placeholder
		"does not create one",             // fak mac is a launcher, not an installer
		"docs/fak/server-quickstart.md",   // where to stand the gateway up
		`export FAK_MAC_GATEWAY="http://`, // the exact next command
		"docs/fak/mac-agent-ui.md",        // the operator walkthrough, once it works
	} {
		if !strings.Contains(msg, want) {
			t.Fatalf("refusal missing %q:\n%s", want, msg)
		}
	}
}

// TestClaudeMacFakAcceptsAConfiguredGateway is the other half of the gate: the
// refusal keys on the untouched PLACEHOLDER, never on "a gateway I cannot reach".
// Supplying any gateway — including one that is merely down — must get past it,
// or the check would break every real operator instead of helping a newcomer.
func TestClaudeMacFakAcceptsAConfiguredGateway(t *testing.T) {
	t.Setenv("FAK_MAC_GATEWAY", "http://mac.example:8080")
	t.Setenv("FAK_GATEWAY_KEY", "super-secret-test-key")

	var stdout, stderr bytes.Buffer
	code := runClaudeMacFak(&stdout, &stderr, []string{
		"--dry-run",
		"--claude-config-dir", t.TempDir(),
		"--model", "qwen-local",
	})
	if code != 0 {
		t.Fatalf("configured gateway should not trip the placeholder gate: code=%d stderr=%s", code, stderr.String())
	}
	if strings.Contains(stderr.String(), "no Mac gateway configured") {
		t.Fatalf("placeholder refusal fired on a configured gateway:\n%s", stderr.String())
	}
}

// TestEnsureClaudeMacGatewayKeyNamesThePlaceholderSSHHost covers the second step
// of the same trap (#5457): the reader sets FAK_MAC_GATEWAY, gets past the gate
// above, and then fails on the OTHER placeholder — the ssh host. The ssh stderr
// alone still names a host they never chose, so the error must say which knob is
// still unset rather than leaving them to guess.
func TestEnsureClaudeMacGatewayKeyNamesThePlaceholderSSHHost(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "")
	orig := execCommand
	t.Cleanup(func() { execCommand = orig })
	execCommand = func(name string, args ...string) *exec.Cmd {
		cs := append([]string{"-test.run=TestClaudeMacFakSSHHelperProcess", "--", name}, args...)
		cmd := exec.Command(os.Args[0], cs...)
		cmd.Env = append(os.Environ(), "GO_WANT_SSH_HELPER_PROCESS=1")
		return cmd
	}

	// Gateway configured (a real-looking remote), ssh host still the placeholder.
	err := ensureClaudeMacGatewayKey("FAK_GATEWAY_KEY", true, defaultClaudeMacSSHHost, "", "http://mac.example:8080")
	if err == nil {
		t.Fatal("expected an error when the ssh fetch fails against the placeholder host")
	}
	msg := err.Error()
	for _, want := range []string{
		"Could not resolve hostname",         // still carries the real ssh cause
		"PLACEHOLDER ssh host",               // ...plus which knob is unset
		"FAK_MAC_SSH_HOST=<user>@<your-mac>", // the exact fix
		"set FAK_GATEWAY_KEY directly",       // the skip-the-fetch escape
	} {
		if !strings.Contains(msg, want) {
			t.Fatalf("error missing %q:\n%s", want, msg)
		}
	}
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

// TestRunClaudeMacMetricsFetchesBothWithBearer proves the --metrics one-shot:
// it sends the bearer to both surfaces, prints /debug/vars (JSON, indented) and
// /metrics (verbatim text), and never echoes the key.
func TestRunClaudeMacMetricsFetchesBothWithBearer(t *testing.T) {
	var sawMetrics, sawVars bool
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer super-secret-test-key" {
			http.Error(w, "missing or invalid credentials", http.StatusUnauthorized)
			return
		}
		switch r.URL.Path {
		case "/debug/vars":
			sawVars = true
			_, _ = w.Write([]byte(`{"gateway":{"up":true,"vdso":true}}`))
		case "/metrics":
			sawMetrics = true
			w.Header().Set("Content-Type", "text/plain")
			_, _ = w.Write([]byte("fak_gateway_up 1\n"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()

	c := &claudeMacDebugClient{base: ts.URL, key: "super-secret-test-key", hc: ts.Client()}
	var stdout, stderr bytes.Buffer
	if code := runClaudeMacMetrics(&stdout, &stderr, c); code != 0 {
		t.Fatalf("runClaudeMacMetrics code=%d stderr=%s", code, stderr.String())
	}
	if !sawVars || !sawMetrics {
		t.Fatalf("did not fetch both endpoints: vars=%v metrics=%v", sawVars, sawMetrics)
	}
	out := stdout.String()
	for _, want := range []string{"== /debug/vars ==", `"up": true`, "== /metrics ==", "fak_gateway_up 1"} {
		if !strings.Contains(out, want) {
			t.Fatalf("--metrics output missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "super-secret-test-key") || strings.Contains(stderr.String(), "super-secret-test-key") {
		t.Fatalf("--metrics leaked the bearer:\nstdout=%s\nstderr=%s", out, stderr.String())
	}
}

// TestRunClaudeMacMetricsHintsOnUnauthorized proves a 401 surfaces an actionable
// hint (set the key / run on the gateway host) rather than a bare status line.
func TestRunClaudeMacMetricsHintsOnUnauthorized(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "missing or invalid credentials", http.StatusUnauthorized)
	}))
	defer ts.Close()

	c := &claudeMacDebugClient{base: ts.URL, key: "", hc: ts.Client()}
	var stdout, stderr bytes.Buffer
	if code := runClaudeMacMetrics(&stdout, &stderr, c); code != 1 {
		t.Fatalf("runClaudeMacMetrics on 401 code=%d, want 1", code)
	}
	if !strings.Contains(stderr.String(), "FAK_GATEWAY_KEY") || !strings.Contains(stderr.String(), "loopback-exempt") {
		t.Fatalf("401 did not surface an actionable hint:\n%s", stderr.String())
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
		"== fak · Claude Code -> your own Mac's local model ==",
		"fak debug " + appversion.Current(),
		"planner(live)=mock",
		"vdso=on",
		"cache-hit 0.88",
		"inflight 2",
		"auth gateway-bearer",
		"fak claude-mac-fak --metrics",
		"off-box needs the bearer",
		"grafana (local stack): http://grafana.example",
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

func TestRenderClaudeMacPreflightShowsProviderExtraBodyKeys(t *testing.T) {
	var v claudeMacDebugVars
	v.Upstream.ProviderExtraBodySet = true
	v.Upstream.ProviderExtraBodyKeys = []string{"chat_template_kwargs", "top_k"}

	out := renderClaudeMacPreflight(
		claudeMacHealth{OK: true, Engine: "fak", Model: "lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M", Planner: "proxy"},
		v,
		"http://node.example:8080",
		"",
		"gateway-bearer",
		"",
	)
	for _, want := range []string{
		"request tuning: provider extra body set",
		"keys: chat_template_kwargs, top_k",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("preflight missing %q:\n%s", want, out)
		}
	}
	for _, forbidden := range []string{"preserve_thinking", `"top_k":20`} {
		if strings.Contains(out, forbidden) {
			t.Fatalf("preflight leaked provider extra body value %q:\n%s", forbidden, out)
		}
	}
}

func TestRenderClaudeMacPreflightWarnsQwenWithoutProviderExtraBody(t *testing.T) {
	var v claudeMacDebugVars

	out := renderClaudeMacPreflight(
		claudeMacHealth{OK: true, Engine: "fak", Model: "lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M", Planner: "proxy"},
		v,
		"http://node.example:8080",
		"",
		"gateway-bearer",
		"",
	)
	for _, want := range []string{
		"WARN: Qwen3.6 request tuning is not visible",
		"FAK_PROVIDER_EXTRA_BODY_JSON",
		"top_k",
		"preserve_thinking",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("preflight warning missing %q:\n%s", want, out)
		}
	}
}

// TestRenderClaudeMacPreflightLinkAnnotations pins the two link-honesty fixes:
// the default (unreachable) Grafana URL is SUPPRESSED rather than printed as a
// dead link, and the bearer-gated note is omitted when no auth is in force.
func TestRenderClaudeMacPreflightLinkAnnotations(t *testing.T) {
	var v claudeMacDebugVars
	v.Gateway.VDSO = true

	// Default Grafana + no auth: no grafana line, no bearer note.
	out := renderClaudeMacPreflight(
		claudeMacHealth{OK: true, Engine: "fak", Planner: "inkernel"},
		v, "http://node.example:8080", "m", "none", defaultClaudeMacGrafana,
	)
	if strings.Contains(out, "grafana") {
		t.Fatalf("default Grafana URL must be suppressed, not printed:\n%s", out)
	}
	if strings.Contains(out, "needs the bearer") {
		t.Fatalf("bearer note must not show when auth is not gateway-bearer:\n%s", out)
	}
	// The metrics links themselves are always shown.
	if !strings.Contains(out, "/metrics") || !strings.Contains(out, "/debug/vars") {
		t.Fatalf("metrics/vars links must always be present:\n%s", out)
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

func TestClaudeMacFakProbeAbortsOnUnreachableGateway(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "super-secret-test-key")
	dir := t.TempDir()

	var stdout, stderr bytes.Buffer
	code := runClaudeMacFak(&stdout, &stderr, []string{
		"--probe",
		"--claude-config-dir", dir,
		// 127.0.0.1:1 is a closed reserved port: connection refused, fast.
		"--gateway-url", "http://127.0.0.1:1",
		"--model", "qwen-local",
		"--command", "fak-no-such-claude-binary-xyz",
	})
	if code != 1 {
		t.Fatalf("unreachable probe must abort with code 1, got %d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "gateway unreachable") {
		t.Fatalf("expected a gateway-unreachable error, stderr=%q", stderr.String())
	}
	if strings.Contains(stdout.String(), "launching claude") {
		t.Fatalf("must not launch claude after an unreachable-gateway abort:\n%s", stdout.String())
	}
}

func TestClaudeMacFakProbePreflightsQuietly(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "super-secret-test-key")
	var sawHealth, sawVars bool
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/healthz":
			sawHealth = true
			_, _ = w.Write([]byte(`{"ok":true,"engine":"metal","model":"qwen-local","planner":"inkernel"}`))
		case "/debug/vars":
			sawVars = true
			_, _ = w.Write([]byte(`{"gateway":{"up":true,"vdso":true},"kernel":{"submits":1},"runtime":{"num_goroutine":1,"memory":{"heap_alloc_bytes":1}}}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()
	dir := t.TempDir()

	var stdout, stderr bytes.Buffer
	_ = runClaudeMacFak(&stdout, &stderr, []string{
		"--probe",
		"--claude-config-dir", dir,
		"--gateway-url", ts.URL,
		"--model", "qwen-local",
		"--command", "fak-no-such-claude-binary-xyz",
	})
	if !sawHealth || !sawVars {
		t.Fatalf("probe did not preflight the gateway: health=%v vars=%v", sawHealth, sawVars)
	}
	if strings.Contains(stdout.String(), "fak debug") || strings.Contains(stdout.String(), "launching claude") {
		t.Fatalf("headless probe preflight must stay quiet on stdout:\n%s", stdout.String())
	}
}

// TestMacAliasRoutesToClaudeMacFak proves the crisp `fak mac` handle (#2693) is a live
// dispatch spelling that resolves to the SAME verb as the long `fak claude-mac-fak` — an
// alias, not a rename. The routing itself is the shared `case "claude-mac-fak", "mac":`
// arm in main.go; this pins it through the live, dispatch-derived verb catalog, so a
// later edit that drops either spelling — or forks them onto different handlers — reds
// here. The identical launch plan is guaranteed by construction: both arms make the one
// `cmdClaudeMacFak(os.Args[2:])` call, so there is no divergent code path to drift.
func TestMacAliasRoutesToClaudeMacFak(t *testing.T) {
	cat := helpCatalog()
	if cat == nil {
		t.Skip("devindex catalog unavailable (no repo root); alias routing is only checkable in-repo")
	}
	// Both spellings must be LIVE dispatch tokens derived from main.go's switch.
	live := map[string]bool{}
	for _, v := range cat.Verbs() {
		for _, sp := range v.Spellings() {
			live[strings.ToLower(sp)] = true
		}
	}
	for _, sp := range []string{"mac", "claude-mac-fak"} {
		if !live[sp] {
			t.Errorf("%q is not a live dispatch spelling — the alias is not wired in main.go", sp)
		}
	}
	// And both must resolve to the one verb: two names, one handler.
	macV, ok := cat.VerbByName("mac")
	if !ok {
		t.Fatal("VerbByName(mac) did not resolve — mac is not a registered alias")
	}
	longV, ok := cat.VerbByName("claude-mac-fak")
	if !ok {
		t.Fatal("VerbByName(claude-mac-fak) did not resolve")
	}
	if macV.Name != "claude-mac-fak" || macV.Name != longV.Name {
		t.Errorf("mac -> %q, claude-mac-fak -> %q; want both = claude-mac-fak", macV.Name, longV.Name)
	}
}

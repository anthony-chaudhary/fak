package main

import (
	"bytes"
	"net"
	"net/http"
	"strings"
	"testing"
)

// node_bugs_test.go covers the six confirmed correctness bugs from the #995 adversarial
// review: addr/port parsing + loopback classification (#5), key non-rotation (#4), the
// status real-port probe (#1), the Windows restart-on-failure task (#6), the port-free
// pre-start check (#3), and the honest install health gate (#2).

// --- #5: parseNodeAddr decomposition + loopback classification ----------------------

func TestParseNodeAddr(t *testing.T) {
	cases := []struct {
		name        string
		addr        string
		port        int
		remote      bool
		wantBind    string
		wantPort    string
		wantOffHost bool
		wantErr     bool
	}{
		// Host-only --addr must KEEP --port (the #5 case-1 bug: --port was dropped, producing
		// a bogus :0.0.0.0 health URL).
		{"host-only addr keeps --port", "0.0.0.0", 9000, false, "0.0.0.0:9000", "9000", true, false},
		// IPv6 loopback is LOCAL (the #5 case-2 bug: [::1] was classified off-host).
		{"ipv6 loopback is local", "[::1]:8080", 8080, false, "[::1]:8080", "8080", false, false},
		{"ipv6 loopback host-only is local", "::1", 8080, false, "[::1]:8080", "8080", false, false},
		{"localhost literal is local", "localhost:8080", 8080, false, "localhost:8080", "8080", false, false},
		{"127.0.0.1 is local", "127.0.0.1:8080", 8080, false, "127.0.0.1:8080", "8080", false, false},
		{"127.x loopback range is local", "127.5.5.5:8080", 8080, false, "127.5.5.5:8080", "8080", false, false},
		// addr with explicit port wins over --port.
		{"addr port wins over --port", "0.0.0.0:9000", 8080, false, "0.0.0.0:9000", "9000", true, false},
		// no addr: loopback default, off-host with --remote.
		{"no addr loopback default", "", 8080, false, "127.0.0.1:8080", "8080", false, false},
		{"no addr remote is 0.0.0.0", "", 8080, true, "0.0.0.0:8080", "8080", true, false},
		// a routable bind is off-host.
		{"routable addr is off-host", "192.168.1.5:8080", 8080, false, "192.168.1.5:8080", "8080", true, false},
		// errors: out-of-range port, unresolvable port.
		{"port out of range", "", 70000, false, "", "", false, true},
		{"wildcard :port is off-host", ":9000", 8080, false, "0.0.0.0:9000", "9000", true, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			bind, port, off, err := parseNodeAddr(c.addr, c.port, c.remote)
			if c.wantErr {
				if err == nil {
					t.Fatalf("parseNodeAddr(%q,%d,%v) = (%q,%q,%v,nil), want error", c.addr, c.port, c.remote, bind, port, off)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseNodeAddr(%q,%d,%v) unexpected error: %v", c.addr, c.port, c.remote, err)
			}
			if bind != c.wantBind || port != c.wantPort || off != c.wantOffHost {
				t.Errorf("parseNodeAddr(%q,%d,%v) = (%q,%q,%v), want (%q,%q,%v)",
					c.addr, c.port, c.remote, bind, port, off, c.wantBind, c.wantPort, c.wantOffHost)
			}
		})
	}
}

// TestParseNodeAddrNeverProducesBogusHealthURL is the targeted regression for the
// `http://127.0.0.1:0.0.0.0/healthz` symptom: the returned localPort must always be a bare
// numeric port, never an address fragment.
func TestParseNodeAddrNeverProducesBogusHealthURL(t *testing.T) {
	_, port, _, err := parseNodeAddr("0.0.0.0", 9000, false)
	if err != nil {
		t.Fatal(err)
	}
	if _, perr := net.LookupPort("tcp", port); perr != nil {
		t.Errorf("localPort %q is not a valid port (would form a bogus health URL)", port)
	}
}

// --- #4: nodeResolveKey reuses the persisted key instead of silently rotating ----------

func TestNodeResolveKeyReusesPersistedKey(t *testing.T) {
	dir := nodeTestRedirectConfig(t)
	t.Setenv("FAK_GATEWAY_KEY", "") // ensure no env override

	// First install: nothing persisted ⇒ mint a fresh key.
	k1, minted1, err := nodeResolveKey(dir, "FAK_GATEWAY_KEY", false)
	if err != nil || !minted1 || k1 == "" {
		t.Fatalf("first resolve: key=%q minted=%v err=%v, want a freshly-minted key", k1, minted1, err)
	}
	// Persist it as an install would.
	if werr := nodeWriteInstallState(dir, nodeInstallState{Key: k1, OffHost: true}); werr != nil {
		t.Fatal(werr)
	}

	// Re-install WITHOUT --rotate-key and without an env key ⇒ REUSE k1 (the #4 fix).
	k2, minted2, err := nodeResolveKey(dir, "FAK_GATEWAY_KEY", false)
	if err != nil {
		t.Fatal(err)
	}
	if k2 != k1 || minted2 {
		t.Errorf("re-install rotated the key: got %q (minted=%v), want the persisted %q reused", k2, minted2, k1)
	}

	// --rotate-key DOES mint a fresh, different key.
	k3, minted3, err := nodeResolveKey(dir, "FAK_GATEWAY_KEY", true)
	if err != nil {
		t.Fatal(err)
	}
	if !minted3 || k3 == k1 {
		t.Errorf("--rotate-key should mint a new key, got %q (minted=%v) vs %q", k3, minted3, k1)
	}
}

func TestNodeResolveKeyEnvWins(t *testing.T) {
	dir := nodeTestRedirectConfig(t)
	if werr := nodeWriteInstallState(dir, nodeInstallState{Key: "persisted"}); werr != nil {
		t.Fatal(werr)
	}
	t.Setenv("FAK_GATEWAY_KEY", "from-env")
	k, minted, err := nodeResolveKey(dir, "FAK_GATEWAY_KEY", false)
	if err != nil {
		t.Fatal(err)
	}
	if k != "from-env" || minted {
		t.Errorf("an explicit env key must win and not be flagged as minted: got %q minted=%v", k, minted)
	}
}

// TestNodeInstallStateRoundTrip pins the persisted-state read/write the status + key fixes
// depend on.
func TestNodeInstallStateRoundTrip(t *testing.T) {
	dir := nodeTestRedirectConfig(t)
	want := nodeInstallState{Addr: "0.0.0.0:9000", Port: "9000", Key: "k", KeyEnv: "FAK_GATEWAY_KEY", OffHost: true}
	if err := nodeWriteInstallState(dir, want); err != nil {
		t.Fatal(err)
	}
	got, err := nodeReadInstallState(dir)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Errorf("install-state round-trip: got %+v, want %+v", got, want)
	}
}

// --- #1: status probes the installed port, not the literal :8080 ---------------------

func TestStatusProbesInstalledPort(t *testing.T) {
	dir := nodeTestRedirectConfig(t)
	// Record a custom-port install (the host case where there is no client node.json).
	if err := nodeWriteInstallState(dir, nodeInstallState{Addr: "127.0.0.1:9000", Port: "9000", OffHost: false}); err != nil {
		t.Fatal(err)
	}

	var probed []string
	nodeTestSwapTransport(t, roundTripFunc(func(r *http.Request) (*http.Response, error) {
		probed = append(probed, r.URL.Host)
		return &http.Response{StatusCode: 200, Status: "200 OK", Body: http.NoBody, Header: make(http.Header)}, nil
	}))

	var out, errb bytes.Buffer
	_ = nodeStatus(&out, &errb, nil)

	// The probe must hit :9000 (the installed port), never the literal :8080.
	hit9000 := false
	for _, h := range probed {
		if strings.Contains(h, ":9000") {
			hit9000 = true
		}
		if strings.Contains(h, ":8080") {
			t.Errorf("status probed the hardcoded :8080 despite a :9000 install (%v)", probed)
		}
	}
	if !hit9000 {
		t.Errorf("status did not probe the installed :9000 port; probed=%v", probed)
	}
}

// --- #6: the Windows task XML carries restart-on-failure -----------------------------

func TestWindowsTaskXMLHasRestartOnFailure(t *testing.T) {
	xml := nodeWindowsTaskXML(`C:\fak-test\serve-runner.cmd`)
	for _, want := range []string{"<RestartOnFailure>", "<BootTrigger>", "<Count>", "cmd.exe"} {
		if !strings.Contains(xml, want) {
			t.Errorf("windows task XML missing %q:\n%s", want, xml)
		}
	}
	// The runner path must be present and XML-escaped (no raw special chars breaking the doc).
	if !strings.Contains(xml, "serve-runner.cmd") {
		t.Errorf("windows task XML must reference the runner path:\n%s", xml)
	}
}

func TestWindowsTaskXMLEscapesPath(t *testing.T) {
	// A path with an ampersand must be escaped so the XML stays well-formed.
	xml := nodeWindowsTaskXML(`C:\a & b\runner.cmd`)
	if strings.Contains(xml, "a & b") {
		t.Errorf("ampersand in the runner path must be XML-escaped, got:\n%s", xml)
	}
	if !strings.Contains(xml, "a &amp; b") {
		t.Errorf("expected &amp;-escaped path in:\n%s", xml)
	}
}

// --- #3: nodeWaitPortFree detects a free vs held port --------------------------------

func TestNodeWaitPortFree(t *testing.T) {
	// A port nobody is listening on is reported free quickly.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	_, heldPort, _ := net.SplitHostPort(ln.Addr().String())
	// While the listener is open, the port is HELD ⇒ not free.
	if nodeWaitPortFree(heldPort) {
		t.Errorf("port %s is held by a live listener but was reported free", heldPort)
	}
	ln.Close()
	// After close, the port should be free.
	if !nodeWaitPortFree(heldPort) {
		t.Errorf("port %s was reported held after the listener closed", heldPort)
	}
}

// --- #2: the install health gate is honest -------------------------------------------

// TestNodeWaitHealthyReportsDown pins that the shared health gate returns false when nothing
// answers — the signal the installers now use to warn + fail instead of falsely succeeding.
func TestNodeWaitHealthyReportsDown(t *testing.T) {
	nodeTestSwapTransport(t, roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return nil, &nodeDialErr{} // gateway never comes up
	}))
	// Point at a port nothing serves; the probe always fails, so the gate must be false.
	var out bytes.Buffer
	if nodeWaitHealthy(&out, "http://127.0.0.1:1") {
		t.Errorf("nodeWaitHealthy must return false when the gateway never answers")
	}
}

func TestNodeWaitHealthyReportsUp(t *testing.T) {
	nodeTestSwapTransport(t, roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Status: "200 OK", Body: http.NoBody, Header: make(http.Header)}, nil
	}))
	var out bytes.Buffer
	if !nodeWaitHealthy(&out, "http://127.0.0.1:8080") {
		t.Errorf("nodeWaitHealthy must return true when the gateway answers 2xx")
	}
}

// TestNodeWarnUnhealthyPointsAtLogs pins the loud failure banner names both serve log files.
func TestNodeWarnUnhealthyPointsAtLogs(t *testing.T) {
	var errb bytes.Buffer
	nodeWarnUnhealthy(&errb, "/var/fak/logs")
	s := errb.String()
	if !strings.Contains(s, "serve.log") || !strings.Contains(s, "serve.err") {
		t.Errorf("unhealthy warning must point at serve.log and serve.err, got:\n%s", s)
	}
	if !strings.Contains(strings.ToLower(s), "error") {
		t.Errorf("unhealthy warning must read as an error, got:\n%s", s)
	}
}

// TestNodeReportKeyDispositionFlagsRotation pins that a fresh mint on a re-install is flagged
// loudly (the silent-rotation #4 failure surfaced), and a reuse is noted.
func TestNodeReportKeyDispositionFlagsRotation(t *testing.T) {
	var rotated, reused bytes.Buffer
	nodeReportKeyDisposition(&rotated, true, true)
	if !strings.Contains(rotated.String(), "re-run") {
		t.Errorf("a rotation must tell the operator clients must re-run use, got: %s", rotated.String())
	}
	nodeReportKeyDisposition(&reused, false, false)
	if !strings.Contains(strings.ToLower(reused.String()), "reusing") {
		t.Errorf("a reuse must say so, got: %s", reused.String())
	}
}

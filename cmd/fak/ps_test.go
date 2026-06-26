package main

// ps_test.go — exercises `fak ps` / `fak top` (runPS) against the same stub
// gateway the session CLI tests use (stubGateway serves /v1/fak/sessions with two
// canned drive records). It proves the read-only fold: the aligned table, the
// raw --json passthrough, the empty reading, the transport-error exit, the
// flag/usage rejection, and that --watch --frames renders a bounded number of
// frames and that `fak top`'s watch-on default is overridable.

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// runPSAt drives runPS against addr and returns (stdout, stderr, exit).
// watchDefault matches the cmdPS (false) vs cmdTop (true) entry points.
func runPSAt(t *testing.T, addr string, watchDefault bool, args ...string) (string, string, int) {
	t.Helper()
	var out, errb bytes.Buffer
	argv := append([]string{}, args...)
	argv = append(argv, "--addr", addr)
	code := runPS(&out, &errb, argv, watchDefault)
	return out.String(), errb.String(), code
}

func TestPSTableRendersOneRowPerSession(t *testing.T) {
	g := &stubGateway{}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()

	out, errb, code := runPSAt(t, ts.URL, false)
	if code != 0 {
		t.Fatalf("ps exit = %d (%s)", code, errb)
	}
	// The header + both canned sessions + the count footer.
	for _, want := range []string{"TRACE", "STATE", "TURNS", "PRIO", "REV", "REASON",
		"urgent", "running", "bg", "throttled", "operator-throttle", "2 session(s)"} {
		if !strings.Contains(out, want) {
			t.Fatalf("ps table missing %q:\n%s", want, out)
		}
	}
	// The unbounded urgent budget renders as "inf", and a session with no reason
	// renders "-" rather than a blank column collapse.
	if !strings.Contains(out, "inf") {
		t.Fatalf("ps table did not render an unbounded budget axis as inf:\n%s", out)
	}
	if g.lastMethod != http.MethodGet || g.lastPath != "/v1/fak/sessions" {
		t.Fatalf("ps hit %s %s, want GET /v1/fak/sessions", g.lastMethod, g.lastPath)
	}
}

func TestPSJSONIsRawSessionList(t *testing.T) {
	g := &stubGateway{}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()

	out, errb, code := runPSAt(t, ts.URL, false, "--json")
	if code != 0 {
		t.Fatalf("ps --json exit = %d (%s)", code, errb)
	}
	// Machine-readable: the wire field names, not the human table headers.
	for _, want := range []string{`"trace_id"`, "urgent", "bg", `"count"`} {
		if !strings.Contains(out, want) {
			t.Fatalf("ps --json missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "TRACE\t") || strings.Contains(out, "session(s)") {
		t.Fatalf("ps --json leaked the human table:\n%s", out)
	}
}

func TestPSEmptyListReadsExplicitly(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/fak/sessions", func(w http.ResponseWriter, r *http.Request) {
		writeTestJSON(w, 200, map[string]any{"sessions": []any{}, "count": 0})
	})
	ts := httptest.NewServer(mux)
	defer ts.Close()

	out, _, code := runPSAt(t, ts.URL, false)
	if code != 0 {
		t.Fatalf("ps empty exit = %d", code)
	}
	if !strings.Contains(out, "no live sessions") {
		t.Fatalf("empty ps did not read explicitly:\n%s", out)
	}
}

func TestPSTransportErrorExits1(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/fak/sessions", func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	})
	ts := httptest.NewServer(mux)
	defer ts.Close()

	_, errb, code := runPSAt(t, ts.URL, false)
	if code != 1 {
		t.Fatalf("ps over a 500 gateway exit = %d, want 1", code)
	}
	if !strings.Contains(errb, "fak ps:") {
		t.Fatalf("ps error not surfaced on stderr: %q", errb)
	}
}

func TestPSRejectsStrayPositional(t *testing.T) {
	_, errb, code := runPSAt(t, "http://127.0.0.1:0", false, "sess-1")
	if code != 2 {
		t.Fatalf("ps with a stray positional exit = %d, want 2", code)
	}
	if !strings.Contains(errb, "unexpected argument") {
		t.Fatalf("ps did not reject the stray positional: %q", errb)
	}
}

func TestPSWatchRendersBoundedFrames(t *testing.T) {
	g := &stubGateway{}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()

	// --frames 2 with a tiny interval: two frames render, then the loop exits 0
	// (the bounded `top`). Count the table header occurrences to prove two frames.
	out, errb, code := runPSAt(t, ts.URL, true, "--frames", "2", "--interval", "1ms")
	if code != 0 {
		t.Fatalf("ps --watch --frames 2 exit = %d (%s)", code, errb)
	}
	if n := strings.Count(out, "TRACE"); n != 2 {
		t.Fatalf("watch rendered %d frames, want 2:\n%s", n, out)
	}
	if !strings.Contains(out, "every 1ms") {
		t.Fatalf("watch frame header missing the interval:\n%s", out)
	}
}

func TestPSTopWatchDefaultIsOverridable(t *testing.T) {
	g := &stubGateway{}
	ts := httptest.NewServer(g.handler())
	defer ts.Close()

	// `fak top` defaults watch on; --watch=false forces a single shot so the call
	// returns instead of looping. One frame, no watch-frame header.
	out, errb, code := runPSAt(t, ts.URL, true, "--watch=false")
	if code != 0 {
		t.Fatalf("fak top --watch=false exit = %d (%s)", code, errb)
	}
	if strings.Contains(out, "Ctrl-C to stop") {
		t.Fatalf("--watch=false still ran the watch loop:\n%s", out)
	}
	if n := strings.Count(out, "TRACE"); n != 1 {
		t.Fatalf("single-shot rendered %d frames, want 1:\n%s", n, out)
	}
}

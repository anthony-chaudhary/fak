package main

// session_query_test.go — the #3461 cross-process discovery path: `fak session ls`
// enumerates the guard-session index (pid-liveness checked), and `fak session status
// <handle>` reads a resolved LIVE session's raw /debug/vars from its own published
// gateway with the read-scoped bearer — refusing to dial a stale (dead-pid) row.

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/guardsessions"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// withSessionQueryRelations injects a fixed process-relations snapshot for liveness.
func withSessionQueryRelations(t *testing.T, procs []procguard.Proc) {
	t.Helper()
	orig := sessionQueryCollectRelations
	sessionQueryCollectRelations = func() ([]procguard.Proc, string) { return procs, "" }
	t.Cleanup(func() { sessionQueryCollectRelations = orig })
}

func recordSessionQueryRow(t *testing.T, dir string, row guardsessions.Row) {
	t.Helper()
	if err := guardsessions.Record(dir, row); err != nil {
		t.Fatal(err)
	}
}

func TestSessionStatusFetchesLiveRowWithBearer(t *testing.T) {
	t.Setenv("FAK_ADDR", "") // no prior gateway knowledge: the index answers
	var hits atomic.Int64
	var gotAuth atomic.Value
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/debug/vars" {
			http.NotFound(w, r)
			return
		}
		hits.Add(1)
		gotAuth.Store(r.Header.Get("Authorization"))
		if r.Header.Get("Authorization") != "Bearer tok-read-1" {
			w.WriteHeader(401)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"fak_gateway":{"drive":"running","startup_report":"ok"}}`))
	}))
	defer ts.Close()

	dir := t.TempDir()
	row := guardsessions.NewRow("trace-live-1", "claude", os.Getpid(), `C:\work\x`, "", "", time.Now()).
		WithGateway(ts.URL, "tok-read-1")
	recordSessionQueryRow(t, dir, row)
	withSessionQueryRelations(t, []procguard.Proc{{PID: os.Getpid()}})

	var out, errb bytes.Buffer
	rc := runSession(&out, &errb, []string{"status", row.Handle, "--reg-dir", dir, "--json"})
	if rc != 0 {
		t.Fatalf("status exit = %d, stderr=%s", rc, errb.String())
	}
	if hits.Load() != 1 {
		t.Fatalf("gateway hit %d times, want 1", hits.Load())
	}
	if auth, _ := gotAuth.Load().(string); auth != "Bearer tok-read-1" {
		t.Fatalf("fetch presented %q, want the row's read bearer", auth)
	}
	// --json passes the fetched /debug/vars JSON through verbatim (plus a newline).
	var doc map[string]any
	if err := json.Unmarshal(out.Bytes(), &doc); err != nil {
		t.Fatalf("--json output is not the raw JSON document: %v\n%s", err, out.String())
	}
	if _, ok := doc["fak_gateway"]; !ok {
		t.Fatalf("fetched document missing status block: %s", out.String())
	}

	// A trace-id PREFIX resolves the same row (human mode prints a header + the body).
	out.Reset()
	errb.Reset()
	if rc := runSession(&out, &errb, []string{"status", "trace-live", "--reg-dir", dir}); rc != 0 {
		t.Fatalf("prefix status exit = %d, stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), row.Handle) || !strings.Contains(out.String(), "startup_report") {
		t.Fatalf("human status output missing header/body: %s", out.String())
	}
}

func TestSessionStatusStaleRowDoesNotFetch(t *testing.T) {
	t.Setenv("FAK_ADDR", "")
	var hits atomic.Int64
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
	}))
	defer ts.Close()

	dir := t.TempDir()
	deadPID := 999_999_999 // absent from the injected relations snapshot below
	row := guardsessions.NewRow("trace-dead-1", "claude", deadPID, `C:\work\x`, "", "", time.Now()).
		WithGateway(ts.URL, "tok-read-2")
	recordSessionQueryRow(t, dir, row)
	withSessionQueryRelations(t, []procguard.Proc{{PID: os.Getpid()}})

	var out, errb bytes.Buffer
	rc := runSession(&out, &errb, []string{"status", row.Handle, "--reg-dir", dir})
	if rc != 1 {
		t.Fatalf("stale status exit = %d, want 1 (stdout=%s stderr=%s)", rc, out.String(), errb.String())
	}
	if !strings.Contains(errb.String(), "stale") {
		t.Fatalf("stale row not reported as stale: %s", errb.String())
	}
	if hits.Load() != 0 {
		t.Fatalf("stale row's gateway was dialed %d times, want 0", hits.Load())
	}

	// --json reports the machine shape without fetching either.
	out.Reset()
	errb.Reset()
	if rc := runSession(&out, &errb, []string{"status", row.Handle, "--reg-dir", dir, "--json"}); rc != 1 {
		t.Fatalf("stale --json exit = %d, want 1", rc)
	}
	var rep struct {
		Schema string `json:"schema"`
		Handle string `json:"handle"`
		State  string `json:"state"`
	}
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("stale --json undecodable: %v\n%s", err, out.String())
	}
	if rep.Schema != "fak.session-status.v1" || rep.Handle != row.Handle || rep.State != "stale" {
		t.Fatalf("stale --json shape wrong: %+v", rep)
	}
	if hits.Load() != 0 {
		t.Fatalf("stale row's gateway was dialed after --json, want 0 hits")
	}
}

func TestSessionLsListsIndexRowsWithLiveness(t *testing.T) {
	t.Setenv("FAK_ADDR", "")
	dir := t.TempDir()
	live := guardsessions.NewRow("trace-ls-live", "claude", os.Getpid(), `C:\work\a`, "", "", time.Now()).
		WithGateway("http://127.0.0.1:50001", "secret-live")
	dead := guardsessions.NewRow("trace-ls-dead", "codex", 999_999_998, `C:\work\b`, "", "", time.Now().Add(-time.Minute)).
		WithGateway("http://127.0.0.1:50002", "secret-dead")
	recordSessionQueryRow(t, dir, live)
	recordSessionQueryRow(t, dir, dead)
	withSessionQueryRelations(t, []procguard.Proc{{PID: os.Getpid()}})

	var out, errb bytes.Buffer
	if rc := runSession(&out, &errb, []string{"ls", "--reg-dir", dir, "--json"}); rc != 0 {
		t.Fatalf("ls --json exit = %d, stderr=%s", rc, errb.String())
	}
	var doc struct {
		Schema   string `json:"schema"`
		Sessions []struct {
			Handle     string `json:"handle"`
			TraceID    string `json:"trace_id"`
			GatewayURL string `json:"gateway_url"`
			Bearer     string `json:"bearer"`
			State      string `json:"state"`
		} `json:"sessions"`
	}
	if err := json.Unmarshal(out.Bytes(), &doc); err != nil {
		t.Fatalf("ls --json undecodable: %v\n%s", err, out.String())
	}
	if doc.Schema != "fak.session-ls.v1" || len(doc.Sessions) != 2 {
		t.Fatalf("ls --json shape wrong: %s", out.String())
	}
	states := map[string]string{}
	for _, s := range doc.Sessions {
		states[s.TraceID] = s.State
		if s.Bearer != "" {
			t.Fatalf("ls --json leaked a read bearer for %s", s.Handle)
		}
		if s.GatewayURL == "" {
			t.Fatalf("ls --json missing gateway_url for %s", s.Handle)
		}
	}
	if states["trace-ls-live"] != "live" || states["trace-ls-dead"] != "stale" {
		t.Fatalf("liveness states wrong: %v", states)
	}
	if strings.Contains(out.String(), "secret-live") || strings.Contains(out.String(), "secret-dead") {
		t.Fatalf("bearer bytes leaked into ls output: %s", out.String())
	}

	// Human table: both rows list, dead one marked stale, with the gateway column.
	out.Reset()
	errb.Reset()
	if rc := runSession(&out, &errb, []string{"ls", "--reg-dir", dir}); rc != 0 {
		t.Fatalf("ls exit = %d, stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), live.Handle) || !strings.Contains(out.String(), dead.Handle) ||
		!strings.Contains(out.String(), "stale") || !strings.Contains(out.String(), "http://127.0.0.1:50001") {
		t.Fatalf("ls table missing rows/columns:\n%s", out.String())
	}
}

func TestSessionStatusUnmatchedQueryFallsBackToLegacyGateway(t *testing.T) {
	t.Setenv("FAK_ADDR", "")
	dir := t.TempDir()
	withSessionQueryRelations(t, nil)
	var out, errb bytes.Buffer
	// Nothing in the index matches; the legacy single-gateway read runs instead (on a
	// box with no default-port gateway it fails to dial — exit 1). The assertions are
	// deliberately loose about the LEGACY outcome (a shared dev box may or may not have
	// something on the default port): what matters is that the index path neither
	// claimed the query (no "stale"/ambiguity report) nor swallowed it (rc 0/1 only).
	rc := runSession(&out, &errb, []string{"status", "no-such-session", "--reg-dir", dir})
	if rc != 0 && rc != 1 {
		t.Fatalf("unmatched status exit = %d, want the legacy path's 0/1", rc)
	}
	if strings.Contains(errb.String(), "stale") {
		t.Fatalf("unmatched query mis-reported as stale: %s", errb.String())
	}
}

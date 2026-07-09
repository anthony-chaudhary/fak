package stepbatoncapture

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/stepbaton"
)

// snapshotJSON is one gateway ctxvalue wire body with the given sessions, written as raw
// JSON so the test exercises the real decode path (and pins the json tags against the
// gateway's CtxValueSnapshot contract, not against this package's own structs).
func snapshotJSON(sessions string) string {
	return `{"schema":"fak-ctxvalue-report/1","budget_tokens":100000,"sessions":[` + sessions + `]}`
}

// reportJSON is one session report with the fields this package projects.
func reportJSON(trace, stepClass, basis, reason, phase string, resident, budget, turns int) string {
	return `{` +
		`"trace_id":"` + trace + `",` +
		`"tokens":{"resident_tokens":` + itoa(resident) + `,"budget_tokens":` + itoa(budget) + `},` +
		`"turns":{"turns_observed":` + itoa(turns) + `},` +
		`"session":{"phase":"` + phase + `"},` +
		`"step_advice":{"step_class":"` + stepClass + `","basis":"` + basis + `","reason":"` + reason + `"}` +
		`}`
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

// serveSnapshot returns an httptest server that answers GET /v1/fak/ctxvalue with body.
func serveSnapshot(t *testing.T, body string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/fak/ctxvalue" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestCtxvalueURL(t *testing.T) {
	cases := map[string]string{
		"":                             "",
		"   ":                          "",
		"http://h:9/":                  "http://h:9/v1/fak/ctxvalue",
		"http://h:9":                   "http://h:9/v1/fak/ctxvalue",
		"http://h:9/v1":                "http://h:9/v1/fak/ctxvalue",
		"http://h:9/v1/":               "http://h:9/v1/fak/ctxvalue",
		"http://h:9/metrics":           "http://h:9/v1/fak/ctxvalue", // the hook's own scrape URL
		"https://host.example/base/v1": "https://host.example/base/v1/fak/ctxvalue",
	}
	for in, want := range cases {
		if got := ctxvalueURL(in); got != want {
			t.Errorf("ctxvalueURL(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestSelectReport(t *testing.T) {
	var snap wireSnapshot
	if _, ok := selectReport(snap, ""); ok {
		t.Fatal("empty snapshot must select nothing")
	}
	if _, ok := selectReport(snap, "sess-1"); ok {
		t.Fatal("empty snapshot with a hint must still select nothing")
	}

	one := wireReport{TraceID: "only"}
	if got, ok := selectReport(wireSnapshot{Sessions: []wireReport{one}}, ""); !ok || got.TraceID != "only" {
		t.Fatalf("single session must be selected, got %q ok=%v", got.TraceID, ok)
	}
	// A hint that matches nothing still falls back to the sole session.
	if got, ok := selectReport(wireSnapshot{Sessions: []wireReport{one}}, "no-match"); !ok || got.TraceID != "only" {
		t.Fatalf("unmatched hint over one session must fall back to it, got %q ok=%v", got.TraceID, ok)
	}

	a := wireReport{TraceID: "a"}
	a.Turns.TurnsObserved = 3
	b := wireReport{TraceID: "b"}
	b.Turns.TurnsObserved = 9
	multi := wireSnapshot{Sessions: []wireReport{a, b}}
	// Exact hint wins even when it is not the most-live.
	if got, ok := selectReport(multi, "a"); !ok || got.TraceID != "a" {
		t.Fatalf("exact hint must win, got %q ok=%v", got.TraceID, ok)
	}
	// No hint -> the most-live (most observed turns).
	if got, ok := selectReport(multi, ""); !ok || got.TraceID != "b" {
		t.Fatalf("no hint must pick most-live, got %q ok=%v", got.TraceID, ok)
	}
}

func TestCaptureProjectsLiveReport(t *testing.T) {
	body := snapshotJSON(reportJSON("trace-xyz", stepbaton.StepCheckpoint, "token_headroom",
		"resident 80k of 100k budget (80% used)", "crowding", 80000, 100000, 12))
	srv := serveSnapshot(t, body)

	stamp, ok, err := Capture(context.Background(), Options{
		BaseURL: srv.URL, TraceHint: "trace-xyz", SHA: "deadbeef", Client: srv.Client(),
	})
	if err != nil || !ok {
		t.Fatalf("Capture: ok=%v err=%v", ok, err)
	}
	if stamp.StepClass != stepbaton.StepCheckpoint {
		t.Errorf("StepClass = %q, want %q", stamp.StepClass, stepbaton.StepCheckpoint)
	}
	if stamp.Basis != "token_headroom" {
		t.Errorf("Basis = %q", stamp.Basis)
	}
	if stamp.ResidentTokens != 80000 || stamp.BudgetTokens != 100000 {
		t.Errorf("tokens = %d/%d, want 80000/100000", stamp.ResidentTokens, stamp.BudgetTokens)
	}
	if stamp.Phase != "crowding" {
		t.Errorf("Phase = %q, want crowding", stamp.Phase)
	}
	if stamp.TraceID != "trace-xyz" {
		t.Errorf("TraceID = %q", stamp.TraceID)
	}
	if stamp.CapturedAtSHA != "deadbeef" {
		t.Errorf("CapturedAtSHA = %q", stamp.CapturedAtSHA)
	}
	if !stamp.ShouldCarry() {
		t.Error("a checkpoint stamp must ShouldCarry()")
	}
}

func TestCaptureAndWritePersists(t *testing.T) {
	body := snapshotJSON(reportJSON("t1", stepbaton.StepRebuild, "context_event",
		"context event 1 turn(s) ago", "post_event", 40000, 100000, 5))
	srv := serveSnapshot(t, body)
	dir := t.TempDir()

	stamp, ok, err := CaptureAndWrite(context.Background(), Options{
		BaseURL: srv.URL, Dir: dir, SessionID: "sess/../weird id", Client: srv.Client(),
	})
	if err != nil || !ok {
		t.Fatalf("CaptureAndWrite: ok=%v err=%v", ok, err)
	}
	got, present, err := stepbaton.Read(stepbaton.Path(dir, "sess/../weird id"))
	if err != nil || !present {
		t.Fatalf("Read back: present=%v err=%v", present, err)
	}
	if got.StepClass != stamp.StepClass || got.StepClass != stepbaton.StepRebuild {
		t.Errorf("round-trip class = %q, want %q", got.StepClass, stepbaton.StepRebuild)
	}
	if got.ResidentTokens != 40000 || got.Phase != "post_event" {
		t.Errorf("round-trip fields = %d/%q", got.ResidentTokens, got.Phase)
	}
	if !got.ShouldCarry() {
		t.Error("a rebuild stamp must ShouldCarry()")
	}
}

func TestCaptureEmptySnapshotNoWrite(t *testing.T) {
	srv := serveSnapshot(t, snapshotJSON(""))
	dir := t.TempDir()

	stamp, ok, err := CaptureAndWrite(context.Background(), Options{
		BaseURL: srv.URL, Dir: dir, SessionID: "s1", Client: srv.Client(),
	})
	if err != nil {
		t.Fatalf("empty snapshot must not error: %v", err)
	}
	if ok {
		t.Fatalf("empty snapshot must select nothing, got stamp %+v", stamp)
	}
	if _, err := os.Stat(stepbaton.Path(dir, "s1")); !os.IsNotExist(err) {
		t.Errorf("no stamp file must be written on an empty snapshot (stat err=%v)", err)
	}
	// The directory itself must not have been created for a no-op capture.
	if entries, _ := os.ReadDir(dir); len(entries) != 0 {
		t.Errorf("empty snapshot wrote %d entries, want 0", len(entries))
	}
}

func TestCaptureNoGatewayConfigured(t *testing.T) {
	stamp, ok, err := Capture(context.Background(), Options{BaseURL: "   "})
	if err != nil || ok {
		t.Fatalf("no gateway must fail open: ok=%v err=%v stamp=%+v", ok, err, stamp)
	}
}

func TestCaptureUnreachableIsError(t *testing.T) {
	srv := serveSnapshot(t, snapshotJSON(""))
	url := srv.URL
	client := srv.Client()
	srv.Close() // now the endpoint is dead

	_, ok, err := Capture(context.Background(), Options{BaseURL: url, Client: client})
	if err == nil {
		t.Fatal("an unreachable gateway must return an error")
	}
	if ok {
		t.Fatal("an unreachable gateway must select nothing")
	}
}

func TestCaptureNon2xxIsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)

	_, ok, err := Capture(context.Background(), Options{BaseURL: srv.URL, Client: srv.Client()})
	if err == nil || ok {
		t.Fatalf("a 500 must be a fail-open error: ok=%v err=%v", ok, err)
	}
}

func TestCaptureOffVocabClassNormalized(t *testing.T) {
	body := snapshotJSON(reportJSON("t", "wat-not-a-class", "token_headroom", "r", "cruising", 1, 2, 1))
	srv := serveSnapshot(t, body)

	stamp, ok, err := Capture(context.Background(), Options{BaseURL: srv.URL, Client: srv.Client()})
	if err != nil || !ok {
		t.Fatalf("Capture: ok=%v err=%v", ok, err)
	}
	if stamp.StepClass != stepbaton.StepUnknown {
		t.Errorf("off-vocab class = %q, want normalized to %q", stamp.StepClass, stepbaton.StepUnknown)
	}
	if stamp.ShouldCarry() {
		t.Error("an unknown stamp must not ShouldCarry()")
	}
}

func TestCapturePathRequested(t *testing.T) {
	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(snapshotJSON("")))
	}))
	t.Cleanup(srv.Close)

	_, _, _ = Capture(context.Background(), Options{BaseURL: srv.URL, Client: srv.Client()})
	if gotPath != "/v1/fak/ctxvalue" {
		t.Errorf("requested path = %q, want /v1/fak/ctxvalue", gotPath)
	}
}

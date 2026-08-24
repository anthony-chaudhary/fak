package main

// guard_session_gateway_publish_test.go — #5400, the PRODUCER half of guard-session gateway
// discovery. #3461 shipped the readers (`fak session ls|status`, `fak cachevalue census`)
// and the gateway's read-scoped auth, but nothing ever WROTE gateway_url/bearer: every one
// of those consumers read an empty field forever. These cases drive the REAL record path
// (recordGuardSessionIndex → publishGuardSessionGateway → guardsessions.Load), not
// Row.WithGateway in isolation, and prove the published token is read-scoped.

import (
	"bytes"
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/guardsessions"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// withGuardSessionPublications isolates the process-wide publication registry so one case
// cannot publish another's rows.
func withGuardSessionPublications(t *testing.T) {
	t.Helper()
	prev := guardSessionGatewayPublications
	guardSessionGatewayPublications = nil
	t.Cleanup(func() { guardSessionGatewayPublications = prev })
}

// guardSessionRowByHandle re-reads the index from disk and returns the folded row for
// handle — the exact view a SECOND process gets.
func guardSessionRowByHandle(t *testing.T, regDir, handle string) guardsessions.Row {
	t.Helper()
	res := guardsessions.Resolve(guardsessions.Load(regDir), handle)
	if res.Matched != 1 {
		t.Fatalf("handle %q resolved %d rows in %s, want exactly 1", handle, res.Matched, guardsessions.IndexPath(regDir))
	}
	return res.Row
}

// The defect #5400 names is not a missing function — WithGateway already existed and was
// already tested. It is a missing PRODUCTION CALL: the launch path never registered its row
// for the stamp, so the writer had no caller and every consumer read an empty field. This
// case therefore drives the guard's OWN entry points, not the helpers: the launch record via
// maybeRecordGuardSessionIndex (`fak guard -- claude` reaches it at guard.go:623 with the
// audit journal, the argv, and the start instant) and the post-bind stamp via
// publishGuardSessionGateway (guard.go:971, once guardWaitHealthy + MarkReady have made the
// address answer). The assertion is on the row READ BACK from the ledger file — the exact
// bytes a second process folds — not on an in-memory struct.
//
// Non-vacuity: delete the trackGuardSessionGatewayPublish registration inside
// recordGuardSessionIndex (guard_sessions.go:64) and this case fails, because the launch row
// lands in the index and is then never reachable — which IS the shipped-consumer-without-a-
// producer bug this ticket exists for.
func TestGuardLaunchEntryPointRegistersItsRowForTheGatewayStamp(t *testing.T) {
	withGuardSessionPublications(t)
	reg := t.TempDir()
	t.Setenv("FLEET_REG_DIR", reg)

	audit, err := journal.Open(filepath.Join(t.TempDir(), "audit.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer audit.Close()

	// Exactly the guard.go:623 call: the audit journal, the wrapped argv, the start instant.
	command := []string{"claude", "--dangerously-skip-permissions"}
	handle := maybeRecordGuardSessionIndex(audit, "trace-entry-1", command, time.Now())
	if handle == "" {
		t.Fatal("maybeRecordGuardSessionIndex recorded no session — the guard launch path is not writing the index at all")
	}
	launched := guardSessionRowByHandle(t, reg, handle)
	if launched.GatewayURL != "" || launched.Bearer != "" {
		t.Fatalf("launch row carries a gateway before the listener bound: %+v", launched)
	}
	if launched.Agent != command[0] || launched.AuditPath != audit.Path() {
		t.Fatalf("launch row lost its provenance: %+v", launched)
	}

	// Exactly the guard.go:971 call, with a freshly minted read bearer (guard.go:751). No
	// literal token: it is synthesized here the same way the guard synthesizes it.
	bearer := newGuardReadBearer()
	if bearer == "" {
		t.Fatal("newGuardReadBearer minted nothing")
	}
	const gwURL = "http://127.0.0.1:54322"
	published, errs := publishGuardSessionGateway(gwURL, bearer)
	if len(errs) != 0 {
		t.Fatalf("publish errors: %v", errs)
	}
	if published != 1 {
		t.Fatalf("the guard launch path published %d rows, want 1 — its recorded row was never registered for the stamp (#5400)", published)
	}

	// Read the ledger FILE back, the way a second process does.
	raw, err := os.ReadFile(guardsessions.IndexPath(reg))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), `"gateway_url":"`+gwURL+`"`) {
		t.Fatalf("no row on disk publishes the gateway:\n%s", raw)
	}
	got := guardSessionRowByHandle(t, reg, handle)
	if got.GatewayURL != gwURL || got.Bearer != bearer {
		t.Fatalf("folded row gateway_url=%q bearer=%q, want %q with the minted bearer", got.GatewayURL, got.Bearer, gwURL)
	}
	if rows := guardsessions.Load(reg); len(rows) != 1 {
		t.Fatalf("the republish folded to %d sessions, want 1 — a stamp must supersede the launch row, not invent a session", len(rows))
	}
	if got.Agent != command[0] || got.AuditPath != audit.Path() || got.PID != os.Getpid() {
		t.Fatalf("the republished row dropped launch provenance: %+v", got)
	}
}

// The launch record necessarily runs before the listener binds, so the row starts with no
// gateway; the post-bind publish must stamp THAT row (same handle) so a second process can
// reach the session from the index alone.
func TestGuardLaunchPublishesGatewayIntoTheRecordedRow(t *testing.T) {
	withGuardSessionPublications(t)
	reg := t.TempDir()
	t.Setenv("FLEET_REG_DIR", reg)

	started := time.Now()
	handle := recordGuardSessionIndex("trace-pub-1", "claude", "audit.jsonl", "nonce-1", started)
	if handle == "" {
		t.Fatal("recordGuardSessionIndex returned no handle")
	}
	if pre := guardSessionRowByHandle(t, reg, handle); pre.GatewayURL != "" || pre.Bearer != "" {
		t.Fatalf("launch row already carries a gateway before the bind: %+v", pre)
	}

	// The operator-terminal row goes through its own recorder (guard.go mirrors it into the
	// machine registry); it must be published too, and stamped IN PLACE so the clean-exit
	// tombstone recorded from the same variable keeps the fields.
	interactive := guardsessions.NewInteractiveRow("trace-pub-1", "claude", os.Getpid(), reg, "audit.jsonl", "", started.Add(time.Second), []string{"claude"}, false)
	if err := guardsessions.Record(reg, interactive); err != nil {
		t.Fatal(err)
	}
	trackGuardSessionGatewayPublish(&interactive, func(r guardsessions.Row) error { return guardsessions.Record(reg, r) })

	const gwURL = "http://127.0.0.1:54321"
	const bearer = "fakread-0123456789abcdef"
	n, errs := publishGuardSessionGateway(gwURL, bearer)
	if len(errs) != 0 {
		t.Fatalf("publish errors: %v", errs)
	}
	if n != 2 {
		t.Fatalf("published %d rows, want 2 (the index row and the interactive row)", n)
	}

	got := guardSessionRowByHandle(t, reg, handle)
	if got.GatewayURL != gwURL || got.Bearer != bearer {
		t.Fatalf("recorded row published gateway_url=%q bearer=%q, want %q / %q", got.GatewayURL, got.Bearer, gwURL, bearer)
	}
	// Publishing supersedes; it must not fabricate extra sessions.
	if rows := guardsessions.Load(reg); len(rows) != 2 {
		t.Fatalf("index folded to %d sessions, want 2 (one per handle)", len(rows))
	}
	if iv := guardSessionRowByHandle(t, reg, interactive.Handle); iv.GatewayURL != gwURL || iv.Bearer != bearer {
		t.Fatalf("interactive row not published: %+v", iv)
	}
	if interactive.GatewayURL != gwURL {
		t.Fatalf("in-place stamp did not reach the caller's row: %+v", interactive)
	}
	if ended := interactive.Ended(time.Now()); ended.GatewayURL != gwURL || ended.Bearer != bearer {
		t.Fatalf("exit tombstone dropped the published gateway: %+v", ended)
	}
}

// A guard that binds no gateway must still record its session and still omit both fields —
// their absence stays a legal row shape, not an empty-string stamp.
func TestGuardSessionPublishWithoutGatewayLeavesTheRowUnstamped(t *testing.T) {
	withGuardSessionPublications(t)
	reg := t.TempDir()
	t.Setenv("FLEET_REG_DIR", reg)

	handle := recordGuardSessionIndex("trace-nogw", "codex", "audit.jsonl", "nonce-2", time.Now())
	if handle == "" {
		t.Fatal("recordGuardSessionIndex returned no handle")
	}
	n, errs := publishGuardSessionGateway("   ", "fakread-should-not-land")
	if n != 0 || len(errs) != 0 {
		t.Fatalf("empty-url publish wrote %d rows (errs %v), want a no-op", n, errs)
	}
	raw, err := os.ReadFile(guardsessions.IndexPath(reg))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), "gateway_url") || strings.Contains(string(raw), "bearer") {
		t.Fatalf("no-gateway session leaked a gateway key into the index: %s", raw)
	}
	if row := guardSessionRowByHandle(t, reg, handle); row.GatewayURL != "" || row.Bearer != "" {
		t.Fatalf("no-gateway row is not clean: %+v", row)
	}
}

// The published bearer must be READ-scoped: it admits the observability read a consumer
// needs (from a NON-loopback caller, where the loopback exemption does not apply) and is
// refused on every mutating route. Publishing a control-grade token into a world-readable
// index would be strictly worse than publishing nothing.
func TestPublishedGuardSessionBearerIsReadScopedAndAnswersSessionStatus(t *testing.T) {
	withGuardSessionPublications(t)
	t.Setenv("FAK_ADDR", "") // no prior port knowledge: the index is the only route in
	reg := t.TempDir()
	t.Setenv("FLEET_REG_DIR", reg)

	bearer := newGuardReadBearer()
	if !strings.HasPrefix(bearer, "fakread-") || len(bearer) != len("fakread-")+32 {
		t.Fatalf("minted read bearer %q is not the expected 128-bit form", bearer)
	}
	if second := newGuardReadBearer(); second == bearer {
		t.Fatal("read bearer is not fresh per launch")
	}

	srv, err := gateway.New(gateway.Config{
		ExposeProfile: "headless",
		RequireKey:    "guard-control-key", // the full-strength credential the index never carries
		ReadBearer:    bearer,
	})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	handle := recordGuardSessionIndex("trace-scope-1", "claude", "audit.jsonl", "nonce-3", time.Now())
	if handle == "" {
		t.Fatal("recordGuardSessionIndex returned no handle")
	}
	if n, errs := publishGuardSessionGateway(ts.URL, bearer); n != 1 || len(errs) != 0 {
		t.Fatalf("publish wrote %d rows (errs %v), want 1", n, errs)
	}
	row := guardSessionRowByHandle(t, reg, handle)
	if row.Bearer != bearer || row.GatewayURL != ts.URL {
		t.Fatalf("published row = %+v, want gateway %q with the minted bearer", row, ts.URL)
	}

	// Acceptance: `fak session status <handle>` reads this session cross-process with no
	// --addr and no key — from the row alone.
	withSessionQueryRelations(t, []procguard.Proc{{PID: os.Getpid()}})
	var out, errb bytes.Buffer
	if rc := runSession(&out, &errb, []string{"status", handle, "--reg-dir", reg, "--json"}); rc != 0 {
		t.Fatalf("session status exit = %d, stderr=%s", rc, errb.String())
	}
	var doc map[string]any
	if err := json.Unmarshal(out.Bytes(), &doc); err != nil {
		t.Fatalf("session status did not return the gateway's /debug/vars document: %v\n%s", err, out.String())
	}
	if len(doc) == 0 {
		t.Fatalf("session status returned an empty document: %s", out.String())
	}

	// The SAME token the index publishes is refused on control routes.
	h := srv.Handler()
	for _, tc := range []struct{ method, path string }{
		{"POST", "/v1/messages"},
		{"POST", "/v1/fak/policy/reload"},
		{"POST", "/v1/fak/session/sess-1/run"},
	} {
		req := httptest.NewRequest(tc.method, tc.path, nil)
		req.Header.Set("Authorization", "Bearer "+row.Bearer)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != 401 {
			t.Fatalf("%s %s with the PUBLISHED bearer = %d, want 401 — the index token must not control the session", tc.method, tc.path, rec.Code)
		}
	}
	// ...and it does admit the read it exists for, off the loopback exemption.
	req := httptest.NewRequest("GET", "/debug/vars", nil)
	req.Header.Set("Authorization", "Bearer "+row.Bearer)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("GET /debug/vars with the published bearer = %d, want 200", rec.Code)
	}
	// A token that is not the published one stays out, so the read scope is the bearer's,
	// not the path's.
	req = httptest.NewRequest("GET", "/debug/vars", nil)
	req.Header.Set("Authorization", "Bearer fakread-wrong")
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != 401 {
		t.Fatalf("GET /debug/vars with a wrong bearer = %d, want 401", rec.Code)
	}
}

// Publishing a bearer into the index gives every row-RENDERING surface a live credential to
// echo. `fak session ls --json` already refuses to print it; `fak guard sessions --json` is
// the same query over the same rows and must give the same answer, in both its list and its
// single-row resolve. The discovery half (gateway_url) still renders — it is not a secret,
// and it is useless without the token.
func TestGuardSessionsJSONDoesNotEchoThePublishedBearer(t *testing.T) {
	reg := t.TempDir()
	const gwURL = "http://127.0.0.1:54323"
	bearer := newGuardReadBearer()
	row := guardsessions.NewRow("trace-redact", "claude", os.Getpid(), reg, "audit.jsonl", "", time.Now()).
		WithGateway(gwURL, bearer)
	if err := guardsessions.Record(reg, row); err != nil {
		t.Fatal(err)
	}
	// The token IS on disk — that is the contract. What must not happen is a render echoing it.
	raw, err := os.ReadFile(guardsessions.IndexPath(reg))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), bearer) {
		t.Fatalf("precondition: the index does not carry the bearer:\n%s", raw)
	}

	// cmdGuard peels the leading "sessions" before dispatching here, so argv starts after it.
	for _, argv := range [][]string{
		{"--reg-dir", reg, "--json"},             // the list
		{"--reg-dir", reg, "--json", row.Handle}, // the single-row resolve
	} {
		var out, errb bytes.Buffer
		if rc := runGuardSessions(&out, &errb, argv); rc != 0 {
			t.Fatalf("`fak guard sessions %v` exit = %d (stderr=%s)", argv, rc, errb.String())
		}
		if strings.Contains(out.String(), bearer) {
			t.Fatalf("`fak guard sessions %v` echoed the published read bearer:\n%s", argv, out.String())
		}
		if !strings.Contains(out.String(), gwURL) {
			t.Fatalf("`fak guard sessions %v` dropped the gateway_url discovery half:\n%s", argv, out.String())
		}
	}
}

// The refusal text must stop guessing "recorded by an older fak?" at every row: a session
// that started after the producer landed and still has no gateway_url published nothing,
// which is a different fact from a row written before any build could publish.
func TestSessionStatusMissingGatewayNamesTheRealCause(t *testing.T) {
	t.Setenv("FAK_ADDR", "")
	dir := t.TempDir()
	withSessionQueryRelations(t, []procguard.Proc{{PID: os.Getpid()}})

	current := guardsessions.NewRow("trace-nogw-current", "claude", os.Getpid(), dir, "", "",
		guardsessions.GatewayPublishEpoch.Add(time.Hour))
	legacy := guardsessions.NewRow("trace-nogw-legacy", "claude", os.Getpid(), dir, "", "",
		guardsessions.GatewayPublishEpoch.Add(-48*time.Hour))
	for _, row := range []guardsessions.Row{current, legacy} {
		if err := guardsessions.Record(dir, row); err != nil {
			t.Fatal(err)
		}
	}

	var out, errb bytes.Buffer
	if rc := runSession(&out, &errb, []string{"status", current.Handle, "--reg-dir", dir}); rc != 1 {
		t.Fatalf("status on an unpublished live row exit = %d, want 1 (stderr=%s)", rc, errb.String())
	}
	if strings.Contains(errb.String(), "older fak") {
		t.Fatalf("a row newer than the publish epoch was still blamed on an older fak: %s", errb.String())
	}
	if !strings.Contains(errb.String(), "bound no gateway") {
		t.Fatalf("refusal does not name the real cause: %s", errb.String())
	}

	out.Reset()
	errb.Reset()
	if rc := runSession(&out, &errb, []string{"status", legacy.Handle, "--reg-dir", dir}); rc != 1 {
		t.Fatalf("status on a legacy row exit = %d, want 1 (stderr=%s)", rc, errb.String())
	}
	if !strings.Contains(errb.String(), "before fak published session gateways") {
		t.Fatalf("a genuinely pre-publish row did not get the version explanation: %s", errb.String())
	}
	if !strings.Contains(errb.String(), guardsessions.GatewayPublishEpoch.Format(time.RFC3339)) {
		t.Fatalf("version explanation does not cite the publish epoch: %s", errb.String())
	}
}

package probe

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/deploykit/stamp"
)

const (
	commitA = "d5840f2e9ee30648b486b4284cb49f3cfccdac4a"
	commitB = "0123456789abcdef0123456789abcdef01234567"
)

// fakeDeployable serves the four probe routes from settable responses and
// records every request path and Authorization header in arrival order.
type fakeDeployable struct {
	mu      sync.Mutex
	status  map[string]int
	body    map[string]string
	hits    []string
	auth    map[string]string
	readyzN atomic.Int32
	// notReadyFor makes /readyz answer 503 for its first n requests.
	notReadyFor int32
}

func newFake(version string) *fakeDeployable {
	return &fakeDeployable{
		status: map[string]int{},
		body: map[string]string{
			PathHealth:  `{"status":"ok"}`,
			PathReady:   `{"ready":true}`,
			PathModels:  `{"object":"list","data":[{"id":"fak","object":"model"}]}`,
			PathVersion: version,
		},
		auth: map[string]string{},
	}
}

func versionDoc(commit string, dirty, stamped bool) string {
	d, s := "false", "false"
	if dirty {
		d = "true"
	}
	if stamped {
		s = "true"
	}
	return `{"app_version":"0.55.0","commit":"` + commit + `","dirty":` + d + `,"stamped":` + s + `}`
}

func (f *fakeDeployable) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	f.hits = append(f.hits, r.URL.Path)
	f.auth[r.URL.Path] = r.Header.Get("Authorization")
	status, ok := f.status[r.URL.Path]
	body, known := f.body[r.URL.Path]
	f.mu.Unlock()
	if r.URL.Path == PathReady && f.readyzN.Add(1) <= f.notReadyFor {
		w.WriteHeader(http.StatusServiceUnavailable)
		return
	}
	if !known {
		http.NotFound(w, r)
		return
	}
	if !ok {
		status = http.StatusOK
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write([]byte(body))
}

func (f *fakeDeployable) set(path string, status int, body string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.status[path] = status
	f.body[path] = body
}

func (f *fakeDeployable) paths() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.hits...)
}

func serve(t *testing.T, h http.Handler) string {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return srv.URL
}

func fastClient() Client {
	return Client{Timeout: 5 * time.Second, Interval: 5 * time.Millisecond, RequestTimeout: time.Second}
}

func asProbeError(t *testing.T, err error) *ProbeError {
	t.Helper()
	var pe *ProbeError
	if !errors.As(err, &pe) {
		t.Fatalf("error %v (%T) is not a *ProbeError", err, err)
	}
	return pe
}

func TestWaitPassesAllFourChecksInOrder(t *testing.T) {
	f := newFake(versionDoc(commitA, false, true))
	base := serve(t, f)
	res, err := fastClient().Wait(context.Background(), Target{BaseURL: base}, Expect{Commit: strings.ToUpper(commitA)})
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if !res.Ready || res.Attempts != 1 || res.Failure != "" || res.Models != 1 {
		t.Fatalf("result = %+v, want ready after one attempt with one model", res)
	}
	if res.Identity == nil || res.Identity.Commit != commitA || !res.Identity.Stamped {
		t.Fatalf("identity = %+v, want stamped %s", res.Identity, commitA)
	}
	want := []string{PathHealth, PathReady, PathModels, PathVersion}
	if got := f.paths(); strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("request order = %v, want %v", got, want)
	}
}

func TestWaitRetriesNotReadyThenPasses(t *testing.T) {
	f := newFake(versionDoc(commitA, false, true))
	f.notReadyFor = 2
	res, err := fastClient().Wait(context.Background(), Target{BaseURL: serve(t, f)}, Expect{Commit: commitA})
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if !res.Ready || res.Attempts != 3 {
		t.Fatalf("result = %+v, want ready on attempt 3", res)
	}
}

func TestWaitRetriesCatalogIsReadyFalse(t *testing.T) {
	f := newFake(versionDoc(commitA, false, true))
	f.set(PathModels, http.StatusOK, `{"object":"list","data":[],"is_ready":false}`)
	c := fastClient()
	c.Timeout = 400 * time.Millisecond
	res, err := c.Wait(context.Background(), Target{BaseURL: serve(t, f)}, Expect{Commit: commitA})
	pe := asProbeError(t, err)
	if pe.Kind != FailureNotReady || pe.Check != CheckCatalog || !res.TimedOut || res.Attempts < 2 {
		t.Fatalf("kind=%s check=%s result=%+v, want not_ready on catalog, retried until the deadline", pe.Kind, pe.Check, res)
	}
}

// hangingReadyz answers /readyz 503 for its first n requests, then holds every
// later request open until the client gives up, like a server that stops
// answering mid-restart.
func hangingReadyz(n int32) http.Handler {
	var seen atomic.Int32
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case PathHealth:
			w.WriteHeader(http.StatusOK)
		case PathReady:
			if seen.Add(1) <= n {
				w.WriteHeader(http.StatusServiceUnavailable)
				return
			}
			<-r.Context().Done()
		default:
			http.NotFound(w, r)
		}
	})
}

func TestWaitDeadlineMidRequestReportsLastCompleteSample(t *testing.T) {
	c := fastClient()
	c.Timeout = 300 * time.Millisecond
	c.RequestTimeout = 10 * time.Second
	res, err := c.Wait(context.Background(), Target{BaseURL: serve(t, hangingReadyz(2))}, Expect{Commit: commitA})
	pe := asProbeError(t, err)
	if pe.Kind != FailureNotReady || pe.Check != CheckReadiness || pe.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("kind=%s check=%s status=%d (%v), want the last complete sample: not_ready 503 on readiness", pe.Kind, pe.Check, pe.StatusCode, err)
	}
	if !res.TimedOut || res.Attempts != 2 || res.Failure != FailureNotReady || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("result = %+v err=%v, want 2 complete samples then a deadline", res, err)
	}
}

func TestWaitParentCancelMidRequestReportsLastCompleteSample(t *testing.T) {
	c := fastClient()
	c.RequestTimeout = 10 * time.Second
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(200*time.Millisecond, cancel)
	res, err := c.Wait(ctx, Target{BaseURL: serve(t, hangingReadyz(1))}, Expect{Commit: commitA})
	pe := asProbeError(t, err)
	if pe.Kind != FailureNotReady || res.TimedOut || res.Attempts != 1 || !errors.Is(err, context.Canceled) {
		t.Fatalf("kind=%s result=%+v err=%v, want the complete not_ready sample, cancelled not timed out", pe.Kind, res, err)
	}
}

// refusingDoer fails the first n requests like a dial to a port nobody is
// listening on, then hands every request to next.
type refusingDoer struct {
	n    int32
	seen atomic.Int32
	next HTTPDoer
}

func (d *refusingDoer) Do(r *http.Request) (*http.Response, error) {
	if d.seen.Add(1) <= d.n {
		return nil, &net.OpError{Op: "dial", Net: "tcp", Err: errors.New("connection refused")}
	}
	return d.next.Do(r)
}

func TestWaitRetriesNotListeningThenPasses(t *testing.T) {
	f := newFake(versionDoc(commitA, false, true))
	c := fastClient()
	c.HTTP = &refusingDoer{n: 3, next: http.DefaultClient}
	res, err := c.Wait(context.Background(), Target{BaseURL: serve(t, f)}, Expect{Commit: commitA})
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if !res.Ready || res.Attempts != 4 {
		t.Fatalf("result = %+v, want ready on attempt 4", res)
	}
}

func TestWaitGivesUpAtDeadlineWithLastKind(t *testing.T) {
	c := fastClient()
	c.Timeout = 50 * time.Millisecond
	c.HTTP = &refusingDoer{n: 1 << 30}
	start := time.Now()
	res, err := c.Wait(context.Background(), Target{BaseURL: "http://127.0.0.1:1"}, Expect{Commit: commitA})
	pe := asProbeError(t, err)
	if pe.Kind != FailureNotListening || pe.Check != CheckLiveness {
		t.Fatalf("kind=%s check=%s, want not_listening on liveness", pe.Kind, pe.Check)
	}
	if !errors.Is(err, context.DeadlineExceeded) || !res.TimedOut || res.Ready || res.Attempts < 2 {
		t.Fatalf("err=%v result=%+v, want a deadline after several attempts", err, res)
	}
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Fatalf("Wait took %s, want it bounded by the 50ms deadline", elapsed)
	}
}

func TestWaitHonorsParentCancel(t *testing.T) {
	c := fastClient()
	c.HTTP = &refusingDoer{n: 1 << 30}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	res, err := c.Wait(ctx, Target{BaseURL: "http://127.0.0.1:1"}, Expect{Commit: commitA})
	if !errors.Is(err, context.Canceled) || res.TimedOut || res.Ready || res.Attempts != 0 || res.Check != "" {
		t.Fatalf("err=%v result=%+v, want context.Canceled, no sample and no check invented", err, res)
	}
}

func TestProbeTLSMisconfigurationFailsFast(t *testing.T) {
	f := newFake(versionDoc(commitA, false, true))
	tlsSrv := httptest.NewTLSServer(f)
	t.Cleanup(tlsSrv.Close)
	c := fastClient()
	c.Timeout = 30 * time.Second
	// A cold TLS handshake on a loaded host can take over a second; a timeout
	// here would be a retryable not_listening and hide what is being tested.
	c.RequestTimeout = 20 * time.Second

	// Untrusted certificate: a server answered, and retrying cannot fix it.
	res, err := c.Wait(context.Background(), Target{BaseURL: tlsSrv.URL}, Expect{Commit: commitA})
	if pe := asProbeError(t, err); pe.Kind != FailureTransport || res.Attempts != 1 || pe.Kind.Retryable() {
		t.Fatalf("untrusted TLS: kind=%s result=%+v, want one non-retryable transport_error", pe.Kind, res)
	}
	// https against a plain-HTTP listener.
	res, err = c.Wait(context.Background(), Target{BaseURL: strings.Replace(serve(t, f), "http://", "https://", 1)}, Expect{Commit: commitA})
	if pe := asProbeError(t, err); pe.Kind != FailureTransport || res.Attempts != 1 {
		t.Fatalf("scheme mismatch: kind=%s result=%+v (%v), want one transport_error", pe.Kind, res, err)
	}
	// A client that trusts the certificate passes.
	c.HTTP = tlsSrv.Client()
	if res, err := c.Wait(context.Background(), Target{BaseURL: tlsSrv.URL}, Expect{Commit: commitA}); err != nil || !res.Ready {
		t.Fatalf("trusted TLS: %+v %v", res, err)
	}
}

// privateRouterCatalogFixtures are the /v1/models bodies the private router
// updater's TestValidateRouterReadinessResponseRequiresValidModelCatalog feeds
// its validator (fak-private cmd/fak-sync/update_router_test.go), with its
// verdicts. The probe must refuse every body that validator refuses.
var privateRouterCatalogFixtures = []struct {
	name    string
	status  int
	body    string
	wantErr bool
}{
	{name: "live discovered catalog", status: 200, body: `{"object":"list","data":[{"id":"deepseek-ai/DeepSeek-V4-Flash","object":"model"}]}`},
	{name: "explicit ready catalog", status: 200, body: `{"object":"list","data":[{"id":"deepseek-ai/DeepSeek-V4-Flash","object":"model"}],"is_ready":true}`},
	{name: "empty catalog", status: 200, body: `{"object":"list","data":[]}`, wantErr: true},
	{name: "empty model ID", status: 200, body: `{"object":"list","data":[{"id":"","object":"model"}]}`, wantErr: true},
	{name: "untrimmed model ID", status: 200, body: `{"object":"list","data":[{"id":" deepseek-ai/DeepSeek-V4-Flash ","object":"model"}]}`, wantErr: true},
	{name: "wrong catalog object", status: 200, body: `{"object":"model","data":[{"id":"deepseek-ai/DeepSeek-V4-Flash","object":"model"}]}`, wantErr: true},
	{name: "wrong item object", status: 200, body: `{"object":"list","data":[{"id":"deepseek-ai/DeepSeek-V4-Flash","object":"not-model"}]}`, wantErr: true},
	{name: "explicit not ready", status: 200, body: `{"object":"list","data":[{"id":"deepseek-ai/DeepSeek-V4-Flash","object":"model"}],"is_ready":false}`, wantErr: true},
	{name: "malformed body", status: 200, body: `{"object":"list","data":`, wantErr: true},
	{name: "non OK status", status: 503, body: `{"object":"list","data":[{"id":"deepseek-ai/DeepSeek-V4-Flash","object":"model"}]}`, wantErr: true},
	{name: "trailing JSON", status: 200, body: `{"object":"list","data":[{"id":"deepseek-ai/DeepSeek-V4-Flash","object":"model"}]}{}`, wantErr: true},
}

func TestCatalogMatchesPrivateRouterVerdicts(t *testing.T) {
	for _, tt := range privateRouterCatalogFixtures {
		t.Run(tt.name, func(t *testing.T) {
			f := newFake(versionDoc(commitA, false, true))
			f.set(PathModels, tt.status, tt.body)
			_, err := fastClient().Probe(context.Background(), Target{BaseURL: serve(t, f)}, Expect{Commit: commitA})
			if (err != nil) != tt.wantErr {
				t.Fatalf("Probe with catalog %s error = %v, wantErr %v", tt.body, err, tt.wantErr)
			}
			if err != nil && asProbeError(t, err).Check != CheckCatalog {
				t.Fatalf("failure %v is not on the catalog check", err)
			}
		})
	}
}

// TestProbeIdentityDriftFails is a negative witness: /version reports a clean
// commit other than the expected one, and Wait fails at once with
// identity_drift instead of retrying until the deadline.
func TestProbeIdentityDriftFails(t *testing.T) {
	f := newFake(versionDoc(commitB, false, true))
	c := fastClient()
	c.Timeout = 30 * time.Second
	start := time.Now()
	res, err := c.Wait(context.Background(), Target{BaseURL: serve(t, f)}, Expect{Commit: commitA})
	pe := asProbeError(t, err)
	if pe.Kind != FailureIdentityDrift || pe.Check != CheckIdentity {
		t.Fatalf("kind=%s check=%s, want identity_drift on identity", pe.Kind, pe.Check)
	}
	if res.Ready || res.Attempts != 1 || res.TimedOut || res.Failure != FailureIdentityDrift {
		t.Fatalf("result = %+v, want one failed attempt, no retry", res)
	}
	if res.Identity == nil || res.Identity.Commit != commitB {
		t.Fatalf("identity = %+v, want the observed %s recorded", res.Identity, commitB)
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Fatalf("drift took %s to report; it must fail fast, not wait out the 30s deadline", elapsed)
	}
	if !strings.Contains(err.Error(), commitB[:12]) || !strings.Contains(err.Error(), commitA[:12]) {
		t.Fatalf("error %q should name both commits", err)
	}
}

// TestProbeEmptyCatalogFails is a negative witness: a /v1/models reply with
// data:[] fails the probe at once with empty_catalog.
func TestProbeEmptyCatalogFails(t *testing.T) {
	f := newFake(versionDoc(commitA, false, true))
	f.set(PathModels, http.StatusOK, `{"object":"list","data":[]}`)
	c := fastClient()
	c.Timeout = 30 * time.Second
	res, err := c.Wait(context.Background(), Target{BaseURL: serve(t, f)}, Expect{Commit: commitA})
	pe := asProbeError(t, err)
	if pe.Kind != FailureEmptyCatalog || pe.Check != CheckCatalog {
		t.Fatalf("kind=%s check=%s, want empty_catalog on catalog", pe.Kind, pe.Check)
	}
	if res.Ready || res.Attempts != 1 {
		t.Fatalf("result = %+v, want one failed attempt", res)
	}
	for _, p := range f.paths() {
		if p == PathVersion {
			t.Fatal("identity was probed after the catalog failed; checks must stop at the first failure")
		}
	}
}

func TestProbeFailFastKinds(t *testing.T) {
	tests := []struct {
		name      string
		path      string
		status    int
		body      string
		wantKind  FailureKind
		wantCheck Check
	}{
		{"liveness 404 has no contract", PathHealth, http.StatusNotFound, `not found`, FailureBadStatus, CheckLiveness},
		{"liveness 500", PathHealth, http.StatusInternalServerError, `{}`, FailureBadStatus, CheckLiveness},
		{"catalog needs auth", PathModels, http.StatusUnauthorized, `{"error":"unauthorized"}`, FailureUnauthorized, CheckCatalog},
		{"catalog forbidden", PathModels, http.StatusForbidden, `{}`, FailureUnauthorized, CheckCatalog},
		{"catalog not JSON", PathModels, http.StatusOK, `<html>`, FailureMalformedResponse, CheckCatalog},
		{"catalog without data", PathModels, http.StatusOK, `{"object":"list"}`, FailureMalformedResponse, CheckCatalog},
		{"catalog wrong object", PathModels, http.StatusOK, `{"object":"model","data":[{"id":"x"}]}`, FailureMalformedResponse, CheckCatalog},
		{"catalog without object", PathModels, http.StatusOK, `{"data":[{"id":"x","object":"model"}]}`, FailureMalformedResponse, CheckCatalog},
		{"catalog null object", PathModels, http.StatusOK, `{"object":null,"data":[{"id":"x","object":"model"}]}`, FailureMalformedResponse, CheckCatalog},
		{"catalog item without object", PathModels, http.StatusOK, `{"object":"list","data":[{"id":"x"}]}`, FailureMalformedResponse, CheckCatalog},
		{"catalog empty id", PathModels, http.StatusOK, `{"object":"list","data":[{"id":""}]}`, FailureMalformedResponse, CheckCatalog},
		{"catalog untrimmed id", PathModels, http.StatusOK, `{"object":"list","data":[{"id":" fak "}]}`, FailureMalformedResponse, CheckCatalog},
		{"catalog trailing data", PathModels, http.StatusOK, `{"object":"list","data":[{"id":"x"}]} {}`, FailureMalformedResponse, CheckCatalog},
		{"version missing route", PathVersion, http.StatusNotFound, `404`, FailureBadStatus, CheckIdentity},
		{"version not a document", PathVersion, http.StatusOK, `{"commit":"x"}`, FailureMalformedResponse, CheckIdentity},
		{"version dirty build", PathVersion, http.StatusOK, versionDoc(commitA, true, true), FailureIdentityUnproven, CheckIdentity},
		{"version unstamped build", PathVersion, http.StatusOK, versionDoc("", false, false), FailureIdentityUnproven, CheckIdentity},
		{"version short commit", PathVersion, http.StatusOK, versionDoc(commitA[:12], false, true), FailureIdentityUnproven, CheckIdentity},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFake(versionDoc(commitA, false, true))
			f.set(tt.path, tt.status, tt.body)
			c := fastClient()
			c.Timeout = 30 * time.Second
			res, err := c.Wait(context.Background(), Target{BaseURL: serve(t, f)}, Expect{Commit: commitA})
			pe := asProbeError(t, err)
			if pe.Kind != tt.wantKind || pe.Check != tt.wantCheck {
				t.Fatalf("kind=%s check=%s (%v), want %s on %s", pe.Kind, pe.Check, err, tt.wantKind, tt.wantCheck)
			}
			if res.Attempts != 1 || res.Ready || res.Failure != tt.wantKind || res.Check != tt.wantCheck {
				t.Fatalf("result = %+v, want exactly one failed attempt", res)
			}
		})
	}
}

// TestProbeDoesNotFollowRedirects: a /readyz that redirects to a route
// answering 200 must fail as bad_status 302, not pass as ready by way of the
// redirect target.
func TestProbeDoesNotFollowRedirects(t *testing.T) {
	var healthz atomic.Int32
	base := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case PathHealth:
			healthz.Add(1)
			w.WriteHeader(http.StatusOK)
		case PathReady:
			http.Redirect(w, r, PathHealth, http.StatusFound)
		default:
			http.NotFound(w, r)
		}
	}))
	res, err := fastClient().Probe(context.Background(), Target{BaseURL: base}, Expect{SkipIdentity: true})
	pe := asProbeError(t, err)
	if pe.Kind != FailureBadStatus || pe.Check != CheckReadiness || pe.StatusCode != http.StatusFound || res.Ready {
		t.Fatalf("kind=%s check=%s status=%d, want bad_status 302 on readiness", pe.Kind, pe.Check, pe.StatusCode)
	}
	if n := healthz.Load(); n != 1 {
		t.Fatalf("/healthz requested %d times, want 1 (the redirect must not be followed)", n)
	}
}

func TestProbeOversizedBodyIsMalformed(t *testing.T) {
	f := newFake(versionDoc(commitA, false, true))
	f.set(PathVersion, http.StatusOK, `{"pad":"`+strings.Repeat("x", maxVersionBody)+`"}`)
	_, err := fastClient().Probe(context.Background(), Target{BaseURL: serve(t, f)}, Expect{Commit: commitA})
	if pe := asProbeError(t, err); pe.Kind != FailureMalformedResponse || !errors.Is(err, errBodyTooLarge) {
		t.Fatalf("err = %v, want malformed_response for an oversized body", err)
	}
}

func TestProbeSendsBearerOnlyToCatalogAndIdentity(t *testing.T) {
	f := newFake(versionDoc(commitA, false, true))
	if _, err := fastClient().Probe(context.Background(), Target{BaseURL: serve(t, f), Bearer: "k-123"}, Expect{Commit: commitA}); err != nil {
		t.Fatalf("Probe: %v", err)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	for path, want := range map[string]string{PathHealth: "", PathReady: "", PathModels: "Bearer k-123", PathVersion: "Bearer k-123"} {
		if got := f.auth[path]; got != want {
			t.Errorf("Authorization on %s = %q, want %q", path, got, want)
		}
	}
}

func TestProbeSkipIdentityNeverCallsVersion(t *testing.T) {
	f := newFake(versionDoc(commitB, false, true))
	res, err := fastClient().Probe(context.Background(), Target{BaseURL: serve(t, f)}, Expect{SkipIdentity: true})
	if err != nil || !res.Ready || res.Identity != nil {
		t.Fatalf("res=%+v err=%v, want ready with no identity observed", res, err)
	}
	for _, p := range f.paths() {
		if p == PathVersion {
			t.Fatal("SkipIdentity still requested /version")
		}
	}
}

func TestInvalidTargetFailsBeforeAnyRequest(t *testing.T) {
	var sent atomic.Int32
	c := fastClient()
	c.HTTP = doerFunc(func(*http.Request) (*http.Response, error) {
		sent.Add(1)
		return nil, errors.New("must not be called")
	})
	tests := []struct {
		name string
		t    Target
		e    Expect
	}{
		{"empty base", Target{}, Expect{Commit: commitA}},
		{"no scheme", Target{BaseURL: "127.0.0.1:8080"}, Expect{Commit: commitA}},
		{"ftp scheme", Target{BaseURL: "ftp://127.0.0.1:8080"}, Expect{Commit: commitA}},
		{"path", Target{BaseURL: "http://127.0.0.1:8080/v1"}, Expect{Commit: commitA}},
		{"query", Target{BaseURL: "http://127.0.0.1:8080?x=1"}, Expect{Commit: commitA}},
		{"user info", Target{BaseURL: "http://ops:hunter2@127.0.0.1:8080"}, Expect{Commit: commitA}},
		{"query secret", Target{BaseURL: "http://127.0.0.1:8080/?api_key=hunter2"}, Expect{Commit: commitA}},
		{"unparseable with secret", Target{BaseURL: "http://ops:hunter2%zz@127.0.0.1:8080"}, Expect{Commit: commitA}},
		{"bearer header injection", Target{BaseURL: "http://127.0.0.1:8080", Bearer: "k\r\nX-Evil: 1"}, Expect{Commit: commitA}},
		{"no expected commit", Target{BaseURL: "http://127.0.0.1:8080"}, Expect{}},
		{"short expected commit", Target{BaseURL: "http://127.0.0.1:8080"}, Expect{Commit: commitA[:12]}},
		{"non-hex expected commit", Target{BaseURL: "http://127.0.0.1:8080"}, Expect{Commit: strings.Repeat("z", 40)}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res, err := c.Wait(context.Background(), tt.t, tt.e)
			if pe := asProbeError(t, err); pe.Kind != FailureInvalidTarget || res.Failure != FailureInvalidTarget || res.Attempts != 0 {
				t.Fatalf("kind=%s result=%+v, want invalid_target with no attempts", pe.Kind, res)
			}
			// A refused target is never echoed into the result or the error:
			// it may carry credentials.
			if res.BaseURL != "" || strings.Contains(err.Error(), "hunter2") {
				t.Fatalf("result %+v / error %q echo the refused target", res, err)
			}
		})
	}
	if n := sent.Load(); n != 0 {
		t.Fatalf("%d requests were sent for invalid input", n)
	}
}

type doerFunc func(*http.Request) (*http.Response, error)

func (f doerFunc) Do(r *http.Request) (*http.Response, error) { return f(r) }

func TestFailureKindRetryableIsExactlyNotListeningAndNotReady(t *testing.T) {
	for _, k := range []FailureKind{
		FailureInvalidTarget, FailureNotListening, FailureTransport, FailureNotReady, FailureUnauthorized, FailureBadStatus,
		FailureMalformedResponse, FailureEmptyCatalog, FailureIdentityDrift, FailureIdentityUnproven,
	} {
		want := k == FailureNotListening || k == FailureNotReady
		if got := k.Retryable(); got != want {
			t.Errorf("%s.Retryable() = %t, want %t", k, got, want)
		}
	}
}

func TestProbeResultIdentityUsesStampParser(t *testing.T) {
	id, err := stamp.ParseVersionJSON([]byte(versionDoc(commitA, false, true)))
	if err != nil {
		t.Fatal(err)
	}
	got, perr := checkIdentity([]byte(versionDoc(commitA, false, true)), commitA)
	if perr != nil || got == nil || *got != id {
		t.Fatalf("checkIdentity = %+v, %v; want the stamp parser's %+v", got, perr, id)
	}
}

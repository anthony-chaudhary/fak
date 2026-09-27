package probe

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/deploykit/stamp"
)

var identityA = stamp.Identity{AppVersion: "0.55.0", Commit: commitA, Stamped: true}

func readyOK(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = io.WriteString(w, `{"ok":true}`)
}

func do(t *testing.T, h http.Handler, method, target, auth string, body io.Reader) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(method, target, body)
	if auth != "" {
		req.Header.Set("Authorization", auth)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestMountServesHealthReadyVersionAndNoDrain(t *testing.T) {
	var readyCalls atomic.Int32
	mux := http.NewServeMux()
	err := Mount(mux, Hooks{
		Ready: func(w http.ResponseWriter, r *http.Request) {
			readyCalls.Add(1)
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusServiceUnavailable)
		},
		Version: func() stamp.Identity { return identityA },
	})
	if err != nil {
		t.Fatalf("Mount: %v", err)
	}

	if rec := do(t, mux, http.MethodGet, PathHealth, "", nil); rec.Code != http.StatusOK || strings.TrimSpace(rec.Body.String()) != `{"status":"ok"}` {
		t.Fatalf("/healthz = %d %q", rec.Code, rec.Body.String())
	}
	// The Ready hook is registered unchanged: its status and headers pass through.
	rec := do(t, mux, http.MethodGet, PathReady, "", nil)
	if rec.Code != http.StatusServiceUnavailable || rec.Header().Get("Retry-After") != "1" || readyCalls.Load() != 1 {
		t.Fatalf("/readyz = %d retry-after=%q calls=%d, want the hook's 503", rec.Code, rec.Header().Get("Retry-After"), readyCalls.Load())
	}
	rec = do(t, mux, http.MethodGet, PathVersion, "", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("/version = %d", rec.Code)
	}
	id, err := stamp.ParseVersionJSON(rec.Body.Bytes())
	if err != nil || id != identityA {
		t.Fatalf("/version parsed = %+v, %v; want %+v", id, err, identityA)
	}
	if rec := do(t, mux, http.MethodPost, PathVersion, "", nil); rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST /version = %d, want 405", rec.Code)
	}
	if rec := do(t, mux, http.MethodPost, PathDrain, "Bearer anything", nil); rec.Code != http.StatusNotFound {
		t.Fatalf("/v1/admin/drain with Drain=nil = %d, want 404 (not mounted)", rec.Code)
	}
}

// TestMountDrainRefusesWithoutAdminAuth is a negative witness: no key and a
// wrong key are both refused with 401/403, whatever the method, and the Drain
// hook is never called.
func TestMountDrainRefusesWithoutAdminAuth(t *testing.T) {
	var drains atomic.Int32
	mux := http.NewServeMux()
	if err := Mount(mux, Hooks{
		Ready: readyOK,
		Drain: func(context.Context, time.Duration) DrainResponse {
			drains.Add(1)
			return DrainResponse{Status: DrainStatusDrained}
		},
		AdminAuth: BearerAuth("admin-secret"),
	}); err != nil {
		t.Fatalf("Mount: %v", err)
	}
	tests := []struct {
		name   string
		method string
		auth   string
		want   int
	}{
		{"no key", http.MethodPost, "", http.StatusUnauthorized},
		{"no key GET", http.MethodGet, "", http.StatusUnauthorized},
		{"wrong key", http.MethodPost, "Bearer not-the-secret", http.StatusForbidden},
		{"wrong key GET", http.MethodGet, "Bearer not-the-secret", http.StatusForbidden},
		{"secret prefix", http.MethodPost, "Bearer admin-secre", http.StatusForbidden},
		{"wrong scheme", http.MethodPost, "Basic admin-secret", http.StatusForbidden},
		{"empty bearer", http.MethodPost, "Bearer ", http.StatusForbidden},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := do(t, mux, tt.method, PathDrain+"?timeout=1s", tt.auth, nil)
			if rec.Code != tt.want {
				t.Fatalf("%s %s auth=%q = %d, want %d", tt.method, PathDrain, tt.auth, rec.Code, tt.want)
			}
			if strings.Contains(rec.Body.String(), "drained") {
				t.Fatalf("refusal leaked a drain body: %s", rec.Body.String())
			}
		})
	}
	if n := drains.Load(); n != 0 {
		t.Fatalf("Drain hook called %d times without valid admin auth", n)
	}
	// X-Api-Key alone counts as a presented (and refused) credential.
	req := httptest.NewRequest(http.MethodPost, PathDrain, nil)
	req.Header.Set("X-Api-Key", "admin-secret")
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusForbidden || drains.Load() != 0 {
		t.Fatalf("X-Api-Key drain = %d (drains=%d), want 403 and no drain", rec.Code, drains.Load())
	}
}

func TestMountDrainAuthorizedWritesLiveShape(t *testing.T) {
	var gotTimeout time.Duration
	mux := http.NewServeMux()
	if err := Mount(mux, Hooks{
		Ready: readyOK,
		Drain: func(_ context.Context, timeout time.Duration) DrainResponse {
			gotTimeout = timeout
			return DrainResponse{Status: DrainStatusDrained, ElapsedMS: 12}
		},
		AdminAuth: BearerAuth("admin-secret"),
	}); err != nil {
		t.Fatalf("Mount: %v", err)
	}
	rec := do(t, mux, http.MethodPost, PathDrain+"?timeout=5s", "bearer admin-secret", nil)
	if rec.Code != http.StatusOK || gotTimeout != 5*time.Second {
		t.Fatalf("drain = %d timeout=%s, want 200 and 5s", rec.Code, gotTimeout)
	}
	// Byte-for-byte what fak-server's json.Encoder writes for the same drain.
	if got, want := rec.Body.String(), "{\"status\":\"drained\",\"active_connections\":0,\"elapsed_ms\":12}\n"; got != want {
		t.Fatalf("drain body = %q, want %q", got, want)
	}
	// Same headers as fak-server's drain reply: Content-Type only.
	if len(rec.Header()) != 1 || rec.Header().Get("Content-Type") != "application/json" {
		t.Fatalf("drain headers = %v, want only Content-Type: application/json", rec.Header())
	}
	resp, err := DecodeDrainResponse(rec.Code, rec.Body)
	if err != nil || !resp.Drained() {
		t.Fatalf("DecodeDrainResponse = %+v, %v", resp, err)
	}
	// Authorized but the wrong method: fak-server's plain-text 405, no drain.
	gotTimeout = 0
	rec = do(t, mux, http.MethodGet, PathDrain, "Bearer admin-secret", nil)
	if rec.Code != http.StatusMethodNotAllowed || rec.Header().Get("Allow") != http.MethodPost || gotTimeout != 0 ||
		rec.Body.String() != "Method Not Allowed\n" || rec.Header().Get("Content-Type") != "text/plain; charset=utf-8" {
		t.Fatalf("authorized GET drain = %d allow=%q ct=%q body=%q, want fak-server's 405 and no drain",
			rec.Code, rec.Header().Get("Allow"), rec.Header().Get("Content-Type"), rec.Body.String())
	}
}

func TestMountDrainRunsOneAtATime(t *testing.T) {
	var calls, inflight, maxInflight atomic.Int32
	release := make(chan struct{})
	mux := http.NewServeMux()
	if err := Mount(mux, Hooks{
		Ready: readyOK,
		Drain: func(context.Context, time.Duration) DrainResponse {
			n := inflight.Add(1)
			defer inflight.Add(-1)
			for {
				m := maxInflight.Load()
				if n <= m || maxInflight.CompareAndSwap(m, n) {
					break
				}
			}
			if calls.Add(1) == 1 {
				<-release
			}
			return DrainResponse{Status: DrainStatusDrained}
		},
		AdminAuth: BearerAuth("k"),
	}); err != nil {
		t.Fatal(err)
	}
	post := func(ctx context.Context) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, PathDrain, nil).WithContext(ctx)
		req.Header.Set("Authorization", "Bearer k")
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, req)
		return rec
	}

	first := make(chan *httptest.ResponseRecorder)
	go func() { first <- post(context.Background()) }()
	for calls.Load() == 0 {
		time.Sleep(time.Millisecond)
	}
	// A drain cancelled while it waits behind the first never runs.
	queuedCtx, cancelQueued := context.WithCancel(context.Background())
	queued := make(chan *httptest.ResponseRecorder)
	go func() { queued <- post(queuedCtx) }()
	// A third, live drain waits its turn and then runs alone.
	third := make(chan *httptest.ResponseRecorder)
	go func() { third <- post(context.Background()) }()
	time.Sleep(50 * time.Millisecond)
	// Cancel the queued drain while the first still holds the lock, then let
	// the first finish. A cancelled context cannot unblock the lock itself, so
	// the queued request returns only after it acquires the lock and sees the
	// cancellation.
	cancelQueued()
	close(release)
	<-queued
	if rec := <-first; rec.Code != http.StatusOK {
		t.Fatalf("first drain = %d", rec.Code)
	}
	if rec := <-third; rec.Code != http.StatusOK {
		t.Fatalf("third drain = %d", rec.Code)
	}
	if calls.Load() != 2 || maxInflight.Load() != 1 {
		t.Fatalf("drain calls=%d max concurrent=%d, want 2 calls (queued-cancelled one skipped), never concurrent", calls.Load(), maxInflight.Load())
	}
}

func TestMountDrainTimeoutMatchesLiveParsing(t *testing.T) {
	tests := []struct {
		name  string
		query string
		body  string
		want  time.Duration
	}{
		{"default", "", "", DefaultDrainTimeout},
		{"query duration", "?timeout=5s", "", 5 * time.Second},
		{"query seconds", "?timeout=7", "", 7 * time.Second},
		{"query wins over body", "?timeout=2s", `{"timeout":"9s"}`, 2 * time.Second},
		{"query unparseable", "?timeout=soon", "", DefaultDrainTimeout},
		{"query negative", "?timeout=-3s", "", DefaultDrainTimeout},
		{"query zero", "?timeout=0", "", DefaultDrainTimeout},
		{"query overflow", "?timeout=99999999999999999", "", DefaultDrainTimeout},
		// Deliberate difference: fak-server wraps this to a ~290ms window.
		{"query overflow that fak-server wraps", "?timeout=18446744074", "", DefaultDrainTimeout},
		{"body timeout_seconds overflow", "", `{"timeout_seconds":18446744074}`, DefaultDrainTimeout},
		{"body duration", "", `{"timeout":"3s"}`, 3 * time.Second},
		{"body seconds string", "", `{"timeout":"4"}`, 4 * time.Second},
		{"body timeout_seconds", "", `{"timeout_seconds":6}`, 6 * time.Second},
		{"body timeout wins", "", `{"timeout":"8s","timeout_seconds":6}`, 8 * time.Second},
		{"body garbage", "", `not json`, DefaultDrainTimeout},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got time.Duration
			mux := http.NewServeMux()
			if err := Mount(mux, Hooks{
				Ready: readyOK,
				Drain: func(_ context.Context, d time.Duration) DrainResponse {
					got = d
					return DrainResponse{Status: DrainStatusDrained}
				},
				AdminAuth: func(*http.Request) bool { return true },
			}); err != nil {
				t.Fatal(err)
			}
			var body io.Reader
			if tt.body != "" {
				body = strings.NewReader(tt.body)
			}
			if rec := do(t, mux, http.MethodPost, PathDrain+tt.query, "", body); rec.Code != http.StatusOK {
				t.Fatalf("drain = %d", rec.Code)
			}
			if got != tt.want {
				t.Fatalf("timeout = %s, want %s", got, tt.want)
			}
		})
	}
}

func TestMountRejectsInvalidHooksAndLeavesMuxUnchanged(t *testing.T) {
	drain := func(context.Context, time.Duration) DrainResponse { return DrainResponse{} }
	tests := []struct {
		name string
		mux  *http.ServeMux
		h    Hooks
	}{
		{"nil mux", nil, Hooks{Ready: readyOK}},
		{"nil ready", http.NewServeMux(), Hooks{}},
		{"drain without admin auth", http.NewServeMux(), Hooks{Ready: readyOK, Drain: drain}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := Mount(tt.mux, tt.h); !errors.Is(err, ErrInvalidHooks) {
				t.Fatalf("Mount error = %v, want ErrInvalidHooks", err)
			}
			if tt.mux == nil {
				return
			}
			for _, p := range []string{PathHealth, PathReady, PathVersion, PathDrain} {
				if serves(tt.mux, p) {
					t.Errorf("failed Mount still registered %s", p)
				}
			}
		})
	}
}

func TestMountRefusesTakenRouteAtomically(t *testing.T) {
	for _, taken := range []string{
		PathReady, PathVersion, PathDrain, "GET " + PathReady,
		// Method-only patterns: a GET probe alone would miss these, and an
		// unauthenticated "POST /v1/admin/drain" would then shadow Mount's.
		"POST " + PathDrain, "POST " + PathReady, "PUT " + PathVersion, "DELETE " + PathDrain,
	} {
		t.Run(taken, func(t *testing.T) {
			mux := http.NewServeMux()
			mux.HandleFunc(taken, readyOK)
			err := Mount(mux, Hooks{
				Ready:     readyOK,
				Drain:     func(context.Context, time.Duration) DrainResponse { return DrainResponse{} },
				AdminAuth: BearerAuth("k"),
			})
			if !errors.Is(err, ErrRouteTaken) {
				t.Fatalf("Mount over %q = %v, want ErrRouteTaken", taken, err)
			}
			if serves(mux, PathHealth) {
				t.Fatal("failed Mount still registered /healthz")
			}
		})
	}
}

func TestMountKeepsHostLiveness(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc(PathHealth, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{"ok":true,"host":"own"}`)
	})
	if err := Mount(mux, Hooks{Ready: readyOK}); err != nil {
		t.Fatalf("Mount with a host /healthz: %v", err)
	}
	if rec := do(t, mux, http.MethodGet, PathHealth, "", nil); rec.Body.String() != `{"ok":true,"host":"own"}` {
		t.Fatalf("/healthz = %q, want the host's own handler kept", rec.Body.String())
	}
}

func TestMountDefaultVersionIsSelf(t *testing.T) {
	mux := http.NewServeMux()
	if err := Mount(mux, Hooks{Ready: readyOK}); err != nil {
		t.Fatal(err)
	}
	var got stamp.Identity
	if err := json.Unmarshal(do(t, mux, http.MethodGet, PathVersion, "", nil).Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if want := SelfVersion(); got != want {
		t.Fatalf("/version = %+v, want SelfVersion %+v", got, want)
	}
}

func TestSelfVersionIsHonest(t *testing.T) {
	id := SelfVersion()
	if id.Stamped != (id.Commit != "") || id.Commit != strings.ToLower(id.Commit) {
		t.Fatalf("SelfVersion = %+v: stamped must mean a commit is present, lower-cased", id)
	}
	raw, err := json.Marshal(id)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := stamp.ParseVersionJSON(raw)
	if err != nil && !errors.Is(err, stamp.ErrInvalidCommit) {
		t.Fatalf("SelfVersion document does not parse: %v (%s)", err, raw)
	}
	if err == nil && parsed != id {
		t.Fatalf("parsed %+v, want %+v", parsed, id)
	}
}

func TestBearerAuth(t *testing.T) {
	req := func(h string) *http.Request {
		r := httptest.NewRequest(http.MethodPost, PathDrain, nil)
		if h != "" {
			r.Header.Set("Authorization", h)
		}
		return r
	}
	if BearerAuth("")(req("Bearer ")) || BearerAuth("  ")(req("Bearer   ")) {
		t.Fatal("an empty key must admit nothing")
	}
	auth := BearerAuth("s3cret")
	for h, want := range map[string]bool{
		"Bearer s3cret": true, "bearer s3cret": true, "BEARER  s3cret ": true,
		"": false, "s3cret": false, "Bearer": false, "Bearer s3cre": false, "Bearer s3cret2": false, "Token s3cret": false,
	} {
		if got := auth(req(h)); got != want {
			t.Errorf("BearerAuth(%q) = %t, want %t", h, got, want)
		}
	}
}

// TestMountAndClientShareOneContract mounts the server half and checks it
// with the client half over real HTTP: the updater's view of ready is exactly
// what Mount serves.
func TestMountAndClientShareOneContract(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc(PathModels, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"object":"list","data":[{"id":"fak","object":"model"}]}`)
	})
	var ready atomic.Bool
	if err := Mount(mux, Hooks{
		Ready: func(w http.ResponseWriter, _ *http.Request) {
			if !ready.Load() {
				w.WriteHeader(http.StatusServiceUnavailable)
				return
			}
			w.WriteHeader(http.StatusOK)
		},
		Version: func() stamp.Identity { return identityA },
	}); err != nil {
		t.Fatal(err)
	}
	base := serve(t, mux)
	c := fastClient()

	if _, err := c.Probe(context.Background(), Target{BaseURL: base}, Expect{Commit: commitA}); asProbeError(t, err).Kind != FailureNotReady {
		t.Fatalf("probe before ready = %v, want not_ready", err)
	}
	time.AfterFunc(20*time.Millisecond, func() { ready.Store(true) })
	res, err := c.Wait(context.Background(), Target{BaseURL: base}, Expect{Commit: commitA})
	if err != nil || !res.Ready || res.Identity == nil || *res.Identity != identityA {
		t.Fatalf("Wait = %+v, %v; want ready with identity %s", res, err, commitA)
	}
	_, err = c.Wait(context.Background(), Target{BaseURL: base}, Expect{Commit: commitB})
	if asProbeError(t, err).Kind != FailureIdentityDrift {
		t.Fatalf("Wait for another commit = %v, want identity_drift", err)
	}
}

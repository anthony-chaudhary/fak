package gateway

// a2a_pushnotify_test.go — contract tests for the A2A pushNotificationConfig surface and
// its best-effort webhook delivery (a2a_pushnotify.go). They pin the four load-bearing
// properties: (1) the outbound body is a re-fetch pointer with NO fak-asserted terminal
// status field; (2) delivery rides an allowlist/SSRF admission floor and a non-admissible
// target is refused with the closed reason and NEVER dialed; (3) a witnessed terminal
// transition fires exactly one POST carrying that pointer body; (4) the per-POST overhead
// is metered against a declared budget and names the closed OVERHEAD_BUDGET_EXCEEDED token.

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/turntaxmeter"
)

// resetA2APushStoreForTest swaps in a fresh push store so a test that mutates the singleton
// (allowlist, client, configs) does not leak into another. Same-package test-only helper.
func resetA2APushStoreForTest() *a2aPushStore {
	a2aPushMu.Lock()
	defer a2aPushMu.Unlock()
	a2aPushSingleton = newA2APushStore()
	return a2aPushSingleton
}

// TestA2APushNotificationHasNoClaimedField reflectively walks the outbound webhook body type
// and fails if any field (by Go name or json tag) asserts a terminal status — claimed,
// completed, success, status, or state. The contract: the receiver is handed a re-fetch
// POINTER, never a fak-asserted completion (the same distrust the relay VerifiedProgress
// fold enforces, projected onto the push body).
func TestA2APushNotificationHasNoClaimedField(t *testing.T) {
	banned := map[string]bool{"claimed": true, "completed": true, "success": true, "status": true, "state": true}
	visited := map[reflect.Type]bool{}
	var walk func(typ reflect.Type, path string)
	walk = func(typ reflect.Type, path string) {
		for typ.Kind() == reflect.Ptr || typ.Kind() == reflect.Slice || typ.Kind() == reflect.Array || typ.Kind() == reflect.Map {
			typ = typ.Elem()
		}
		if typ.Kind() != reflect.Struct || visited[typ] {
			return
		}
		visited[typ] = true
		for i := 0; i < typ.NumField(); i++ {
			f := typ.Field(i)
			name := strings.ToLower(f.Name)
			tag := strings.ToLower(strings.Split(f.Tag.Get("json"), ",")[0])
			if banned[name] || banned[tag] {
				t.Errorf("%s.%s (json=%q): push notification body must not carry a fak-asserted terminal-status field; it is a re-fetch pointer only", path, f.Name, tag)
			}
			walk(f.Type, path+"."+f.Name)
		}
	}
	walk(reflect.TypeOf(a2aPushNotification{}), "a2aPushNotification")
}

func TestA2AWebhookAdmit(t *testing.T) {
	allow := map[string]bool{"hooks.example.com": true, "127.0.0.1": true}
	cases := []struct {
		name          string
		url           string
		allowLoopback bool
		wantAdmit     bool
	}{
		{"allowlisted https host", "https://hooks.example.com/cb", false, true},
		{"non-allowlisted host refused", "https://evil.example.com/cb", false, false},
		{"non-http scheme refused", "ftp://hooks.example.com/cb", false, false},
		{"malformed url refused", "://not a url", false, false},
		{"empty host refused", "https:///cb", false, false},
		{"loopback IP refused without opt-in", "http://127.0.0.1:9000/cb", false, false},
		{"loopback IP admitted with opt-in", "http://127.0.0.1:9000/cb", true, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			u, reason := a2aWebhookAdmit(tc.url, allow, tc.allowLoopback)
			if tc.wantAdmit {
				if reason != "" || u == nil {
					t.Fatalf("admit(%q) = reason %q, want admitted", tc.url, reason)
				}
				return
			}
			if reason != a2aReasonWebhookNotAllowlisted {
				t.Fatalf("admit(%q) reason = %q, want %q", tc.url, reason, a2aReasonWebhookNotAllowlisted)
			}
			if u != nil {
				t.Fatalf("admit(%q) returned a URL on refusal", tc.url)
			}
		})
	}
}

// countingRT is an http.RoundTripper that records how many times it was invoked, so a test
// can prove a refused delivery NEVER dials.
type countingRT struct{ n int32 }

func (c *countingRT) RoundTrip(*http.Request) (*http.Response, error) {
	atomic.AddInt32(&c.n, 1)
	return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader("")), Header: make(http.Header)}, nil
}

func TestA2APushDeliverRefusesNonAllowlistedNeverDials(t *testing.T) {
	rt := &countingRT{}
	ps := newA2APushStore()
	ps.client = &http.Client{Transport: rt}
	ps.allow = map[string]bool{"hooks.example.com": true} // evil.example.com is NOT allowlisted
	ps.configs["task-x"] = a2aPushConfig{URL: "https://evil.example.com/cb"}

	fired, reason := ps.deliver("task-x", a2aPushNotification{TaskID: "task-x"}, nil)
	if fired {
		t.Fatal("deliver reported fired=true for a non-allowlisted target")
	}
	if reason != a2aReasonWebhookNotAllowlisted {
		t.Fatalf("deliver reason = %q, want %q", reason, a2aReasonWebhookNotAllowlisted)
	}
	if got := atomic.LoadInt32(&rt.n); got != 0 {
		t.Fatalf("outbound client was dialed %d time(s) for a refused delivery, want 0 (refuse, don't guess)", got)
	}
	if atomic.LoadUint64(&ps.refused) != 1 {
		t.Fatalf("refused counter = %d, want 1", ps.refused)
	}
}

func TestA2APushDeliverFiresOnePointerPost(t *testing.T) {
	var count int32
	var mu sync.Mutex
	var lastBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&count, 1)
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		lastBody = b
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	ps := newA2APushStore()
	ps.client = srv.Client()
	ps.allow = map[string]bool{"127.0.0.1": true}
	ps.allowLoopback = true
	ps.configs["task-y"] = a2aPushConfig{URL: srv.URL + "/cb"}

	note := a2aPushNotification{TaskID: "task-y", RefetchURL: "/a2a/v1/tasks/task-y"}
	fired, reason := ps.deliver("task-y", note, nil)
	if !fired || reason != "" {
		t.Fatalf("deliver = (fired %v, reason %q), want (true, \"\")", fired, reason)
	}
	if got := atomic.LoadInt32(&count); got != 1 {
		t.Fatalf("webhook received %d POST(s), want exactly 1", got)
	}
	mu.Lock()
	body := lastBody
	mu.Unlock()
	var decoded map[string]interface{}
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("decode webhook body: %v (%s)", err, body)
	}
	if decoded["task_id"] != "task-y" || decoded["refetch_url"] != "/a2a/v1/tasks/task-y" {
		t.Fatalf("webhook body missing re-fetch pointer: %v", decoded)
	}
	if _, ok := decoded["progress"]; !ok {
		t.Fatalf("webhook body missing verified-progress cursor: %v", decoded)
	}
	for _, banned := range []string{"claimed", "completed", "success", "state", "status"} {
		if _, ok := decoded[banned]; ok {
			t.Fatalf("webhook body carries a fak-asserted terminal-status field %q: %v", banned, decoded)
		}
	}
}

func TestA2APushMeterOverheadBudget(t *testing.T) {
	if breach, _ := a2aCheckWebhookOverhead(1 * time.Millisecond); breach {
		t.Fatal("a fast POST must not breach the overhead budget")
	}
	breach, reason := a2aCheckWebhookOverhead(6 * time.Second)
	if !breach {
		t.Fatal("a POST past the declared envelope must breach")
	}
	if reason != turntaxmeter.OverheadBudgetExceeded {
		t.Fatalf("breach token = %q, want %q (must be the meter-the-meter closed token)", reason, turntaxmeter.OverheadBudgetExceeded)
	}
}

// postA2AMessageWithCaller mirrors postA2AMessage (a2a_cancel_session_test.go) but lets the
// test choose the caller id, so a push-config owner-scoping assertion has a second identity.
func postA2AMessageOwned(t *testing.T, srv *Server, caller string, content map[string]interface{}) string {
	t.Helper()
	body, _ := json.Marshal(map[string]interface{}{"message_id": "m-push", "from": caller, "content": content})
	req := httptest.NewRequest(http.MethodPost, "/a2a/v1/messages", bytes.NewReader(body))
	req.Header.Set("X-Caller-ID", caller)
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusCreated {
		t.Fatalf("send-message status = %d, body = %s", w.Code, w.Body.String())
	}
	var resp map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode send-message response: %v", err)
	}
	id, _ := resp["task_id"].(string)
	if id == "" {
		t.Fatalf("no task_id: %v", resp)
	}
	// Keep the shared singleton task store neutral: drop the task (and its push config)
	// this test created so order-dependent whole-store assertions elsewhere are unaffected.
	t.Cleanup(func() {
		store := getA2AStore()
		store.mu.Lock()
		delete(store.tasks, id)
		store.mu.Unlock()
	})
	return id
}

func TestA2APushConfigSetGetRoundTrip(t *testing.T) {
	resetA2APushStoreForTest()
	srv := newTestServer(t)
	taskID := postA2AMessageOwned(t, srv, "peer-agent", map[string]interface{}{"method": "laptop.status"})

	// Set a (syntactically valid) webhook.
	setBody, _ := json.Marshal(map[string]interface{}{"url": "https://hooks.example.com/cb"})
	setReq := httptest.NewRequest(http.MethodPost, "/a2a/v1/tasks/"+taskID+"/pushNotificationConfig", bytes.NewReader(setBody))
	setReq.Header.Set("X-Caller-ID", "peer-agent")
	setW := httptest.NewRecorder()
	srv.Handler().ServeHTTP(setW, setReq)
	if setW.Code != http.StatusOK {
		t.Fatalf("set pushNotificationConfig = %d, body = %s", setW.Code, setW.Body.String())
	}

	// Get it back.
	getReq := httptest.NewRequest(http.MethodGet, "/a2a/v1/tasks/"+taskID+"/pushNotificationConfig", nil)
	getReq.Header.Set("X-Caller-ID", "peer-agent")
	getW := httptest.NewRecorder()
	srv.Handler().ServeHTTP(getW, getReq)
	if getW.Code != http.StatusOK {
		t.Fatalf("get pushNotificationConfig = %d, body = %s", getW.Code, getW.Body.String())
	}
	var got map[string]interface{}
	if err := json.NewDecoder(getW.Body).Decode(&got); err != nil {
		t.Fatalf("decode get: %v", err)
	}
	if got["url"] != "https://hooks.example.com/cb" {
		t.Fatalf("round-tripped url = %v, want the registered webhook", got["url"])
	}

	// A different caller may not read another caller's task webhook.
	otherReq := httptest.NewRequest(http.MethodGet, "/a2a/v1/tasks/"+taskID+"/pushNotificationConfig", nil)
	otherReq.Header.Set("X-Caller-ID", "someone-else")
	otherW := httptest.NewRecorder()
	srv.Handler().ServeHTTP(otherW, otherReq)
	if otherW.Code != http.StatusForbidden {
		t.Fatalf("cross-caller get = %d, want 403", otherW.Code)
	}
}

func TestA2APushConfigSetRefusesMalformedURL(t *testing.T) {
	resetA2APushStoreForTest()
	srv := newTestServer(t)
	taskID := postA2AMessageOwned(t, srv, "peer-agent", map[string]interface{}{"method": "laptop.status"})

	setBody, _ := json.Marshal(map[string]interface{}{"url": "ftp://nope"})
	setReq := httptest.NewRequest(http.MethodPost, "/a2a/v1/tasks/"+taskID+"/pushNotificationConfig", bytes.NewReader(setBody))
	setReq.Header.Set("X-Caller-ID", "peer-agent")
	setW := httptest.NewRecorder()
	srv.Handler().ServeHTTP(setW, setReq)
	if setW.Code != http.StatusBadRequest {
		t.Fatalf("set malformed url = %d, want 400", setW.Code)
	}
	var env map[string]map[string]interface{}
	if err := json.NewDecoder(setW.Body).Decode(&env); err != nil {
		t.Fatalf("decode error envelope: %v", err)
	}
	if code, _ := env["error"]["code"].(string); code != a2aReasonWebhookNotAllowlisted {
		t.Fatalf("error code = %q, want %q", code, a2aReasonWebhookNotAllowlisted)
	}
}

// TestA2ACancelFiresWebhookExactlyOnce drives a REAL session-bound cancel (as the cancel
// tests do) and asserts the witnessed terminal transition fires exactly one pointer POST at
// the registered webhook.
func TestA2ACancelFiresWebhookExactlyOnce(t *testing.T) {
	var count int32
	var mu sync.Mutex
	var lastBody []byte
	hook := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&count, 1)
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		lastBody = b
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	}))
	defer hook.Close()

	ps := resetA2APushStoreForTest()
	ps.allow = map[string]bool{"127.0.0.1": true}
	ps.allowLoopback = true
	ps.client = hook.Client()

	srv := newTestServer(t)
	srv.controlSession = func(_ context.Context, traceID, _ string, _ SessionControlRequest) (SessionState, bool, error) {
		return SessionState{TraceID: traceID, Run: "draining", Rev: 2}, true, nil
	}

	taskID := postA2AMessageOwned(t, srv, "peer-agent", map[string]interface{}{
		"method":     "laptop.status",
		"session_id": "sess-push-1",
	})

	// Register the webhook before the transition.
	setBody, _ := json.Marshal(map[string]interface{}{"url": hook.URL + "/cb"})
	setReq := httptest.NewRequest(http.MethodPost, "/a2a/v1/tasks/"+taskID+"/pushNotificationConfig", bytes.NewReader(setBody))
	setReq.Header.Set("X-Caller-ID", "peer-agent")
	setW := httptest.NewRecorder()
	srv.Handler().ServeHTTP(setW, setReq)
	if setW.Code != http.StatusOK {
		t.Fatalf("set pushNotificationConfig = %d, body = %s", setW.Code, setW.Body.String())
	}

	// Drive the witnessed transition: cancel the real session.
	cancelReq := httptest.NewRequest(http.MethodPost, "/a2a/v1/tasks/"+taskID+"/cancel", nil)
	cancelReq.Header.Set("X-Caller-ID", "peer-agent")
	cancelW := httptest.NewRecorder()
	srv.Handler().ServeHTTP(cancelW, cancelReq)
	if cancelW.Code != http.StatusOK {
		t.Fatalf("cancel = %d, body = %s", cancelW.Code, cancelW.Body.String())
	}

	if got := atomic.LoadInt32(&count); got != 1 {
		t.Fatalf("webhook received %d POST(s) for one witnessed transition, want exactly 1", got)
	}
	mu.Lock()
	body := lastBody
	mu.Unlock()
	var decoded map[string]interface{}
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("decode webhook body: %v (%s)", err, body)
	}
	if decoded["task_id"] != taskID {
		t.Fatalf("webhook body task_id = %v, want %s", decoded["task_id"], taskID)
	}
	for _, banned := range []string{"claimed", "completed", "success", "state", "status"} {
		if _, ok := decoded[banned]; ok {
			t.Fatalf("webhook body carries a fak-asserted terminal-status field %q: %v", banned, decoded)
		}
	}
}

package gateway

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func authenticatedResponsesCall(t *testing.T, method, url, key string) (int, []byte) {
	t.Helper()
	req, err := http.NewRequest(method, url, nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+key)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return resp.StatusCode, raw
}

func newRetrieveFixture(t *testing.T) (*Server, *httptest.Server) {
	t.Helper()
	planner := &continuationPlanner{completions: []*agent.Completion{
		{Message: agent.Message{Role: agent.RoleAssistant, Content: "stored answer"}, FinishReason: "stop"},
	}}
	srv := newContinuationServer(t, planner, map[string]string{"alice-key": "alice", "bob-key": "bob"})
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return srv, ts
}

func TestResponsesRetrieveReturnsStoredSnapshot(t *testing.T) {
	_, ts := newRetrieveFixture(t)
	code, created := postAuthenticatedResponse(t, ts.URL, "alice-key", map[string]any{"model": "fixture", "input": "hi"})
	if code != http.StatusOK {
		t.Fatalf("create status = %d: %s", code, created)
	}
	id := responseID(t, created)

	code, got := authenticatedResponsesCall(t, http.MethodGet, ts.URL+"/v1/responses/"+id, "alice-key")
	if code != http.StatusOK {
		t.Fatalf("retrieve status = %d: %s", code, got)
	}
	var want, have map[string]any
	if err := json.Unmarshal(created, &want); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(got, &have); err != nil {
		t.Fatalf("decode snapshot: %v: %s", err, got)
	}
	if have["id"] != id || have["object"] != "response" || have["status"] != "completed" || have["output_text"] != "stored answer" {
		t.Fatalf("snapshot = %s", got)
	}
	if want["created_at"] != have["created_at"] || want["model"] != have["model"] {
		t.Fatalf("snapshot diverges from served body:\nserved=%s\nstored=%s", created, got)
	}
}

func TestResponsesRetrieveMissesShareOneNotFoundShape(t *testing.T) {
	_, ts := newRetrieveFixture(t)
	_, created := postAuthenticatedResponse(t, ts.URL, "alice-key", map[string]any{"model": "fixture", "input": "hi"})
	id := responseID(t, created)
	_, unstored := postAuthenticatedResponse(t, ts.URL, "alice-key", map[string]any{"model": "fixture", "input": "hi", "store": false})
	unstoredID := responseID(t, unstored)

	cases := map[string]struct{ path, key string }{
		"unknown":     {"/v1/responses/resp_fak_missing", "alice-key"},
		"wrong owner": {"/v1/responses/" + id, "bob-key"},
		"store false": {"/v1/responses/" + unstoredID, "alice-key"},
		"nested path": {"/v1/responses/" + id + "/input_items", "alice-key"},
	}
	var shape []byte
	for name, tc := range cases {
		code, body := authenticatedResponsesCall(t, http.MethodGet, ts.URL+tc.path, tc.key)
		if code != http.StatusNotFound {
			t.Fatalf("%s: status = %d: %s", name, code, body)
		}
		if shape == nil {
			shape = body
		} else if string(shape) != string(body) {
			t.Fatalf("%s: 404 body %s differs from %s", name, body, shape)
		}
	}
	// Negative control: the owner still reads the stored response.
	if code, body := authenticatedResponsesCall(t, http.MethodGet, ts.URL+"/v1/responses/"+id, "alice-key"); code != http.StatusOK {
		t.Fatalf("owner retrieve status = %d: %s", code, body)
	}
}

func TestResponsesRetrieveRejectsOtherMethods(t *testing.T) {
	_, ts := newRetrieveFixture(t)
	for _, method := range []string{http.MethodPost, http.MethodPut, http.MethodPatch} {
		if code, body := authenticatedResponsesCall(t, method, ts.URL+"/v1/responses/resp_fak_any", "alice-key"); code != http.StatusMethodNotAllowed {
			t.Fatalf("%s status = %d: %s", method, code, body)
		}
	}
}

func TestResponsesRetrieveRequiresAuth(t *testing.T) {
	_, ts := newRetrieveFixture(t)
	resp, err := http.Get(ts.URL + "/v1/responses/resp_fak_any")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("unauthenticated status = %d", resp.StatusCode)
	}
}

func TestResponsesDeleteRemovesOwnedResponse(t *testing.T) {
	_, ts := newRetrieveFixture(t)
	_, created := postAuthenticatedResponse(t, ts.URL, "alice-key", map[string]any{"model": "fixture", "input": "hi"})
	id := responseID(t, created)

	if code, body := authenticatedResponsesCall(t, http.MethodDelete, ts.URL+"/v1/responses/"+id, "bob-key"); code != http.StatusNotFound {
		t.Fatalf("wrong-owner delete status = %d: %s", code, body)
	}
	code, body := authenticatedResponsesCall(t, http.MethodDelete, ts.URL+"/v1/responses/"+id, "alice-key")
	if code != http.StatusOK {
		t.Fatalf("delete status = %d: %s", code, body)
	}
	var deleted responsesDeleted
	if err := json.Unmarshal(body, &deleted); err != nil || deleted != (responsesDeleted{ID: id, Object: "response", Deleted: true}) {
		t.Fatalf("delete body = %s (%v)", body, err)
	}
	if code, body := authenticatedResponsesCall(t, http.MethodGet, ts.URL+"/v1/responses/"+id, "alice-key"); code != http.StatusNotFound {
		t.Fatalf("retrieve after delete status = %d: %s", code, body)
	}
	if code, body := authenticatedResponsesCall(t, http.MethodDelete, ts.URL+"/v1/responses/"+id, "alice-key"); code != http.StatusNotFound {
		t.Fatalf("second delete status = %d: %s", code, body)
	}
	code, body = postAuthenticatedResponse(t, ts.URL, "alice-key", map[string]any{"model": "fixture", "input": "again", "previous_response_id": id})
	if code != http.StatusBadRequest {
		t.Fatalf("continuation from deleted id status = %d: %s", code, body)
	}
}

func TestResponsesRetrieveStoreExpiryAndAccounting(t *testing.T) {
	store := newResponsesContinuationStore(responsesContinuationStoreConfig{MaxEntries: 4, MaxBytes: 1 << 20, TTL: time.Minute})
	now := time.Now()
	msgs := []agent.Message{{Role: agent.RoleAssistant, Content: "x"}}
	if !store.PutResponse("r1", "alice", msgs, []byte(`{"id":"r1"}`), now) {
		t.Fatal("put failed")
	}
	if !store.Put("r2", "alice", msgs, now) {
		t.Fatal("put without snapshot failed")
	}
	if _, ok := store.Snapshot("r2", "alice", now); ok {
		t.Fatal("entry stored without a snapshot must not be retrievable")
	}
	if _, ok := store.Get("r2", "alice", now); !ok {
		t.Fatal("continuation without snapshot must still be addressable")
	}
	got, ok := store.Snapshot("r1", "alice", now)
	if !ok || string(got) != `{"id":"r1"}` {
		t.Fatalf("snapshot = %q, %v", got, ok)
	}
	got[0] = 'X'
	if again, _ := store.Snapshot("r1", "alice", now); string(again) != `{"id":"r1"}` {
		t.Fatalf("snapshot aliased store memory: %q", again)
	}
	if _, ok := store.Snapshot("r1", "alice", now.Add(2*time.Minute)); ok {
		t.Fatal("expired snapshot retrieved")
	}
	if store.Delete("r1", "alice", now) {
		t.Fatal("expired entry deleted")
	}
	before := store.bytes
	if !store.PutResponse("r3", "alice", msgs, []byte(`{}`), now.Add(2*time.Minute)) || !store.Delete("r3", "alice", now.Add(2*time.Minute)) {
		t.Fatal("put/delete r3 failed")
	}
	if store.bytes > before || store.Len() != 0 {
		t.Fatalf("bytes=%d before=%d len=%d after delete", store.bytes, before, store.Len())
	}
}

func TestResponsesRetrieveStoreDropsOversizedSnapshotKeepsContinuation(t *testing.T) {
	msgs := []agent.Message{{Role: agent.RoleAssistant, Content: "x"}}
	limit := responsesMessagesBytes(msgs) + 4
	store := newResponsesContinuationStore(responsesContinuationStoreConfig{MaxEntries: 4, MaxBytes: limit, TTL: time.Minute})
	now := time.Now()
	if !store.PutResponse("big", "alice", msgs, []byte(`{"id":"big"}`), now) {
		t.Fatal("continuation refused because of its snapshot")
	}
	if _, ok := store.Get("big", "alice", now); !ok {
		t.Fatal("continuation lost")
	}
	if _, ok := store.Snapshot("big", "alice", now); ok {
		t.Fatal("over-cap snapshot was stored")
	}
	if store.bytes > limit {
		t.Fatalf("bytes=%d exceed cap %d", store.bytes, limit)
	}
	// Negative control: a snapshot that fits is kept.
	if !store.PutResponse("small", "alice", msgs, []byte(`{}`), now) {
		t.Fatal("put small failed")
	}
	if got, ok := store.Snapshot("small", "alice", now); !ok || string(got) != `{}` {
		t.Fatalf("small snapshot = %q, %v", got, ok)
	}
}

func TestResponsesRetrieveMetricsRouteIsBounded(t *testing.T) {
	if got := routeForMetrics("/v1/responses/resp_fak_0123456789abcdef"); got != "/v1/responses/*" {
		t.Fatalf("routeForMetrics = %q", got)
	}
	if got := routeForMetrics("/v1/responses"); got != "/v1/responses" {
		t.Fatalf("routeForMetrics base = %q", got)
	}
}

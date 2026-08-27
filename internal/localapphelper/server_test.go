package localapphelper

import (
	"context"
	"encoding/json"
	"github.com/anthony-chaudhary/fak/internal/localappcontract"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

type execFunc func(context.Context, TaskRequest) (TaskResult, error)

func (f execFunc) Execute(c context.Context, r TaskRequest) (TaskResult, error) { return f(c, r) }
func testServer(t *testing.T, e Executor) (*Server, string) {
	t.Helper()
	h := HostIdentity{TeamID: "T", BundleID: "B", InstallID: "I", HelperBuild: "R"}
	cap := strings.Repeat("x", 32)
	b, err := Bind(h, []byte(cap))
	if err != nil {
		t.Fatal(err)
	}
	return &Server{Binding: b, Host: h, Capability: []byte(cap), Executor: e}, cap
}
func TestServerAuthenticatesAndRequiresFakNativeReceipt(t *testing.T) {
	e := execFunc(func(_ context.Context, r TaskRequest) (TaskResult, error) {
		return TaskResult{Events: []localappcontract.Event{{Schema: localappcontract.Schema, Sequence: 1, TaskID: r.TaskID, Kind: "completed"}}, Receipt: localappcontract.Receipt{Schema: localappcontract.Schema, TaskID: r.TaskID, Engine: "fak-native", Location: "local", Revision: "r1", Quality: "pass", Attempts: 1, Authority: "app", Terminal: localappcontract.Completed}}, nil
	})
	s, cap := testServer(t, e)
	h, err := s.Handler()
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(h)
	defer ts.Close()
	body := `{"schema":"fak.local-app-contract/1","task_id":"t1","task":"job-apply","payload":{}}`
	req, _ := http.NewRequest("POST", ts.URL+"/v1/tasks", strings.NewReader(body))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 401 {
		t.Fatalf("without capability=%d", resp.StatusCode)
	}
	resp.Body.Close()
	req, _ = http.NewRequest("POST", ts.URL+"/v1/tasks", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+cap)
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("with capability=%d", resp.StatusCode)
	}
	var got TaskResult
	if json.NewDecoder(resp.Body).Decode(&got) != nil || got.Receipt.Engine != "fak-native" {
		t.Fatal("missing native receipt")
	}
}
func TestCancelReachesExecutor(t *testing.T) {
	started := make(chan struct{})
	stopped := make(chan struct{})
	e := execFunc(func(ctx context.Context, r TaskRequest) (TaskResult, error) {
		close(started)
		<-ctx.Done()
		close(stopped)
		return TaskResult{}, ctx.Err()
	})
	s, cap := testServer(t, e)
	h, _ := s.Handler()
	ts := httptest.NewServer(h)
	defer ts.Close()
	go func() {
		req, _ := http.NewRequest("POST", ts.URL+"/v1/tasks", strings.NewReader(`{"schema":"fak.local-app-contract/1","task_id":"t1","task":"x","payload":{}}`))
		req.Header.Set("Authorization", "Bearer "+cap)
		http.DefaultClient.Do(req)
	}()
	<-started
	req, _ := http.NewRequest("DELETE", ts.URL+"/v1/tasks/t1", nil)
	req.Header.Set("Authorization", "Bearer "+cap)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 204 {
		t.Fatalf("cancel=%d", resp.StatusCode)
	}
	select {
	case <-stopped:
	case <-time.After(time.Second):
		t.Fatal("executor not cancelled")
	}
}
func TestListenLocalRejectsLAN(t *testing.T) {
	if _, err := ListenLocal("0.0.0.0:0"); err == nil {
		t.Fatal("LAN listener accepted")
	}
	ln, err := ListenLocal("127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	ln.Close()
}

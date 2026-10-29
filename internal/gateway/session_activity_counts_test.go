package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

type sessionActivityResultEngine struct{}

func (sessionActivityResultEngine) Caps() []abi.Capability { return nil }
func (sessionActivityResultEngine) Complete(_ context.Context, call *abi.ToolCall) (*abi.Result, error) {
	status := abi.StatusOK
	if call.Tool == "allow-error" {
		status = abi.StatusError
	} else if call.Tool == "allow-pending" {
		status = abi.StatusPending
	}
	return &abi.Result{Call: call, Status: status, Payload: call.Args}, nil
}

// fak-test:runtime fast est=30ms lane=default
func TestSessionSyscallActivityWindowAndIsolation(t *testing.T) {
	a := newSessionActivity()
	start := time.Unix(100, 1)
	a.observeSyscallResult("old", "ALLOW", &ResultEnvelope{Status: "ERROR"}, start)
	for i := 0; i < sessionActivityCap; i++ {
		a.observeSyscallResult(strconv.Itoa(i), "ALLOW", &ResultEnvelope{Status: "OK"}, start.Add(time.Second))
	}
	old := a.actionCounts("old")
	if old.Tools != 0 || old.Coverage.Tools != ActivityCoverageUnavailable || old.Coverage.SinceUnixNano != 0 {
		t.Fatal("eviction retained a false observation window")
	}
	restart := start.Add(2 * time.Second)
	a.observeSyscallResult("old", "ALLOW", &ResultEnvelope{Status: "OK"}, restart)
	old = a.actionCounts("old")
	if old.Tools != 1 || old.Fail != 0 || old.Coverage.SinceUnixNano != restart.UnixNano() {
		t.Fatal("new observation window inherited evicted totals")
	}
	old.Coverage.Tools = "mutated-by-reader"
	if a.actionCounts("old").Coverage.Tools != ActivityCoverageNativeSyscallResults {
		t.Fatal("snapshot aliases registry state")
	}
	if len(a.rec) != sessionActivityCap {
		t.Fatalf("retained traces = %d", len(a.rec))
	}
	view, _ := a.snapshot("old", restart)
	if view.LastTool != "" || view.IdleSeconds != 0 {
		t.Fatal("syscall counters changed adjudication activity")
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestSessionSyscallActivityConcurrent(t *testing.T) {
	a := newSessionActivity()
	var wg sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 100; i++ {
				a.observeSyscallResult("trace", "ALLOW", &ResultEnvelope{Status: "ERROR"}, time.Unix(100, 1))
				_ = a.actionCounts("trace")
			}
		}()
	}
	wg.Wait()
	got := a.actionCounts("trace")
	if got.Tools != 800 || got.Fail != 800 {
		t.Fatalf("tools=%d fail=%d", got.Tools, got.Fail)
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestSessionSyscallActivityHTTP(t *testing.T) {
	srv := newTestServer(t)
	abi.RegisterEngine("test", sessionActivityResultEngine{})
	srv.listSessions = func(context.Context) []SessionState {
		return []SessionState{{TraceID: "alpha"}, {TraceID: "beta"}, {TraceID: "unseen"}}
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	post := func(path, trace, tool string) {
		t.Helper()
		body, err := json.Marshal(map[string]any{"trace_id": trace, "tool": tool, "read_only": true, "arguments": map[string]string{"value": "payload-not-for-summary"}})
		if err != nil {
			t.Fatal(err)
		}
		resp, err := http.Post(ts.URL+path, "application/json", bytes.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("request status = %d", resp.StatusCode)
		}
	}
	post("/v1/fak/syscall", "alpha", "allow-ok")
	post("/v1/fak/syscall", "alpha", "allow-error")
	post("/v1/fak/syscall", "alpha", "transform-ok")
	post("/v1/fak/syscall", "beta", "allow-ok")
	post("/v1/fak/syscall", "alpha", "deny-work")
	post("/v1/fak/syscall", "alpha", "allow-pending")
	post("/v1/fak/adjudicate", "alpha", "allow-not-executed")
	resp, err := http.Get(ts.URL + "/v1/fak/sessions")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), "payload-not-for-summary") {
		t.Fatal("summary contains a tool payload")
	}
	var got struct {
		Sessions []struct {
			TraceID  string `json:"trace_id"`
			Activity struct {
				Tools    int `json:"tools"`
				Fail     int `json:"fail"`
				Execs    int `json:"execs"`
				Effects  int `json:"effects"`
				Coverage struct {
					Tools   string `json:"tools"`
					Fail    string `json:"fail"`
					Execs   string `json:"execs"`
					Effects string `json:"effects"`
					Since   int64  `json:"since_unix_nano"`
				} `json:"coverage"`
			} `json:"activity"`
		} `json:"sessions"`
	}
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Sessions) != 3 {
		t.Fatalf("sessions = %d", len(got.Sessions))
	}
	for _, row := range got.Sessions {
		wantTools, wantFail := 0, 0
		if row.TraceID == "alpha" {
			wantTools, wantFail = 3, 1
		}
		if row.TraceID == "beta" {
			wantTools = 1
		}
		if row.Activity.Tools != wantTools || row.Activity.Fail != wantFail {
			t.Fatalf("trace %s tools=%d fail=%d; want %d/%d", row.TraceID, row.Activity.Tools, row.Activity.Fail, wantTools, wantFail)
		}
		if row.Activity.Execs != 0 || row.Activity.Effects != 0 || row.Activity.Coverage.Execs != "unavailable" || row.Activity.Coverage.Effects != "unavailable" {
			t.Fatalf("trace %s invents execution/effect coverage", row.TraceID)
		}
		wantCoverage := "native_syscall_results"
		if row.TraceID == "unseen" {
			wantCoverage = "unavailable"
		}
		if row.Activity.Coverage.Tools != wantCoverage || row.Activity.Coverage.Fail != wantCoverage {
			t.Fatalf("trace %s coverage tools=%s fail=%s", row.TraceID, row.Activity.Coverage.Tools, row.Activity.Coverage.Fail)
		}
		if row.TraceID != "unseen" && row.Activity.Coverage.Since <= 0 {
			t.Fatalf("trace %s missing observation window", row.TraceID)
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestSessionSyscallActivityProviderPrecedence(t *testing.T) {
	srv := newTestServer(t)
	provided := []SessionState{
		{TraceID: "legacy", Activity: ActionCounts{Tools: 9, Execs: 3, Fail: 4, Effects: 5}},
		{TraceID: "explicit", Activity: ActionCounts{Coverage: &ActivityCoverage{
			Tools: ActivityCoverageUnavailable, Execs: ActivityCoverageUnavailable,
			Fail: ActivityCoverageUnavailable, Effects: ActivityCoverageUnavailable,
		}}},
		{TraceID: "fallback"},
	}
	before, err := json.Marshal(provided)
	if err != nil {
		t.Fatal(err)
	}
	srv.listSessions = func(context.Context) []SessionState { return provided }
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	for _, row := range provided {
		var result SyscallResponse
		postJSON(t, ts.URL+"/v1/fak/syscall", SyscallRequest{TraceID: row.TraceID, Tool: "allow-ok", ReadOnly: true}, &result)
		if result.Result == nil || result.Result.Status != "OK" {
			t.Fatal("request produced no terminal result")
		}
	}
	resp, err := http.Get(ts.URL + "/v1/fak/sessions")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var got SessionListResponse
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if len(got.Sessions) != 3 {
		t.Fatalf("sessions=%d", len(got.Sessions))
	}
	for i := 0; i < 2; i++ {
		if !reflect.DeepEqual(got.Sessions[i].Activity, provided[i].Activity) {
			t.Fatalf("provider activity changed for %s", provided[i].TraceID)
		}
	}
	if got.Sessions[2].Activity.Tools != 1 || got.Sessions[2].Activity.Coverage == nil {
		t.Fatal("absent provider activity did not use native observations")
	}
	after, err := json.Marshal(provided)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Fatal("projection mutated provider-owned rows")
	}
}

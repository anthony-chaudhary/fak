package codexsession

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionctl"
	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

func TestAdapterResumesPersistedThreadWithNewExecutionEpoch(t *testing.T) {
	store := sessionctl.NewMemoryCodexStateStore()
	session, err := sessionctl.OpenCodexSession(store, "logical-1", "codex-cli 1.2.3")
	if err != nil {
		t.Fatal(err)
	}
	first, err := New(Config{Command: os.Args[0], Args: []string{"-test.run=TestFakeAppServer", "--"}, Workspace: t.TempDir(), Version: "codex-cli 1.2.3", RunID: "logical-1", Session: session, StartMode: sessionctl.CodexNew, InputLease: "browser-1", Sink: func(harnesskit.Envelope) error { return nil }})
	if err != nil {
		t.Fatal(err)
	}
	if err := first.Run(context.Background(), "turn one"); err != nil {
		t.Fatal(err)
	}
	second, err := New(Config{Command: os.Args[0], Args: []string{"-test.run=TestFakeAppServer", "--"}, Workspace: t.TempDir(), Version: "codex-cli 1.2.3", RunID: "logical-1", Session: session, StartMode: sessionctl.CodexResume, InputLease: "browser-2", Sink: func(harnesskit.Envelope) error { return nil }})
	if err != nil {
		t.Fatal(err)
	}
	if err := second.Run(context.Background(), "turn two"); err != nil {
		t.Fatal(err)
	}
	state := session.State()
	if state.SessionID != "logical-1" || state.Coordinates.ThreadID != "thread-raw-secret" || state.Epoch != 2 {
		t.Fatalf("state = %+v", state)
	}
	if len(state.Events) != 18 { //boundarylint:ignore CHANGE_DETECTOR_TEST the transcript fixture contains exactly 18 lifecycle events whose complete preservation is the contract
		t.Fatalf("events = %d, want two complete nine-event turns", len(state.Events))
	}
}

func TestAdapterProjectsTypedAppServerStream(t *testing.T) {
	helper := writeFakeServer(t, false)
	var got []harnesskit.Envelope
	a, err := New(Config{Command: helper, Args: []string{"-test.run=TestFakeAppServer", "--"}, Workspace: t.TempDir(), Version: "codex-cli 1.2.3", RunID: "run-1", Sink: func(e harnesskit.Envelope) error { got = append(got, e); return nil }})
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Run(context.Background(), "seeded task"); err != nil {
		t.Fatal(err)
	}
	want := []harnesskit.EventType{harnesskit.EventRunStarted, harnesskit.EventMessageStarted, harnesskit.EventMessageDelta, harnesskit.EventToolStarted, harnesskit.EventToolCompleted, harnesskit.EventMessageDelta, harnesskit.EventMessageCompleted, harnesskit.EventUsage, harnesskit.EventRunCompleted}
	if len(got) != len(want) {
		t.Fatalf("events=%d want %d: %#v", len(got), len(want), got)
	}
	for i := range want {
		if got[i].Type != want[i] {
			t.Fatalf("event %d=%s want %s", i, got[i].Type, want[i])
		}
		if got[i].Sequence != uint64(i+1) {
			t.Fatalf("sequence %d=%d", i, got[i].Sequence)
		}
	}
	b, _ := json.Marshal(got)
	if string(b) == "" || !containsAll(string(b), "engine=codex", "stdio://", "hello ", "world", "tool output", "codex-cli 1.2.3") {
		t.Fatalf("receipt missing engine/stream/tool/version: %s", b)
	}
	for _, bad := range []string{"provider_wire_type"} {
		if containsAll(string(b), bad) {
			t.Fatalf("provider identity leaked: %s", bad)
		}
	}
}

func TestAdapterRequiresVersionAndAbsoluteWorkspace(t *testing.T) {
	if _, err := New(Config{Workspace: ".", RunID: "x", Sink: func(harnesskit.Envelope) error { return nil }}); err == nil {
		t.Fatal("missing version accepted")
	}
	a, err := New(Config{Workspace: ".", Version: "v", RunID: "x", Sink: func(harnesskit.Envelope) error { return nil }})
	if err != nil {
		t.Fatal(err)
	}
	if !filepath.IsAbs(a.cfg.Workspace) {
		t.Fatalf("workspace not absolute: %s", a.cfg.Workspace)
	}
}

func containsAll(s string, parts ...string) bool {
	for _, p := range parts {
		if !stringsContains(s, p) {
			return false
		}
	}
	return true
}
func stringsContains(s, p string) bool {
	return len(p) == 0 || len(s) >= len(p) && func() bool {
		for i := 0; i+len(p) <= len(s); i++ {
			if s[i:i+len(p)] == p {
				return true
			}
		}
		return false
	}()
}

func writeFakeServer(t *testing.T, interrupt bool) string { t.Helper(); return os.Args[0] }

func TestFakeAppServer(t *testing.T) {
	if len(os.Args) < 2 || os.Args[len(os.Args)-1] != "--" {
		cmd := exec.Command(os.Args[0], "-test.run=^TestFakeAppServer$", "--")
		cmd.Stdin = bytes.NewBufferString("{\"id\":7,\"method\":\"initialize\"}\n")
		output, err := cmd.Output()
		if err != nil {
			t.Fatalf("fake app server initialize failed: %v", err)
		}
		var response struct {
			ID     int `json:"id"`
			Result struct {
				UserAgent string `json:"userAgent"`
			} `json:"result"`
		}
		if err := json.NewDecoder(bytes.NewReader(output)).Decode(&response); err != nil {
			t.Fatalf("decode initialize response %q: %v", output, err)
		}
		if response.ID != 7 || response.Result.UserAgent != "codex-cli 1.2.3" {
			t.Fatalf("initialize response = %+v", response)
		}
		return
	}
	dec := json.NewDecoder(os.Stdin)
	enc := json.NewEncoder(os.Stdout)
	for {
		var req struct {
			ID     int             `json:"id"`
			Method string          `json:"method"`
			Params json.RawMessage `json:"params"`
		}
		if err := dec.Decode(&req); err != nil {
			return
		}
		switch req.Method {
		case "initialize":
			enc.Encode(map[string]any{"jsonrpc": "2.0", "id": req.ID, "result": map[string]any{"userAgent": "codex-cli 1.2.3"}})
		case "initialized":
		case "thread/start", "thread/resume":
			enc.Encode(map[string]any{"jsonrpc": "2.0", "id": req.ID, "result": map[string]any{"thread": map[string]any{"id": "thread-raw-secret"}}})
		case "turn/start":
			enc.Encode(map[string]any{"jsonrpc": "2.0", "id": req.ID, "result": map[string]any{"turn": map[string]any{"id": "turn-raw-secret", "status": "inProgress", "items": []any{}}}})
			note := func(method string, p any) {
				enc.Encode(map[string]any{"jsonrpc": "2.0", "method": method, "params": p})
			}
			note("item/started", map[string]any{"threadId": "thread-raw-secret", "turnId": "turn-raw-secret", "item": map[string]any{"id": "msg-1", "type": "agentMessage"}})
			note("item/agentMessage/delta", map[string]any{"threadId": "thread-raw-secret", "turnId": "turn-raw-secret", "itemId": "msg-1", "delta": "hello "})
			note("item/started", map[string]any{"threadId": "thread-raw-secret", "turnId": "turn-raw-secret", "item": map[string]any{"id": "tool-1", "type": "commandExecution", "command": "printf world", "status": "inProgress"}})
			note("item/completed", map[string]any{"threadId": "thread-raw-secret", "turnId": "turn-raw-secret", "item": map[string]any{"id": "tool-1", "type": "commandExecution", "status": "completed", "aggregatedOutput": "tool output"}})
			note("item/agentMessage/delta", map[string]any{"threadId": "thread-raw-secret", "turnId": "turn-raw-secret", "itemId": "msg-1", "delta": "world"})
			note("item/completed", map[string]any{"threadId": "thread-raw-secret", "turnId": "turn-raw-secret", "item": map[string]any{"id": "msg-1", "type": "agentMessage"}})
			note("thread/tokenUsage/updated", map[string]any{"threadId": "thread-raw-secret", "turnId": "turn-raw-secret", "tokenUsage": map[string]any{"inputTokens": 7, "outputTokens": 2}})
			note("turn/completed", map[string]any{"threadId": "thread-raw-secret", "turn": map[string]any{"id": "turn-raw-secret", "status": "completed", "items": []any{}}})
			return
		default:
			fmt.Fprintln(os.Stderr, "unknown", req.Method)
			return
		}
	}
}
func TestInterruptUsesTypedRPC(t *testing.T) {
	reader, writer := io.Pipe()
	a := &Adapter{stdin: writer, threadID: "thread-1", turnID: "turn-1"}
	done := make(chan map[string]any, 1)
	go func() {
		var got map[string]any
		_ = json.NewDecoder(reader).Decode(&got)
		done <- got
	}()
	if err := a.Interrupt(); err != nil {
		t.Fatal(err)
	}
	got := <-done
	if got["method"] != "turn/interrupt" {
		t.Fatalf("method=%v", got["method"])
	}
	params := got["params"].(map[string]any)
	if params["threadId"] != "thread-1" || params["turnId"] != "turn-1" {
		t.Fatalf("params=%v", params)
	}
}

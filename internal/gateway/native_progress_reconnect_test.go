package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestNativeStreamReconnectReplaysMissedAdjudication(t *testing.T) {
	srv, err := New(Config{EngineID: "mock", Model: "test", Native: true})
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()
	const session = "resume-native"
	events := []agent.ProgressEvent{{Seq: 1, Kind: agent.ProgressTurnStarted, Turn: 1}, {Seq: 2, Kind: agent.ProgressCallAdjudicated, Turn: 1, CallID: "call-denied", Tool: "delete_account", Verdict: "DENY", Reason: "POLICY_BLOCK"}, {Seq: 3, Kind: agent.ProgressTurnDone, Turn: 1}}
	for _, ev := range events {
		srv.nativeProgress.append(session, ev)
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	body, _ := json.Marshal(map[string]any{"model": "test", "max_tokens": 16, "stream": true, "messages": []map[string]string{{"role": "user", "content": "resume"}}})
	req, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, ts.URL+"/v1/messages", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Trace-Id", session)
	req.Header.Set("Last-Event-ID", "1")
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	frames := readAnthropicSSE(t, bytes.NewReader(raw))
	if len(frames) != 2 {
		t.Fatalf("frames=%d body=%s", len(frames), raw)
	}
	for i, f := range frames {
		want := strconv.Itoa(i + 2)
		if f.id != want {
			t.Fatalf("frame %d id=%q want=%s", i, f.id, want)
		}
	}
	if frames[0].event != "call_adjudicated" {
		t.Fatalf("missed adjudication not first replay: %+v", frames)
	}
}

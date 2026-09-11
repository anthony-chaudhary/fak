package agentbench

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestAgentBenchStreamingProgress(t *testing.T) {
	t.Run("first output precedes response completion", func(t *testing.T) {
		var requests atomic.Int64
		firstFlushed := make(chan struct{})
		progressFlushed := make(chan struct{})
		release := make(chan struct{})
		var releaseOnce sync.Once
		defer releaseOnce.Do(func() { close(release) })
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			id := requests.Add(1)
			w.Header().Set("Content-Type", "text/event-stream")
			if id == 1 {
				io.WriteString(w, `data: {"model":"fixture-model","choices":[{"delta":{"content":"first"}}]}`+"\n\n")
				w.(http.Flusher).Flush()
				close(firstFlushed)
				io.WriteString(w, `data: {"model":"fixture-model","choices":[{"delta":{"content":"middle"}}]}`+"\n\n")
				w.(http.Flusher).Flush()
				close(progressFlushed)
				<-release
			}
			fmt.Fprint(w, `data: {"model":"fixture-model","choices":[{"delta":{"content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}`+"\n\ndata: [DONE]\n\n")
		}))
		defer server.Close()
		outDir := filepath.Join(t.TempDir(), "streaming")
		repo := writeAgentBenchRepo(t)
		var stdout, stderr bytes.Buffer
		done := make(chan int, 1)
		go func() {
			done <- RunReplayCLI(context.Background(), &stdout, &stderr, agentBenchArgs(repo, server.URL, outDir))
		}()
		select {
		case <-firstFlushed:
		case <-time.After(2 * time.Second):
			t.Fatal("first content was not flushed")
		}
		select {
		case <-progressFlushed:
		case <-time.After(2 * time.Second):
			t.Fatal("intermediate content was not flushed")
		}
		deadline := time.Now().Add(time.Second)
		seenFirst, seenTerminal, progressEvents := false, false, 0
		for time.Now().Before(deadline) {
			if events, err := agentBenchTryReadEvents(outDir); err == nil {
				for _, event := range events {
					name := strings.ToLower(fmt.Sprint(event["event"]))
					seenFirst = seenFirst || name == "first_output"
					seenTerminal = seenTerminal || name == "end" || name == "terminal"
					if strings.ToLower(fmt.Sprint(event["event_type"])) == "stream_progress" {
						progressEvents++
					}
				}
			}
			if seenFirst && progressEvents >= 2 {
				break
			}
			time.Sleep(10 * time.Millisecond)
		}
		if !seenFirst || progressEvents < 2 || seenTerminal {
			releaseOnce.Do(func() { close(release) })
			t.Fatalf("while response held open: first_output=%v progress=%d terminal=%v", seenFirst, progressEvents, seenTerminal)
		}
		releaseOnce.Do(func() { close(release) })
		select {
		case code := <-done:
			if code != 0 {
				t.Fatalf("replay exit %d: %s", code, stderr.String())
			}
		case <-time.After(5 * time.Second):
			t.Fatal("replay did not finish after upstream release")
		}
	})

	t.Run("fragmented tool call is useful streamed output", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, `data: {"model":"fixture-model","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"Read","arguments":"{\"file_"}}]}}]}`+"\n\n")
			w.(http.Flusher).Flush()
			io.WriteString(w, `data: {"model":"fixture-model","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"path\":\"README.md\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}`+"\n\ndata: [DONE]\n\n")
		}))
		defer server.Close()
		outDir := filepath.Join(t.TempDir(), "tool-stream")
		var stdout, stderr bytes.Buffer
		if code := RunReplayCLI(context.Background(), &stdout, &stderr, agentBenchArgs(writeAgentBenchRepo(t), server.URL, outDir)); code != 0 {
			t.Fatalf("tool-only replay exit %d: %s", code, stderr.String())
		}
		events := readAgentBenchEvents(t, outDir)
		first, tool := false, false
		for _, event := range events {
			name := strings.ToLower(fmt.Sprint(event["event"]))
			first = first || name == "first_output"
			tool = tool || name == "tool"
		}
		if !first || !tool {
			t.Fatalf("fragmented valid tool call lacked first_output/tool lifecycle: first=%v tool=%v", first, tool)
		}
	})
}

package modelperfobs

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

func TestProxyCapturesStreamingTimingAndUsage(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("X-Fak-Observation-ID"); got == "" {
			t.Error("missing correlation header")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		for _, line := range []string{
			`data: {"choices":[{"delta":{"content":"a"}}]}`,
			`data: {"choices":[{"delta":{"content":"b"}}]}`,
			`data: {"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}`,
			`data: [DONE]`,
		} {
			_, _ = w.Write([]byte(line + "\n"))
			f.Flush()
		}
	}))
	defer backend.Close()
	u, _ := ParseBackend(backend.URL)
	ledger := t.TempDir() + "/observations.jsonl"
	times := []time.Time{time.Unix(0, 0), time.Unix(0, 100e6), time.Unix(0, 140e6), time.Unix(0, 200e6), time.Unix(0, 260e6), time.Unix(0, 300e6)}
	i := 0
	proxy := httptest.NewServer(&Proxy{Backend: u, Ledger: ledger, Now: func() time.Time { v := times[i]; i++; return v }})
	defer proxy.Close()

	resp, err := http.Post(proxy.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"model":"Qwen3.8-27B","stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	if resp.Header.Get("X-Fak-Observation-ID") == "" {
		t.Fatal("response lacks observation ID")
	}

	f, err := os.Open(ledger)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	rows, err := ReadObservations(bufio.NewReader(f))
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 {
		t.Fatalf("rows=%d", len(rows))
	}
	got := rows[0]
	if got.Model != "Qwen3.8-27B" || got.PromptTokens != 10 || got.CompletionTokens != 2 {
		t.Fatalf("observation=%+v", got)
	}
	if got.TTFTMS <= 0 || got.InterChunkCount != 1 || got.TPOTMS <= 0 || got.OutputTokensPerSec <= 0 {
		t.Fatalf("timing=%+v", got)
	}
}

func TestSummarizeNamesPrefillOrQueueBottleneck(t *testing.T) {
	rows := []Observation{
		{Schema: Schema, Status: 200, PromptTokens: 4000, CompletionTokens: 20, DurationMS: 2500, TTFTMS: 1800, TPOTMS: 36, OutputTokensPerSec: 27},
		{Schema: Schema, Status: 200, PromptTokens: 4200, CompletionTokens: 20, DurationMS: 2700, TTFTMS: 2000, TPOTMS: 37, OutputTokensPerSec: 26},
	}
	s := Summarize(rows)
	if s.LikelyBottleneck != "prefill-or-queue" {
		b, _ := json.Marshal(s)
		t.Fatalf("summary=%s", b)
	}
	var b strings.Builder
	if err := WriteMarkdown(&b, s); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(b.String(), "Likely bottleneck: **prefill-or-queue**") {
		t.Fatal(b.String())
	}
}

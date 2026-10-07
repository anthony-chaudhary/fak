package gateway

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

// perf_ledger_served_test.go — the per-request perf row witnessed from a SERVED
// request over a real TCP listener, not from a hand call into the metrics fold.
// Each case boots the gateway behind httptest.NewServer with the durable ledger
// attached, drives /v1/chat/completions the way a client does (buffered and
// streamed), then reads the row back three ways: /v1/fak/perf/recent, the
// compact line, and the JSONL file `fak perf` tails with the server down.

// fak-test:runtime fast est=3s lane=default — synthetic in-kernel model on the CPU
// reference path plus a loopback upstream; no weights, no network.

type perfServedHarness struct {
	t    *testing.T
	srv  *Server
	ts   *httptest.Server
	w    *perfledger.Writer
	path string
}

func newPerfServedHarness(t *testing.T, planner agent.Planner) *perfServedHarness {
	t.Helper()
	srv := newTestServer(t)
	srv.planner = planner
	path := filepath.Join(t.TempDir(), "gateway-perf.jsonl")
	w := perfledger.OpenWriter(path, perfledger.DefaultMaxBytes)
	srv.SetPerfLedger(w, nil, false)
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return &perfServedHarness{t: t, srv: srv, ts: ts, w: w, path: path}
}

func (h *perfServedHarness) chat(model, content string, stream bool) {
	h.t.Helper()
	body := fmt.Sprintf(`{"model":%q,"stream":%t,"max_tokens":6,"messages":[{"role":"user","content":%q}]}`, model, stream, content)
	resp, err := http.Post(h.ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		h.t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		h.t.Fatalf("chat stream=%t: status %d: %s", stream, resp.StatusCode, raw)
	}
}

func (h *perfServedHarness) recent() perfledger.Report {
	h.t.Helper()
	resp, err := http.Get(h.ts.URL + "/v1/fak/perf/recent?n=50")
	if err != nil {
		h.t.Fatal(err)
	}
	defer resp.Body.Close()
	var rep perfledger.Report
	if err := json.NewDecoder(resp.Body).Decode(&rep); err != nil {
		h.t.Fatal(err)
	}
	return rep
}

func (h *perfServedHarness) compact() string {
	h.t.Helper()
	resp, err := http.Get(h.ts.URL + "/v1/fak/perf/recent?format=compact")
	if err != nil {
		h.t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	return strings.TrimSpace(string(raw))
}

// durable closes the writer and returns what `fak perf` would read off disk.
func (h *perfServedHarness) durable() []perfledger.Record {
	h.t.Helper()
	if err := h.w.Close(); err != nil {
		h.t.Fatalf("writer close: %v", err)
	}
	recs, _ := perfledger.ReadTail(h.path)
	return recs
}

func newPerfServedNativePlanner(t *testing.T) *agent.InKernelPlanner {
	t.Helper()
	t.Setenv("FAK_INKERNEL_MAX_TOKENS", "6")
	m := model.NewSynthetic(kvmmuSynthCfg())
	m.Quantize()
	return agent.NewInKernelPlanner(m, newByteLevelTokenizer(t), "native-synth", false, nil, false)
}

// fak-test:runtime fast est=2s lane=default
func TestPerfLedgerServedNativeTurnsCarryEngineAnatomy(t *testing.T) {
	h := newPerfServedHarness(t, newPerfServedNativePlanner(t))
	h.chat("native-synth", "buffered native turn", false)
	h.chat("native-synth", "streamed native turn", true)

	rep := h.recent()
	if rep.Source != "live" || len(rep.Records) != 2 {
		t.Fatalf("recent = source %q rows %d, want live/2: %+v", rep.Source, len(rep.Records), rep)
	}
	for i, r := range rep.Records {
		if r.Model != "native-synth" {
			t.Errorf("row %d model = %q, want native-synth", i, r.Model)
		}
		if r.Engine == nil || r.Engine.Path != perfledger.PathSerial {
			t.Errorf("row %d engine = %+v, want path=serial", i, r.Engine)
		}
		if r.CompletionTokens <= 0 || r.E2EMS <= 0 || r.TTFTMS <= 0 || r.TTFTMS > r.E2EMS {
			t.Errorf("row %d axes = %+v, want completion>0 and 0<ttft<=e2e", i, r)
		}
	}
	if rep.Summary.Native != 2 || rep.Summary.ByPath[perfledger.PathSerial] != 2 || rep.Summary.TTFTMeasured != 2 {
		t.Fatalf("summary = %+v, want native=2 serial=2 ttft_measured=2", rep.Summary)
	}
	if line := h.compact(); !strings.Contains(line, "path: serial=2") {
		t.Fatalf("compact = %q, want the decode-path split", line)
	}
	if recs := h.durable(); len(recs) != 2 || recs[0].Engine == nil || recs[1].Model != "native-synth" {
		t.Fatalf("durable rows = %+v, want both native rows with engine anatomy on disk", recs)
	}
}

// fak-test:runtime fast est=2s lane=default
func TestPerfLedgerServedSpeculativeTurnRecordsItsOwnRounds(t *testing.T) {
	p := newPerfServedNativePlanner(t)
	p.EnableSpeculativeDecoding(model.NewNGramProposalGenerator(model.NgramDrafter{Enabled: true, MinMatch: 1, MaxMatch: 4, MaxDraft: 3}), 3)
	h := newPerfServedHarness(t, p)
	h.chat("native-synth", "abababababababababababababab", false)

	rep := h.recent()
	if len(rep.Records) != 1 || rep.Records[0].Engine == nil {
		t.Fatalf("recent = %+v, want one native row", rep.Records)
	}
	e := rep.Records[0].Engine
	stats := p.SpeculativeEngine().Stats()
	if stats.VerificationRounds == 0 {
		t.Skipf("synthetic drafter proposed nothing on this prompt (engine stats %+v); the serial witness covers the row", stats)
	}
	if e.Path != perfledger.PathSpeculative || e.SpecRounds != int(stats.VerificationRounds) || e.SpecDraftTokens <= 0 || e.SpecAcceptedTokens > e.SpecDraftTokens {
		t.Fatalf("engine = %+v, want path=speculative with this request's %d rounds", e, stats.VerificationRounds)
	}
	s := rep.Summary
	if s.SpecRounds != e.SpecRounds || s.SpecDraftTokens != e.SpecDraftTokens || s.SpecAcceptRate != perfRound4(float64(e.SpecAcceptedTokens)/float64(e.SpecDraftTokens)) {
		t.Fatalf("summary spec = %+v, want the row's rounds and accept rate", s)
	}
	if line := h.compact(); !strings.Contains(line, "spec accept=") {
		t.Fatalf("compact = %q, want the spec acceptance segment", line)
	}
}

func perfRound4(v float64) float64 { return float64(int64(v*10000+0.5)) / 10000 }

// fak-test:runtime fast est=1s lane=default
func TestPerfLedgerServedProxyStreamMeasuresTTFT(t *testing.T) {
	const prefill = 60 * time.Millisecond
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Stream bool `json:"stream"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		usage := `"usage":{"prompt_tokens":120,"completion_tokens":3,"total_tokens":123,"prompt_tokens_details":{"cached_tokens":80}}`
		if !req.Stream {
			fmt.Fprintf(w, `{"id":"x","object":"chat.completion","created":1,"model":"up-model","choices":[{"index":0,"message":{"role":"assistant","content":"one two three"},"finish_reason":"stop"}],%s}`, usage)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		time.Sleep(prefill)
		bw := bufio.NewWriter(w)
		for _, word := range []string{"one", " two", " three"} {
			fmt.Fprintf(bw, "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"up-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":%q},\"finish_reason\":null}]}\n\n", word)
			_ = bw.Flush()
			fl.Flush()
			time.Sleep(10 * time.Millisecond)
		}
		fmt.Fprintf(bw, "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"up-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],%s}\n\ndata: [DONE]\n\n", usage)
		_ = bw.Flush()
		fl.Flush()
	}))
	t.Cleanup(upstream.Close)

	h := newPerfServedHarness(t, agent.NewHTTPPlanner(upstream.URL+"/v1", "up-model", ""))
	h.chat("up-model", "buffered proxy turn", false)
	h.chat("up-model", "streamed proxy turn", true)

	rep := h.recent()
	if len(rep.Records) != 2 {
		t.Fatalf("recent rows = %d, want 2: %+v", len(rep.Records), rep.Records)
	}
	buffered, streamed := rep.Records[0], rep.Records[1]
	if buffered.TTFTMS != 0 || buffered.Engine != nil {
		t.Errorf("buffered proxy row = %+v, want ttft unmeasured and no engine anatomy", buffered)
	}
	if streamed.TTFTMS < float64(prefill/time.Millisecond) || streamed.TTFTMS > streamed.E2EMS || streamed.DecodeTPS <= 0 {
		t.Errorf("streamed proxy row = %+v, want ttft >= %v upstream prefill, <= e2e, with a decode rate", streamed, prefill)
	}
	if streamed.Model != "up-model" || streamed.Engine != nil || streamed.CachedTokens != 80 {
		t.Errorf("streamed proxy row = %+v, want model=up-model, cached=80, no engine anatomy", streamed)
	}
	if rep.Summary.TTFTMeasured != 1 || rep.Summary.Native != 0 {
		t.Fatalf("summary = %+v, want exactly the streamed row measured and no native rows", rep.Summary)
	}
}

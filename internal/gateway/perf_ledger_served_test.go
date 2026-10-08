package gateway

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strconv"
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

func (h *perfServedHarness) chat(model, content string, stream bool) []byte {
	h.t.Helper()
	body := fmt.Sprintf(`{"model":%q,"stream":%t,"max_tokens":6,"messages":[{"role":"user","content":%q}]}`, model, stream, content)
	return h.chatBody(body)
}

func (h *perfServedHarness) chatBody(body string) []byte {
	h.t.Helper()
	resp, err := http.Post(h.ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		h.t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		h.t.Fatalf("chat status %d: %s", resp.StatusCode, raw)
	}
	return raw
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
	buffered := h.chat("native-synth", "buffered native turn", false)
	streamed := h.chat("native-synth", "streamed native turn", true)
	for shape, body := range map[string][]byte{"buffered": buffered, "streamed": streamed} {
		for _, field := range []string{"native_inference_receipt", "token_ids", "token_logprobs"} {
			if strings.Contains(string(body), field) {
				t.Fatalf("default %s response exposed opt-in field %q: %s", shape, field, body)
			}
		}
	}
	strict := h.chatBody(`{"model":"native-synth","messages":[{"role":"user","content":"strict native turn"}],"max_tokens":6,"temperature":0,"fak":{"native_inference_receipt":true}}`)
	var strictResponse ChatResponse
	if err := json.Unmarshal(strict, &strictResponse); err != nil {
		t.Fatalf("decode strict receipt response: %v", err)
	}
	if strictResponse.Fak == nil || strictResponse.Fak.NativeInferenceReceipt == nil {
		t.Fatalf("explicit response omitted strict native receipt: %s", strict)
	}
	strictReceipt := strictResponse.Fak.NativeInferenceReceipt
	if len(strictReceipt.TokenIDs) == 0 || len(strictReceipt.TokenIDs) > 6 || len(strictReceipt.TokenLogprobs) != len(strictReceipt.TokenIDs) || strictResponse.Usage.CompletionTokens != len(strictReceipt.TokenIDs) {
		t.Fatalf("strict receipt token ids/logprobs/usage=%d/%d/%d, want consistent bounded entries", len(strictReceipt.TokenIDs), len(strictReceipt.TokenLogprobs), strictResponse.Usage.CompletionTokens)
	}
	for name, seconds := range map[string]float64{"prefill": strictReceipt.PrefillSeconds, "ttft": strictReceipt.TTFTSeconds, "decode": strictReceipt.DecodeSeconds} {
		if seconds < 0 || math.IsNaN(seconds) || math.IsInf(seconds, 0) {
			t.Fatalf("strict receipt %s_seconds=%v, want finite non-negative", name, seconds)
		}
	}
	if strictReceipt.Model != "native-synth" || strictReceipt.Engine != "inkernel" || strictReceipt.Planner != "inkernel" || strictReceipt.Owner != "fak" || strictReceipt.FallbackActive {
		t.Fatalf("strict receipt execution identity=%+v, want native-synth inkernel without fallback", strictReceipt)
	}

	rep := h.recent()
	if rep.Source != "live" || len(rep.Records) != 3 {
		t.Fatalf("recent = source %q rows %d, want live/3: %+v", rep.Source, len(rep.Records), rep)
	}
	phaseMeasured := func(row int, name string, ms float64, tokens int, rate float64) bool {
		t.Helper()
		if ms == 0 {
			if tokens != 0 || rate != 0 {
				t.Fatalf("row %d unknown %s phase retained tokens/rate: ms=%v tokens=%d rate=%v", row, name, ms, tokens, rate)
			}
			return false
		}
		if ms < 0 || math.IsNaN(ms) || math.IsInf(ms, 0) || tokens < 0 || math.IsNaN(rate) || math.IsInf(rate, 0) || (tokens > 0 && rate <= 0) {
			t.Fatalf("row %d measured %s phase is invalid: ms=%v tokens=%d rate=%v", row, name, ms, tokens, rate)
		}
		return true
	}
	wantPrefillMeasured, wantDecodeMeasured, wantTTFTMeasured := 0, 0, 0
	for i, r := range rep.Records {
		if r.Model != "native-synth" {
			t.Errorf("row %d model = %q, want native-synth", i, r.Model)
		}
		if r.Engine == nil || r.Engine.Path != perfledger.PathSerial {
			t.Errorf("row %d engine = %+v, want path=serial", i, r.Engine)
		}
		prefillMeasured, decodeMeasured := false, false
		if timing := r.NativeTiming; timing != nil {
			prefillMeasured = phaseMeasured(i, "prefill", timing.PrefillMS, timing.PrefillTokens, timing.PrefillTPS)
			decodeMeasured = phaseMeasured(i, "decode", timing.DecodeMS, timing.DecodeTokens, timing.DecodeTPS)
		}
		if prefillMeasured {
			wantPrefillMeasured++
		}
		if decodeMeasured {
			wantDecodeMeasured++
		}
		timingJSON, err := json.Marshal(r.NativeTiming)
		if err != nil {
			t.Fatalf("marshal row %d native timing: %v", i, err)
		}
		for phase, measured := range map[string]bool{"prefill": prefillMeasured, "decode": decodeMeasured} {
			if measured {
				if !strings.Contains(string(timingJSON), `"`+phase+`_ms"`) {
					t.Fatalf("row %d measured %s phase omitted duration: %s", i, phase, timingJSON)
				}
				continue
			}
			for _, suffix := range []string{"_ms", "_tokens", "_tps"} {
				field := phase + suffix
				if strings.Contains(string(timingJSON), field) {
					t.Fatalf("row %d unknown %s phase emitted %s: %s", i, phase, field, timingJSON)
				}
			}
		}
		if r.CompletionTokens <= 0 || r.E2EMS <= 0 || math.IsNaN(r.E2EMS) || math.IsInf(r.E2EMS, 0) || r.TTFTMS < 0 || r.TTFTMS > r.E2EMS || math.IsNaN(r.TTFTMS) || math.IsInf(r.TTFTMS, 0) {
			t.Errorf("row %d axes = %+v, want completion/e2e positive and finite 0<=ttft<=e2e", i, r)
		}
		if i < 2 && r.TTFTMS <= 0 {
			t.Errorf("ordinary row %d axes = %+v, want measured positive TTFT", i, r)
		}
		if r.TTFTMS > 0 {
			wantTTFTMeasured++
		}
	}
	if wantTTFTMeasured < 2 {
		t.Fatalf("ordinary cohort lost measured TTFT: measured=%d", wantTTFTMeasured)
	}
	strictTimingJSON, err := json.Marshal(rep.Records[2].NativeTiming)
	if err != nil {
		t.Fatalf("marshal strict row timing: %v", err)
	}
	wantStrictTiming := perfledger.NewNativeTiming(rep.Records[2].PromptTokens, rep.Records[2].CompletionTokens, strictReceipt.PrefillSeconds*1000, strictReceipt.DecodeSeconds*1000)
	wantStrictTimingJSON, err := json.Marshal(wantStrictTiming)
	if err != nil {
		t.Fatalf("marshal receipt-derived strict timing: %v", err)
	}
	if string(strictTimingJSON) != string(wantStrictTimingJSON) {
		t.Fatalf("strict row timing=%s, want receipt-derived timing=%s", strictTimingJSON, wantStrictTimingJSON)
	}
	if rep.Summary.Native != 3 || rep.Summary.ByPath[perfledger.PathSerial] != 3 || rep.Summary.TTFTMeasured != wantTTFTMeasured {
		t.Fatalf("summary = %+v, want native=3 serial=3 ttft_measured=%d", rep.Summary, wantTTFTMeasured)
	}
	path := rep.Summary.ByPathStats[perfledger.PathSerial]
	if path.Count != 3 || path.PrefillMeasured != wantPrefillMeasured || path.DecodeMeasured != wantDecodeMeasured || path.PrefillTPSWeighted <= 0 || path.DecodeTPSWeighted <= 0 {
		t.Fatalf("live serial path summary = %+v, want measured prefill/decode=%d/%d", path, wantPrefillMeasured, wantDecodeMeasured)
	}
	metrics := h.srv.renderMetrics()
	sample := func(series string) float64 {
		t.Helper()
		for _, line := range strings.Split(metrics, "\n") {
			if strings.HasPrefix(line, series+" ") {
				fields := strings.Fields(line)
				value, err := strconv.ParseFloat(fields[len(fields)-1], 64)
				if err != nil {
					t.Fatalf("parse metric %s: %v", series, err)
				}
				return value
			}
		}
		t.Fatalf("missing metric %s", series)
		return 0
	}
	requestSeries := `fak_native_execution_requests_total{path="serial"}`
	if got := sample(requestSeries); got != 3 || strings.Count(metrics, "fak_native_execution_requests_total{") != 1 {
		t.Fatalf("native execution bookings=%v rows=%d, want exactly three serial bookings", got, strings.Count(metrics, "fak_native_execution_requests_total{"))
	}
	for _, series := range []string{
		`fak_native_execution_phase_seconds_total{path="serial",phase="prefill"}`,
		`fak_native_execution_phase_seconds_total{path="serial",phase="decode"}`,
		`fak_native_execution_tokens_total{path="serial",kind="prompt"}`,
		`fak_native_execution_tokens_total{path="serial",kind="generated"}`,
	} {
		value := sample(series)
		if value <= 0 || math.IsNaN(value) || math.IsInf(value, 0) {
			t.Fatalf("native execution metric %s=%v, want finite positive", series, value)
		}
	}
	strictSeries := `fak_native_receipt_requests_total{engine="inkernel",backend="other",forward_path="other",evidence_class="measured"}`
	if got := sample(strictSeries); got != 1 || strings.Count(metrics, "fak_native_receipt_requests_total{") != 1 {
		t.Fatalf("strict receipt bookings=%v rows=%d, want exactly one strict booking", got, strings.Count(metrics, "fak_native_receipt_requests_total{"))
	}
	line := h.compact()
	for _, token := range []string{"path: serial=3", "serial native n=3", "prefill p50=", "decode p50=", fmt.Sprintf("(measured %d/3)", wantPrefillMeasured), fmt.Sprintf("(measured %d/3)", wantDecodeMeasured)} {
		if !strings.Contains(line, token) {
			t.Fatalf("compact = %q, want native path token %q", line, token)
		}
	}
	recs := h.durable()
	if len(recs) != 3 || recs[0].Engine == nil || recs[1].Model != "native-synth" {
		t.Fatalf("durable rows = %+v, want all three native rows with engine anatomy on disk", recs)
	}
	durableStrictTimingJSON, err := json.Marshal(recs[2].NativeTiming)
	if err != nil || string(durableStrictTimingJSON) != string(strictTimingJSON) {
		t.Fatalf("durable strict timing=%s err=%v, want live timing=%s", durableStrictTimingJSON, err, strictTimingJSON)
	}
	offline := perfledger.BuildReport(recs, len(recs), false, 0)
	if offline.Summary.TTFTMeasured != wantTTFTMeasured {
		t.Fatalf("durable TTFT measured=%d, want live count %d", offline.Summary.TTFTMeasured, wantTTFTMeasured)
	}
	offlinePath := offline.Summary.ByPathStats[perfledger.PathSerial]
	if offlinePath.Count != 3 || offlinePath.PrefillMeasured != wantPrefillMeasured || offlinePath.DecodeMeasured != wantDecodeMeasured || offlinePath.PrefillTPSWeighted <= 0 || offlinePath.DecodeTPSWeighted <= 0 {
		t.Fatalf("durable serial path summary = %+v, want measured prefill/decode=%d/%d", offlinePath, wantPrefillMeasured, wantDecodeMeasured)
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

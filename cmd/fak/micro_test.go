package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metrics"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

// TestMicroConfigPrecedence pins the flags > env > file > defaults ladder the issue
// (#2029) requires: a file value survives where nothing above it sets the field, and
// an env var overrides the file for the field it names.

func TestDriveMicroObservedCapturesTaskAnswerAndProviderUsage(t *testing.T) {
	var prompt string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		if len(req.Messages) > 0 {
			prompt = req.Messages[len(req.Messages)-1].Content
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"paired","model":"provider-model","choices":[{"message":{"role":"assistant","content":"READY"},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":1,"total_tokens":12}}`)
	}))
	defer srv.Close()

	cfg := defaultMicroConfig(false)
	cfg.Engine, cfg.Gateway, cfg.Model = "gateway", srv.URL, "requested-model"
	cfg.Task, cfg.Agents, cfg.Turns = "Reply with exactly READY", 1, 1
	_, _, results, observed, err := driveMicroObserved(cfg)
	if err != nil {
		t.Fatalf("driveMicroObserved: %v", err)
	}
	if len(results) != 1 || !results[0].Done {
		t.Fatalf("results=%+v", results)
	}
	got := observed["micro-000"]
	if prompt != cfg.Task {
		t.Fatalf("prompt=%q want %q", prompt, cfg.Task)
	}
	if got.Answer != "READY" || got.Usage.TotalTokens != 12 || got.Model != "provider-model" {
		t.Fatalf("observation=%+v", got)
	}
}

func TestMicroConfigPrecedence(t *testing.T) {
	cfg := defaultMicroConfig(true) // host mode: agents defaults to the worker count
	if cfg.Engine != "mock" || cfg.Isolation != microagent.BackendGoroutine {
		t.Fatalf("defaults: engine=%q isolation=%q", cfg.Engine, cfg.Isolation)
	}
	if cfg.Agents != microagent.DefaultWorkers {
		t.Fatalf("host default agents=%d, want %d", cfg.Agents, microagent.DefaultWorkers)
	}

	// File overlay: workers=4, turns=5 (below env, above defaults).
	dir := t.TempDir()
	path := filepath.Join(dir, "micro.json")
	if err := os.WriteFile(path, []byte(`{"workers":4,"turns":5}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := loadMicroConfigFile(path, &cfg); err != nil {
		t.Fatalf("loadMicroConfigFile: %v", err)
	}
	if cfg.Workers != 4 || cfg.Turns != 5 {
		t.Fatalf("after file: workers=%d turns=%d, want 4/5", cfg.Workers, cfg.Turns)
	}

	// Env over file: FAK_MICRO_WORKERS beats the file's workers; turns (no env) stays.
	t.Setenv("FAK_MICRO_WORKERS", "9")
	applyMicroEnv(&cfg)
	if cfg.Workers != 9 {
		t.Errorf("env should override file: workers=%d, want 9", cfg.Workers)
	}
	if cfg.Turns != 5 {
		t.Errorf("field with no env should keep file value: turns=%d, want 5", cfg.Turns)
	}
}

// TestMicroValidate pins the isolation-backend guard: only a registered ToolExec
// backend name is accepted, and the numeric caps must be sane.
func TestMicroValidate(t *testing.T) {
	cfg := defaultMicroConfig(false)
	if err := validateMicroConfig(cfg); err != nil {
		t.Fatalf("default config should validate: %v", err)
	}
	bad := cfg
	bad.Isolation = "bogus"
	if err := validateMicroConfig(bad); err == nil {
		t.Error("unknown isolation backend should be refused")
	}
	bad = cfg
	bad.Agents = 0
	if err := validateMicroConfig(bad); err == nil {
		t.Error("agents<1 should be refused")
	}
	bad = cfg
	bad.AdmissionTokenBudget = -1
	if err := validateMicroConfig(bad); err == nil {
		t.Error("negative admission cap should be refused")
	}
}

// TestMicroSlots pins the effective slot-pool derivation: an explicit seat count
// wins, otherwise it falls back to the worker count.
func TestMicroSlots(t *testing.T) {
	if got := (microConfig{Workers: 8, Seats: 0}).slots(); got != 8 {
		t.Errorf("seats=0 should derive from workers: got %d, want 8", got)
	}
	if got := (microConfig{Workers: 8, Seats: 3}).slots(); got != 3 {
		t.Errorf("explicit seats should win: got %d, want 3", got)
	}
}

// TestMicroRunEndToEndOnMock is the #2029 acceptance witness: `fak micro` runs
// agents end-to-end on the Mock engine through the real microagent.Host, the slot
// scheduler, and one audit sink — every spawned agent retires done with the
// configured number of steps.
func TestMicroRunEndToEndOnMock(t *testing.T) {
	cfg := defaultMicroConfig(false) // bare `fak micro`: one agent
	cfg.Turns = 2
	if cfg.Agents != 1 {
		t.Fatalf("bare micro should default to 1 agent, got %d", cfg.Agents)
	}
	if err := runMicro(cfg, false, true, ""); err != nil {
		t.Fatalf("runMicro (1 agent): %v", err)
	}

	// A small fleet also completes cleanly through the host.
	fleet := defaultMicroConfig(true)
	fleet.Agents = 5
	fleet.Turns = 3
	fleet.Seats = 2
	if err := runMicro(fleet, true, true, ""); err != nil {
		t.Fatalf("runMicro (fleet): %v", err)
	}
}

// TestMicroTraceSeparableInHost is the #2031 acceptance witness: a fleet runs
// interleaved in ONE host, each agent records its own structured span timeline, and
// the trace store pulls exactly one agent's timeline out by id — separable even
// though every agent ran concurrently in one process. The traces round-trip through
// a JSONL file, the cross-process readout path behind `fak micro trace --trace-in`.
func TestMicroTraceSeparableInHost(t *testing.T) {
	cfg := defaultMicroConfig(true)
	cfg.Agents = 6
	cfg.Turns = 3
	cfg.Seats = 2
	_, tracer, results, err := driveMicro(cfg)
	if err != nil {
		t.Fatalf("driveMicro: %v", err)
	}
	if len(results) != cfg.Agents {
		t.Fatalf("got %d results, want %d", len(results), cfg.Agents)
	}
	if got := len(tracer.IDs()); got != cfg.Agents {
		t.Fatalf("got %d traces, want %d (one per agent)", got, cfg.Agents)
	}
	// Every agent's timeline is separable and complete: 3 legs (seat, step, verdict)
	// per turn, all ALLOW, and a nonzero token count.
	for i := 0; i < cfg.Agents; i++ {
		id := fmt.Sprintf("micro-%03d", i)
		tr, ok := tracer.Trace(id)
		if !ok {
			t.Fatalf("no trace for %s", id)
		}
		if want := cfg.Turns * 3; len(tr.Spans) != want {
			t.Fatalf("%s: got %d spans, want %d", id, len(tr.Spans), want)
		}
		if tr.Tokens() <= 0 {
			t.Fatalf("%s: want a nonzero token count, got %d", id, tr.Tokens())
		}
		if v := tr.Verdicts(); len(v) != 1 || v[0] != "ALLOW" {
			t.Fatalf("%s: Verdicts()=%v, want [ALLOW]", id, v)
		}
	}

	// Persist → reload: the readout survives a separate process.
	path := filepath.Join(t.TempDir(), "traces.jsonl")
	if err := writeTraceFile(path, tracer); err != nil {
		t.Fatalf("writeTraceFile: %v", err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer f.Close()
	reloaded, err := metrics.ReadTracesJSONL(f)
	if err != nil {
		t.Fatalf("ReadTracesJSONL: %v", err)
	}
	want, _ := tracer.Render("micro-000")
	got, ok := reloaded.Render("micro-000")
	if !ok || got != want {
		t.Fatalf("reload render mismatch:\n got %q\nwant %q", got, want)
	}
}

// TestMicroTraceArgOrdering locks the flag/positional permutation: `fak micro trace`
// must accept <id> before OR after its flags. Go's flag package stops at the first
// non-flag token, so the id-first form the run summary prints
// (`fak micro trace <id> --trace-in <file>`) would otherwise die with a usage dump.
func TestMicroTraceArgOrdering(t *testing.T) {
	newFS := func() (*flag.FlagSet, *string, *bool) {
		fs := flag.NewFlagSet("micro trace", flag.ContinueOnError)
		fs.SetOutput(io.Discard)
		traceIn := fs.String("trace-in", "", "")
		jsonOut := fs.Bool("json", false, "")
		return fs, traceIn, jsonOut
	}

	for _, tc := range []struct {
		name string
		args []string
	}{
		{"id first", []string{"micro-001", "--trace-in", "t.jsonl", "--json"}},
		{"flags first", []string{"--trace-in", "t.jsonl", "--json", "micro-001"}},
		{"id in the middle", []string{"--trace-in", "t.jsonl", "micro-001", "--json"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, traceIn, jsonOut := newFS()
			id, err := parseMicroTraceArgs(fs, tc.args)
			if err != nil {
				t.Fatalf("parseMicroTraceArgs(%v): %v", tc.args, err)
			}
			if id != "micro-001" {
				t.Errorf("id = %q, want micro-001", id)
			}
			if *traceIn != "t.jsonl" {
				t.Errorf("--trace-in = %q, want t.jsonl", *traceIn)
			}
			if !*jsonOut {
				t.Errorf("--json = false, want true")
			}
		})
	}

	// A bare id (the acceptance form) resolves with no flags set.
	fs, traceIn, _ := newFS()
	id, err := parseMicroTraceArgs(fs, []string{"micro-002"})
	if err != nil || id != "micro-002" || *traceIn != "" {
		t.Fatalf("bare id: got (%q, %v, traceIn=%q), want (micro-002, nil, \"\")", id, err, *traceIn)
	}

	// Zero or two positionals are a usage error, not a silently-picked id.
	for _, args := range [][]string{{}, {"micro-000", "micro-001"}} {
		fs, _, _ := newFS()
		if _, err := parseMicroTraceArgs(fs, args); err == nil {
			t.Errorf("parseMicroTraceArgs(%v) = nil error, want a usage error", args)
		}
	}
}

func TestMicroGatewayEngineUsesRealFakKernel(t *testing.T) {
	var calls atomic.Int32
	var models sync.Map
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %q", r.URL.Path)
			http.Error(w, "wrong path", http.StatusNotFound)
			return
		}
		var req struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		models.Store(req.Model, true)
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"cmpl-real-kernel","model":"kernel-model","choices":[{"index":0,"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}`)
	}))
	defer srv.Close()

	cfg := defaultMicroConfig(false)
	cfg.Engine = "gateway"
	cfg.Gateway = srv.URL
	cfg.Model = "kernel-model"
	cfg.Agents = 2
	cfg.Turns = 1
	cfg.Workers = 2
	cfg.Seats = 1
	sink, tracer, results, err := driveMicro(cfg)
	if err != nil {
		t.Fatalf("driveMicro gateway: %v", err)
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("real kernel calls = %d, want 2", got)
	}
	if _, ok := models.Load("kernel-model"); !ok {
		t.Fatal("requested model did not reach fak gateway wire")
	}
	if len(results) != 2 || sink.count(microagent.EventDone) != 2 {
		t.Fatalf("results=%d done=%d", len(results), sink.count(microagent.EventDone))
	}
	for _, id := range []string{"micro-000", "micro-001"} {
		trace, ok := tracer.Trace(id)
		if !ok || trace.Tokens() != 9 {
			t.Fatalf("%s trace = %#v, ok=%v", id, trace, ok)
		}
	}
}

func TestMicroGatewayEngineRequiresEndpointAndModel(t *testing.T) {
	cfg := defaultMicroConfig(false)
	cfg.Engine = "gateway"
	if err := validateMicroConfig(cfg); err == nil || !strings.Contains(err.Error(), "--gateway") {
		t.Fatalf("missing gateway error = %v", err)
	}
	cfg.Gateway = "localhost:8080"
	if err := validateMicroConfig(cfg); err == nil || !strings.Contains(err.Error(), "--model") {
		t.Fatalf("missing model error = %v", err)
	}
}

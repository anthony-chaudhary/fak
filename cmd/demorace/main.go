// Command demorace is the live, on-box demo of fak's value point: REUSE.
//
// It runs a head-to-head LIVE race between three ways of serving the SAME model over the
// SAME 25-request multi-agent session, and builds a reuse CURVE across a model-size ladder.
//
//	The session: C agents share one P-token prefix; each runs T turns, decoding D tokens
//	per turn and ingesting R tool-result tokens between turns. Total requests = C·T (=25).
//
// The three arms (same model, same bit-identical kernels — the delta is PURE work-reuse):
//
//	arm "tuned" (B) — the SOTA serving baseline: a tuned warm per-agent KV cache (vLLM /
//	  SGLang prefix caching, provider prompt-caching). The prefix is prefilled ONCE per
//	  agent, then only the new tool-result tokens are ingested incrementally; serial decode.
//	  No quadratic re-prefill, no cross-agent sharing, no batching. Run FULLY LIVE here.
//	arm "fak"   (C) — prefix prefilled ONCE and cloned into C agents; batched decode (one
//	  weight stream serves all C); incremental result ingestion. Run FULLY LIVE here.
//	arm "naive" (A) — the cold re-prefill loop: re-prefill the ENTIRE growing context every
//	  turn, every agent (quadratic in T, ×C). A worst-case REFERENCE only — NOT a serving
//	  baseline anyone ships. Run live in the race; PROJECTED (faded) in the big-model curve.
//
// HEADLINE = B/C: fak vs the tuned warm-cache baseline — the honest competitive number
// (cross-agent prefix reuse + batched decode ON TOP of a warm cache). The A/C number (vs
// the cold loop) is shown only as the worst-case reference, never as the headline.
//
// Same model, same tokens, same answers; the only difference is how much shared setup the
// system makes the model re-read. That is the whole fak thesis, made visible in real time.
//
// Serve it:
//
//	go run ./cmd/demorace -addr 127.0.0.1:8147
//	# open http://127.0.0.1:8147  → "Run live race" then "Build curve"
//
// The ladder (135m → 0.5B → 1.5B → 3B) is auto-detected from disk; missing rungs are skipped.
package main

import (
	"embed"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/demoui"
	"github.com/anthony-chaudhary/fak/internal/model"
)

//go:embed page.html
var pageFS embed.FS

const version = "fak-demorace-v1"

// heartbeat is how often a blocking phase (model load, prefill measurement) emits a
// keep-alive "tick" so the page updates ~1×/s instead of freezing on a long load.
const heartbeat = 700 * time.Millisecond

// ---------------------------------------------------------------------------
// model ladder
// ---------------------------------------------------------------------------

type spec struct {
	Name    string // display label
	Dir     string // weights dir
	Kind    string // "dir" (fak export) or "hf" (safetensors snapshot, lean Q8 load)
	Params  string // human param count, for the curve x-axis
	Order   int    // ladder order
	Present bool
}

func ladderSpecs() []spec {
	home, _ := os.UserHomeDir()
	hf := filepath.Join(home, ".cache", "fak-models", "hf")
	candidates := []spec{
		{Name: "SmolLM2-135M", Params: "0.14B", Order: 0, Kind: "dir",
			Dir: "internal/model/.cache/smollm2-135m"},
		{Name: "Qwen2.5-0.5B", Params: "0.5B", Order: 1, Kind: "hf",
			Dir: filepath.Join(hf, "Qwen2.5-0.5B-Instruct")},
		{Name: "Qwen2.5-1.5B", Params: "1.5B", Order: 2, Kind: "hf",
			Dir: filepath.Join(hf, "Qwen2.5-1.5B-Instruct")},
		{Name: "Qwen2.5-3B", Params: "3B", Order: 3, Kind: "hf",
			Dir: filepath.Join(hf, "Qwen2.5-3B-Instruct")},
	}
	wd, _ := os.Getwd()
	for i := range candidates {
		dir := candidates[i].Dir
		if !filepath.IsAbs(dir) {
			dir = filepath.Join(wd, dir)
		}
		_, err := os.Stat(dir)
		candidates[i].Present = err == nil
		candidates[i].Dir = dir
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].Order < candidates[j].Order })
	return candidates
}

// registry: lazily load + quantize each model once, memoize.
type registry struct {
	mu    sync.Mutex
	cache map[string]*loaded
}

type loaded struct {
	m     *model.Model
	vocab int
	name  string
}

func newRegistry() *registry { return &registry{cache: map[string]*loaded{}} }

func (r *registry) get(s spec) (*loaded, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if l, ok := r.cache[s.Name]; ok {
		return l, nil
	}
	m, err := loadSpec(s)
	if err != nil {
		return nil, err
	}
	m.Quantize()
	l := &loaded{m: m, vocab: m.Cfg.VocabSize, name: s.Name}
	r.cache[s.Name] = l
	return l, nil
}

func readHFConfig(dir string) (model.Config, error) {
	var cfg model.Config
	cb, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return cfg, fmt.Errorf("config.json: %w", err)
	}
	if err := json.Unmarshal(cb, &cfg); err != nil {
		return cfg, fmt.Errorf("config.json parse: %w", err)
	}
	if cfg.HeadDim == 0 && cfg.NumHeads != 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumHeads
	}
	return cfg, nil
}

func loadSpec(s spec) (*model.Model, error) {
	switch s.Kind {
	case "hf":
		cfg, err := readHFConfig(s.Dir)
		if err != nil {
			return nil, err
		}
		// sharded vs single safetensors
		if _, err := os.Stat(filepath.Join(s.Dir, "model.safetensors.index.json")); err == nil {
			return model.LoadSafetensorsQuantDir(s.Dir, cfg)
		}
		return model.LoadSafetensorsQuant(filepath.Join(s.Dir, "model.safetensors"), cfg)
	default:
		return model.Load(s.Dir)
	}
}

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

func ms(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e6 }

func lcgIDs(n, vocab int, seed uint64) []int {
	ids := make([]int, n)
	state := 2463534242 + seed
	for i := 0; i < n; i++ {
		state = (state*1103515245 + 12345) & 0x7fffffff
		ids[i] = int(state % uint64(vocab))
	}
	return ids
}

// prefillModel captures prefill wall-clock at sampled lengths, so the naive arm's quadratic
// re-prefill can be summed over the exact per-turn context lengths (the sessionbench method).
type prefillModel struct {
	Lens []int     `json:"lens"`
	MS   []float64 `json:"ms"`
}

func measurePrefill(m *model.Model, vocab int, lens []int, reps int) prefillModel {
	pm := prefillModel{}
	for _, L := range lens {
		ids := lcgIDs(L, vocab, uint64(1000+L))
		best := math.MaxFloat64
		for r := 0; r < reps; r++ {
			s := m.NewSession()
			s.Quant = true
			t0 := time.Now()
			s.Prefill(ids)
			d := ms(time.Since(t0))
			if d < best {
				best = d
			}
			s.Close()
		}
		pm.Lens = append(pm.Lens, L)
		pm.MS = append(pm.MS, best)
	}
	return pm
}

func (pm prefillModel) cost(L int) float64 {
	n := len(pm.Lens)
	if n == 0 {
		return 0
	}
	if L <= pm.Lens[0] {
		return pm.MS[0] * float64(L) / float64(pm.Lens[0])
	}
	for i := 1; i < n; i++ {
		if L <= pm.Lens[i] {
			lo, hi := pm.Lens[i-1], pm.Lens[i]
			frac := float64(L-lo) / float64(hi-lo)
			return pm.MS[i-1] + frac*(pm.MS[i]-pm.MS[i-1])
		}
	}
	lo, hi := pm.Lens[n-2], pm.Lens[n-1]
	slope := (pm.MS[n-1] - pm.MS[n-2]) / float64(hi-lo)
	return pm.MS[n-1] + slope*float64(L-hi)
}

// computeAPrefill sums prefillCost over the naive arm's exact per-turn context lengths, ×C.
func computeAPrefill(pm prefillModel, P, T, C, D, R int) float64 {
	var total float64
	for t := 0; t < T; t++ {
		ctx := P + t*(D+R)
		total += pm.cost(ctx)
	}
	return total * float64(C)
}

// prefillTokens is the EXACT, timing-free prefill-token count each arm processes — fixed by
// the session structure alone, so the A/C work-elimination ratio cannot drift with load.
func prefillTokens(P, T, C, D, R int) (a, b, c int) {
	for t := 0; t < T; t++ {
		a += P + t*(D+R)
	}
	a *= C
	b = C * (P + (T-1)*R)
	c = P + C*(T-1)*R
	return
}

// ---------------------------------------------------------------------------
// live arms with per-turn progress
// ---------------------------------------------------------------------------

type event map[string]any

type emitter func(event)

// liveArmC runs the fak fused arm live: prefix ONCE + clone into C agents + batched decode
// + incremental result ingestion. Emits one "turn" event per turn (each turn serves all C
// agents, so requests advance by C per turn).
func liveArmC(l *loaded, P, T, C, D, R int, emit emitter) (totalMS, decodeMS float64) {
	m, vocab := l.m, l.vocab
	prefix := lcgIDs(P, vocab, 1)
	ids0 := lcgIDs(C, vocab, 991)
	start := time.Now()

	base := m.NewSession()
	base.Quant = true
	t0 := time.Now()
	base.Prefill(prefix)
	prefillMS := ms(time.Since(t0))

	bs := m.NewBatchFromPrefixReserve(base.Cache, C, T*(D+R))
	bs.SetQuant(true)
	_ = prefillMS

	ids := append([]int(nil), ids0...)
	var dcAcc float64
	prefilledSoFar := P
	for t := 0; t < T; t++ {
		t1 := time.Now()
		for d := 0; d < D; d++ {
			bs.StepBatch(ids)
			for j := range ids {
				ids[j] = (ids[j]*48271 + 1) % vocab
			}
		}
		dcAcc += ms(time.Since(t1))
		if t < T-1 {
			prompts := make([][]int, C)
			for a := range prompts {
				prompts[a] = lcgIDs(R, vocab, uint64(50000+t*1000+a*97))
			}
			bs.PrefillEach(prompts)
			prefilledSoFar += C * R
		}
		emit(event{
			"type": "turn", "arm": "fak", "turn": t, "agent": -1,
			"requests_done": (t + 1) * C, "total_requests": C * T,
			"tokens_prefilled": prefilledSoFar, "tokens_decoded": C * (t + 1) * D,
			"elapsed_ms": ms(time.Since(start)),
		})
	}
	base.Close()
	totalMS = ms(time.Since(start))
	decodeMS = dcAcc
	return
}

// liveArmB runs the tuned warm-cache arm live — the SOTA serving baseline. Each agent
// keeps a PERSISTENT per-agent KV cache: it prefills the shared prefix ONCE (so C times
// across the fleet — no cross-agent sharing, no batching), then ingests only the NEW
// tool-result tokens incrementally and decodes serially. This is the real baseline a
// tuned single-tenant stack gives you — vLLM / SGLang prefix caching, provider prompt
// caching, a persistent KV per session — so the fak-vs-tuned gap is the honest number
// (cross-agent prefix reuse + batched decode on top of a warm cache), not the strawman
// gap vs the cold re-prefill loop. Emits one "turn" event per (agent,turn).
func liveArmB(l *loaded, P, T, C, D, R int, emit emitter) (totalMS, decodeMS float64) {
	m, vocab := l.m, l.vocab
	prefix := lcgIDs(P, vocab, 1)
	ids0 := lcgIDs(C, vocab, 991)
	start := time.Now()

	var dcAcc, prefilledSoFar float64
	done := 0
	for a := 0; a < C; a++ {
		s := m.NewSession()
		s.Quant = true
		s.Prefill(prefix) // prefix prefilled ONCE per agent (warm KV — never re-prefill the growing context)
		prefilledSoFar += float64(P)
		tok := ids0[a]
		for t := 0; t < T; t++ {
			t1 := time.Now()
			for d := 0; d < D; d++ {
				s.Step(tok)
				tok = (tok*48271 + 1) % vocab
			}
			dcAcc += ms(time.Since(t1))
			if t < T-1 {
				s.Prefill(lcgIDs(R, vocab, uint64(50000+t*1000+a*97))) // ingest ONLY the new result tokens
				prefilledSoFar += float64(R)
			}
			done++
			emit(event{
				"type": "turn", "arm": "tuned", "turn": t, "agent": a,
				"requests_done": done, "total_requests": C * T,
				"tokens_prefilled": int(prefilledSoFar), "tokens_decoded": done * D,
				"elapsed_ms": ms(time.Since(start)),
			})
		}
		s.Close()
	}
	totalMS = ms(time.Since(start))
	decodeMS = dcAcc
	return
}

// liveArmA runs the naive arm live: every (agent,turn) re-prefills the ENTIRE context so far
// and decodes serially. Emits one "turn" event per (agent,turn). This is a worst-case
// REFERENCE only (the cold re-prefill loop) — NOT a serving baseline anyone ships; the
// headline race is fak vs the tuned warm-cache arm B above.
func liveArmA(l *loaded, P, T, C, D, R int, emit emitter) (totalMS, decodeMS float64) {
	m, vocab := l.m, l.vocab
	prefix := lcgIDs(P, vocab, 1)
	ids0 := lcgIDs(C, vocab, 991)
	start := time.Now()

	var dcAcc, prefilledSoFar float64
	done := 0
	for a := 0; a < C; a++ {
		ctx := append([]int(nil), prefix...)
		tok := ids0[a]
		for t := 0; t < T; t++ {
			s := m.NewSession()
			s.Quant = true
			s.Prefill(ctx) // re-prefill the WHOLE context so far
			prefilledSoFar += float64(len(ctx))
			t1 := time.Now()
			for d := 0; d < D; d++ {
				s.Step(tok)
				ctx = append(ctx, tok)
				tok = (tok*48271 + 1) % vocab
			}
			dcAcc += ms(time.Since(t1))
			s.Close()
			if t < T-1 {
				ctx = append(ctx, lcgIDs(R, vocab, uint64(50000+t*1000+a*97))...)
			}
			done++
			emit(event{
				"type": "turn", "arm": "naive", "turn": t, "agent": a,
				"requests_done": done, "total_requests": C * T,
				"tokens_prefilled": int(prefilledSoFar), "tokens_decoded": done * D,
				"elapsed_ms": ms(time.Since(start)),
			})
		}
	}
	totalMS = ms(time.Since(start))
	decodeMS = dcAcc
	return
}

// ---------------------------------------------------------------------------
// HTTP handlers
// ---------------------------------------------------------------------------

var (
	reg   *registry
	runMu sync.Mutex // only one heavy run at a time (avoids CPU-contending two big runs)
	specs []spec
	gomax = runtime.GOMAXPROCS(0)
)

type sseWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
}

func (s *sseWriter) emit(e event) {
	b, _ := json.Marshal(e)
	fmt.Fprintf(s.w, "data: %s\n\n", b)
	s.flusher.Flush()
}

func handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	b, err := pageFS.ReadFile("page.html")
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(b)
}

func handleLadder(w http.ResponseWriter, r *http.Request) {
	type rs struct {
		Name    string `json:"name"`
		Dir     string `json:"dir"`
		Kind    string `json:"kind"`
		Present bool   `json:"present"`
		Params  string `json:"params"`
	}
	out := []rs{}
	for _, s := range specs {
		out = append(out, rs{s.Name, s.Dir, s.Kind, s.Present, s.Params})
	}
	// timing-free work-elimination ratio for the default workload, as the honest floor.
	// HEADLINE = b/c (fak vs the tuned warm-cache baseline); a/c (vs the cold no-cache
	// loop) is the worst-case reference only.
	a, b, c := prefillTokens(512, 5, 5, 16, 32)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"models":     out,
		"gomaxprocs": gomax,
		"hardware":   demoui.Probe(), // cores / workers / accelerator the demo actually runs on

		"prefill_tok_ratio": map[string]any{
			"a": a, "b": b, "c": c,
			"b_over_c": float64(b) / float64(c), "a_over_c": float64(a) / float64(c),
			"note": "exact, timing-free prefill-token work-elimination. HEADLINE b/c = fak vs a tuned warm-cache (per-agent KV) baseline — the real serving baseline (vLLM/SGLang prefix caching, provider prompt-caching). a/c = vs the cold no-cache re-prefill loop, a worst-case reference, NOT a serving baseline.",
		},
	})
}

// args pulls P,T,C,D,R from the query (with defaults).
func args(r *http.Request) (P, T, C, D, R int) {
	P, _ = strconv.Atoi(r.URL.Query().Get("P"))
	T, _ = strconv.Atoi(r.URL.Query().Get("T"))
	C, _ = strconv.Atoi(r.URL.Query().Get("C"))
	D, _ = strconv.Atoi(r.URL.Query().Get("D"))
	R, _ = strconv.Atoi(r.URL.Query().Get("R"))
	if P == 0 {
		P = 512
	}
	if T == 0 {
		T = 5
	}
	if C == 0 {
		C = 5
	}
	if D == 0 {
		D = 16
	}
	if R == 0 {
		R = 32
	}
	return
}

func findSpec(name string) (spec, bool) {
	for _, s := range specs {
		if s.Name == name {
			return s, true
		}
	}
	return spec{}, false
}

// handleRace streams a live A-vs-C race for one model. fak arm runs first (fast), then the
// naive arm (the long grind) — both fully live, same model.
func handleRace(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", 500)
		return
	}
	name := r.URL.Query().Get("model")
	if name == "" {
		name = "SmolLM2-135M"
	}
	sp, ok := findSpec(name)
	if !ok || !sp.Present {
		http.Error(w, "unknown/absent model "+name, 400)
		return
	}
	P, T, C, D, R := args(r)

	runMu.Lock()
	defer runMu.Unlock()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	s := &sseWriter{w, flusher}

	hw := demoui.Probe()
	s.emit(event{"type": "hw", "hardware": hw})
	s.emit(event{"type": "info", "msg": "loading model " + name + " on " + hw.Summary})
	// Model load + quantize is the longest blocking phase (tens of seconds on the big
	// rungs); heartbeat it so the page shows a live elapsed counter instead of freezing.
	var l *loaded
	var err error
	demoui.Beat(heartbeat,
		func(el time.Duration) {
			s.emit(event{"type": "tick", "phase": "loading model " + name, "elapsed_ms": ms(el)})
		},
		func() { l, err = reg.get(sp) },
	)
	if err != nil {
		s.emit(event{"type": "error", "msg": "load: " + err.Error()})
		return
	}
	s.emit(event{"type": "info", "msg": fmt.Sprintf("model loaded (%s). workload: P=%d T=%d C=%d D=%d R=%d → %d requests", name, P, T, C, D, R, C*T)})
	runtime.GC()

	// warm
	ws := l.m.NewSession()
	ws.Quant = true
	ws.Prefill(lcgIDs(8, l.vocab, 77))
	ws.Step(1)
	ws.Close()

	// fak arm (live)
	s.emit(event{"type": "start", "arm": "fak", "total_requests": C * T})
	cTotal, cDecode := liveArmC(l, P, T, C, D, R, s.emit)
	s.emit(event{"type": "done", "arm": "fak", "total_ms": cTotal, "decode_ms": cDecode, "tokens_prefilled": P + C*(T-1)*R})

	// tuned warm-cache arm (live) — the SOTA serving baseline (the HEADLINE comparison)
	s.emit(event{"type": "start", "arm": "tuned", "total_requests": C * T})
	bTotal, bDecode := liveArmB(l, P, T, C, D, R, s.emit)
	s.emit(event{"type": "done", "arm": "tuned", "total_ms": bTotal, "decode_ms": bDecode, "tokens_prefilled": C * (P + (T-1)*R)})

	// naive arm (live) — the cold re-prefill REFERENCE (the long grind; not a serving baseline)
	s.emit(event{"type": "start", "arm": "naive", "total_requests": C * T})
	aTotal, aDecode := liveArmA(l, P, T, C, D, R, s.emit)
	s.emit(event{"type": "done", "arm": "naive", "total_ms": aTotal, "decode_ms": aDecode})

	// HEADLINE = B/C: fak vs the tuned warm-cache baseline. naive_ratio (A/C) is the
	// worst-case reference, surfaced but never the headline.
	ratio := bTotal / cTotal
	s.emit(event{"type": "result", "fak_ms": cTotal, "tuned_ms": bTotal, "naive_ms": aTotal,
		"ratio": ratio, "naive_ratio": aTotal / cTotal, "saved_ms": bTotal - cTotal,
		"fak_decode_ms": cDecode, "tuned_decode_ms": bDecode, "naive_decode_ms": aDecode,
		"model": name, "workload": map[string]int{"P": P, "T": T, "C": C, "D": D, "R": R}})
}

// curveProgress turns an arm's per-turn events into a live curve phase line ("running
// fak arm — 12/15 requests · 1.2k tok/s") so the arm runs visibly instead of as a
// silent block. It reads the raw Go-typed event the arm emits in-process (no JSON
// round-trip), so the int/float fields assert cleanly.
func curveProgress(s *sseWriter, model, arm string) emitter {
	return func(e event) {
		if e["type"] != "turn" {
			return
		}
		done, _ := e["requests_done"].(int)
		total, _ := e["total_requests"].(int)
		dec, _ := e["tokens_decoded"].(int)
		el, _ := e["elapsed_ms"].(float64)
		tps := 0.0
		if el > 0 {
			tps = float64(dec) / (el / 1000)
		}
		s.emit(event{"type": "phase", "model": model,
			"phase": fmt.Sprintf("running %s arm — %d/%d requests · %s", arm, done, total, tokPerSec(tps))})
	}
}

// tokPerSec formats a tokens/sec rate compactly (e.g. "1.2k tok/s", "840 tok/s").
func tokPerSec(tps float64) string {
	if tps >= 1000 {
		return fmt.Sprintf("%.1fk tok/s", tps/1000)
	}
	return fmt.Sprintf("%.0f tok/s", tps)
}

// handleCurve sweeps the ladder: per model, run arm C live + measure prefill cost to PROJECT
// arm A (the honest sessionbench method — naive re-prefill is intractable to run live for big
// models). Emits one curve point per model, then "curvedone".
func handleCurve(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", 500)
		return
	}
	// curve uses a smaller, tractable workload by default (big models are slow on CPU), unless overridden
	P, T, C, D, R := args(r)
	curveDefault := r.URL.Query().Get("P") == ""

	runMu.Lock()
	defer runMu.Unlock()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	s := &sseWriter{w, flusher}

	if curveDefault {
		P, T, C, D, R = 128, 5, 3, 16, 16
	}
	maxCtx := P + (T-1)*(D+R)
	s.emit(event{"type": "info", "msg": fmt.Sprintf("curve sweep: P=%d T=%d C=%d D=%d R=%d (maxCtx=%d). HEADLINE fak vs tuned warm-cache — BOTH run LIVE; the cold no-cache arm is PROJECTED as a faded reference only.", P, T, C, D, R, maxCtx)})

	for _, sp := range specs {
		if !sp.Present {
			continue
		}
		// model load + quantize — heartbeat so a multi-second load shows a live counter.
		var l *loaded
		var err error
		demoui.Beat(heartbeat,
			func(el time.Duration) {
				s.emit(event{"type": "phase", "model": sp.Name, "phase": fmt.Sprintf("loading · %.1fs", el.Seconds())})
			},
			func() { l, err = reg.get(sp) },
		)
		if err != nil {
			s.emit(event{"type": "point", "model": sp.Name, "params": sp.Params, "error": err.Error()})
			continue
		}
		// warm
		ws := l.m.NewSession()
		ws.Quant = true
		ws.Prefill(lcgIDs(8, l.vocab, 77))
		ws.Step(1)
		ws.Close()

		// prefill cost samples spanning [P, maxCtx] for the naive projection — heartbeat it,
		// the sweep over sample lengths is otherwise a silent multi-second stretch.
		lens := sampleLens(P, maxCtx)
		var pm prefillModel
		demoui.Beat(heartbeat,
			func(el time.Duration) {
				s.emit(event{"type": "phase", "model": sp.Name, "phase": fmt.Sprintf("measuring prefill cost (%d sample lengths) · %.1fs", len(lens), el.Seconds())})
			},
			func() { pm = measurePrefill(l.m, l.vocab, lens, 1) },
		)

		// fak arm, LIVE — feed per-turn progress to the phase line so the arm run is no
		// longer a silent block (was a no-op emitter): the viewer sees it advance + tok/s.
		s.emit(event{"type": "phase", "model": sp.Name, "phase": "running fak arm live"})
		cTotal, cDecode := liveArmC(l, P, T, C, D, R, curveProgress(s, sp.Name, "fak"))

		// tuned warm-cache arm, LIVE — the SOTA baseline. It is cheap to run (prefix once
		// per agent + incremental ingestion, no quadratic re-prefill), so the HEADLINE
		// fak-vs-tuned ratio is measured end-to-end, never projected.
		s.emit(event{"type": "phase", "model": sp.Name, "phase": "running tuned warm-cache arm live"})
		bTotal, _ := liveArmB(l, P, T, C, D, R, curveProgress(s, sp.Name, "tuned warm-cache"))

		// naive cold re-prefill arm — PROJECTED reference only (quadratic in T, intractable
		// to run live on the big rungs). Anchored to the live prefix prefill so cross-time
		// contention does not bias it.
		anchor := 1.0
		if base := pm.cost(P); base > 0 {
			anchor = cTotal / (base + 1) // coarse; armC total includes decode too — keep within reason
			anchor = math.Max(0.5, math.Min(anchor, 2.0))
		}
		aPrefill := computeAPrefill(pm, P, T, C, D, R) * anchor
		// naive decode = C× the fak batched decode (serial C agents vs one batched stream)
		aDecode := float64(C) * cDecode
		aTotal := aPrefill + aDecode
		if cTotal > 0 && aTotal < cTotal {
			aTotal = cTotal // never let projection undersell the floor
		}
		ratio := bTotal / cTotal // HEADLINE: fak vs the tuned warm-cache baseline (both live)

		s.emit(event{
			"type": "point", "model": sp.Name, "params": sp.Params,
			"armC_ms": cTotal, "armB_ms": bTotal, "armA_ms": aTotal,
			"ratio": ratio, "naive_ratio": aTotal / cTotal,
			"armC_decode_ms": cDecode, "armA_prefill_ms": aPrefill, "armA_decode_ms": aDecode,
			"prefill_samples": pm, "live": true,
		})
		runtime.GC()
	}
	s.emit(event{"type": "curvedone"})
}

func sampleLens(P, maxCtx int) []int {
	set := map[int]bool{}
	add := func(x int) {
		if x >= 16 {
			set[x] = true
		}
	}
	add(P)
	if maxCtx > P {
		for _, f := range []float64{0.34, 0.67, 1.0} {
			add(P + int(f*float64(maxCtx-P)))
		}
	}
	var out []int
	for k := range set {
		out = append(out, k)
	}
	sort.Ints(out)
	return out
}

func main() {
	addr := flag.String("addr", "127.0.0.1:8147", "listen address")
	jobs := flag.Int("jobs", 0, "cap parallelism to an ABSOLUTE core count (GOMAXPROCS + matmul workers). 0 = all cores. On a shared/active machine pass e.g. 8 so the demo doesn't starve other work.")
	budget := flag.Float64("budget", 0, "cap parallelism to a FRACTION of the machine: 0.75 = 75% of the logical cores (portable across box sizes; 75 or 0.75 accepted). Mutually exclusive with -jobs. 0 = unset.")
	flag.Parse()
	if *jobs > 0 && *budget > 0 {
		fmt.Fprintln(os.Stderr, "-jobs and -budget are mutually exclusive (one is absolute, the other a fraction)")
		os.Exit(2)
	}
	if *jobs > 0 {
		runtime.GOMAXPROCS(*jobs)
		// Route through the model package so numWorkers actually changes — setting the env
		// here is too late (numWorkers was resolved at package init, before this).
		if err := model.SetWorkers(*jobs); err != nil {
			fmt.Fprintln(os.Stderr, "jobs:", err)
			os.Exit(2)
		}
		gomax = *jobs
	} else if *budget > 0 {
		if err := model.SetWorkerBudget(*budget); err != nil {
			fmt.Fprintln(os.Stderr, "budget:", err)
			os.Exit(2)
		}
		// also bound the Go scheduler to the resolved width so the demo's own goroutines
		// don't oversubscribe the share we promised to leave free.
		runtime.GOMAXPROCS(model.NumWorkers())
		gomax = model.NumWorkers()
	}
	reg = newRegistry()
	specs = ladderSpecs()

	mux := http.NewServeMux()
	mux.HandleFunc("/", handleIndex)
	mux.HandleFunc("/api/ladder", handleLadder)
	mux.HandleFunc("/api/race", handleRace)
	mux.HandleFunc("/api/curve", handleCurve)

	present := []string{}
	for _, s := range specs {
		if s.Present {
			present = append(present, s.Name)
		}
	}
	fmt.Fprintf(os.Stderr, "demorace %s on http://%s (GOMAXPROCS=%d)\n", version, listenAddr(*addr), gomax)
	fmt.Fprintf(os.Stderr, "ladder present: %s\n", strings.Join(present, ", "))
	fmt.Fprintf(os.Stderr, "open the URL → 'Run live race' (HEADLINE fak vs tuned warm-cache, both LIVE; cold no-cache shown as reference) then 'Build curve'\n")
	if err := http.ListenAndServe(listenAddr(*addr), mux); err != nil {
		fmt.Fprintln(os.Stderr, "listen:", err)
		os.Exit(1)
	}
}

// listenAddr honors the $PORT contract used by container/VM platforms: when PORT is
// set in the environment, bind 0.0.0.0:$PORT and ignore the -addr loopback default,
// so the same binary serves locally (-addr) and on a public host (PORT=… + an open
// firewall) with no rebuild. A non-default -addr still wins, so an explicit local
// override is never silently lost.
func listenAddr(addr string) string {
	if p := os.Getenv("PORT"); p != "" && addr == "127.0.0.1:8147" {
		return "0.0.0.0:" + p
	}
	return addr
}

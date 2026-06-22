// Command ctxdemo is the live, on-box demo of fak's value in the MULTI-AGENT,
// MULTI-TURN, LONG-CONTEXT regime — the one where the context CHANGES every turn as
// tool calls land heterogeneous, variable-sized results.
//
// Where cmd/demorace races one fixed session shape with a single constant tool-result
// size R, ctxdemo is built around a CATALOG of agentic SHAPES ("scenarios" — the
// modules): support bot, coding agent, deep research, mixed fleet. Each scenario is a
// different point in (prefix size, agent count, turn count, tool-result distribution)
// space, and each agent's every turn appends a DIFFERENT-sized tool result, so the
// running context grows unevenly, agent by agent. That is the "tool-call context
// changing" dimension this demo exists to make visible.
//
// For the selected scenario it shows three things:
//   - the EXACT, timing-free prefill-token work each strategy performs — the
//     load-independent honest floor. The comparison baseline is a WARM per-agent KV
//     cache (each agent already reuses its own KV across turns); fak's win is
//     cross-agent prefix sharing ON TOP of that warm cache. The cold no-cache
//     re-prefill is a labelled worst-case reference only;
//   - a per-agent CONTEXT TIMELINE — every agent's turns, with each tool result drawn
//     to scale, so you can see the long context assemble unevenly; and
//   - a LIVE wall-clock race of the same session through a real in-kernel model: the
//     HEADLINE is fak vs the tuned warm-cache baseline (the SOTA serving baseline a
//     prefix-caching stack gives you), with the cold no-cache loop shown only as a
//     dim worst-case reference, never the headline.
//
// Same model, same tokens, same answers — the only difference is how much shared
// setup the system makes the model re-read. That is the whole fak thesis.
//
// Serve it:
//
//	go run ./cmd/ctxdemo -addr 127.0.0.1:8153 -jobs 8
//	# open http://127.0.0.1:8153 → pick a scenario → "Run live race"
//
// Headless (no browser, no model — instant exact accounting, CI-usable):
//
//	go run ./cmd/ctxdemo -print
//	go run ./cmd/ctxdemo -print -json
//
// Headless live race of one scenario (needs a model on disk; shrink with -P/-C/-T/-D):
//
//	go run ./cmd/ctxdemo -race deep-research -jobs 8 -C 2 -T 3
package main

import (
	"embed"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

//go:embed page.html
var pageFS embed.FS

const version = "fak-ctxdemo-v1"

// projectThreshold: if the cold no-cache (naive) reference arm would re-prefill more
// than this many tokens, project it from the fak arm's measured throughput instead of
// running the (multi-minute) live grind. Small scenarios stay fully live so both arms
// can be watched.
const projectThreshold = 9000

// ---------------------------------------------------------------------------
// model ladder (same detection demorace uses; kept local so the demos stay
// independent commands)
// ---------------------------------------------------------------------------

type spec struct {
	Name    string
	Dir     string
	Kind    string // "dir" (fak export) or "hf" (safetensors snapshot, lean Q8 load)
	Params  string
	Order   int
	Present bool
}

func ladderSpecs() []spec {
	home, _ := os.UserHomeDir()
	hf := filepath.Join(home, ".cache", "fak-models", "hf")
	candidates := []spec{
		{Name: "SmolLM2-135M", Params: "0.14B", Order: 0, Kind: "dir", Dir: "internal/model/.cache/smollm2-135m"},
		{Name: "Qwen2.5-0.5B", Params: "0.5B", Order: 1, Kind: "hf", Dir: filepath.Join(hf, "Qwen2.5-0.5B-Instruct")},
		{Name: "Qwen2.5-1.5B", Params: "1.5B", Order: 2, Kind: "hf", Dir: filepath.Join(hf, "Qwen2.5-1.5B-Instruct")},
		{Name: "Qwen2.5-3B", Params: "3B", Order: 3, Kind: "hf", Dir: filepath.Join(hf, "Qwen2.5-3B-Instruct")},
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

func smallestPresent() (spec, bool) {
	for _, s := range specs {
		if s.Present {
			return s, true
		}
	}
	return spec{}, false
}

type loaded struct {
	m     *model.Model
	vocab int
	name  string
}

type registry struct {
	mu    sync.Mutex
	cache map[string]*loaded
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
		if _, err := os.Stat(filepath.Join(s.Dir, "model.safetensors.index.json")); err == nil {
			return model.LoadSafetensorsQuantDir(s.Dir, cfg)
		}
		return model.LoadSafetensorsQuant(filepath.Join(s.Dir, "model.safetensors"), cfg)
	default:
		return model.Load(s.Dir)
	}
}

// ---------------------------------------------------------------------------
// shared globals
// ---------------------------------------------------------------------------

var (
	reg   *registry
	specs []spec
	runMu sync.Mutex // one heavy run at a time, so two races don't contend the box
	gomax = runtime.GOMAXPROCS(0)
)

// scenarioView is the JSON shape the page renders: the scenario, its workload matrix,
// and the exact timing-free token accounting (the honest floor).
type scenarioView struct {
	Scenario Scenario `json:"scenario"`
	Workload Workload `json:"workload"`
	Tokens   tokens   `json:"tokens"`
}

type tokens struct {
	NaiveReprefill int     `json:"naive_reprefill"`
	PerAgentKV     int     `json:"per_agent_kv"`
	FakFused       int     `json:"fak_fused"`
	NaiveOverFak   float64 `json:"naive_over_fak"`
	TunedOverFak   float64 `json:"tuned_over_fak"`
	ResultTokens   int     `json:"result_tokens"`
	MaxContext     int     `json:"max_context"`
}

func viewOf(s Scenario) scenarioView {
	w := s.Build()
	a, b, c := w.prefillTokens()
	// the longest single context any agent reaches (prefix + its decode + its results)
	maxCtx := 0
	for ci := 0; ci < s.Agents; ci++ {
		ctx := s.Prefix
		for t := 0; t < len(w.Results[ci]); t++ {
			ctx += s.Decode + w.Results[ci][t]
		}
		ctx += s.Decode // final turn's decode
		if ctx > maxCtx {
			maxCtx = ctx
		}
	}
	return scenarioView{
		Scenario: s, Workload: w,
		Tokens: tokens{
			NaiveReprefill: a, PerAgentKV: b, FakFused: c,
			NaiveOverFak: ratio(a, c), TunedOverFak: ratio(b, c),
			ResultTokens: w.totalResultTokens(), MaxContext: maxCtx,
		},
	}
}

func ratio(n, d int) float64 {
	if d == 0 {
		return 0
	}
	return float64(n) / float64(d)
}

// ---------------------------------------------------------------------------
// HTTP handlers
// ---------------------------------------------------------------------------

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

// handleScenarios returns the whole catalog (each with its exact token accounting and
// workload matrix) plus the detected model ladder. The page renders the modular
// overview and the per-agent context timeline from this — no model run required.
func handleScenarios(w http.ResponseWriter, r *http.Request) {
	views := make([]scenarioView, 0)
	for _, s := range catalog() {
		views = append(views, viewOf(s))
	}
	type ms struct {
		Name, Params string
		Present      bool
	}
	models := []ms{}
	for _, s := range specs {
		models = append(models, ms{s.Name, s.Params, s.Present})
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"scenarios":  views,
		"models":     models,
		"gomaxprocs": gomax,
	})
}

func intArg(r *http.Request, k string) int {
	v, _ := strconv.Atoi(r.URL.Query().Get(k))
	return v
}

// handleRace streams a live A-vs-C race for one scenario. The fak arm always runs
// live; the naive arm runs live for small scenarios and is projected (from the fak
// arm's measured throughput) for the long ones, so even deep-research stays tractable.
func handleRace(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", 500)
		return
	}
	id := r.URL.Query().Get("scenario")
	if id == "" {
		id = "support-bot"
	}
	scn, ok := findScenario(id)
	if !ok {
		http.Error(w, "unknown scenario "+id, 400)
		return
	}
	scn = scn.scale(intArg(r, "P"), intArg(r, "C"), intArg(r, "T"), intArg(r, "D"))

	modelName := r.URL.Query().Get("model")
	var sp spec
	if modelName == "" {
		var has bool
		if sp, has = smallestPresent(); !has {
			http.Error(w, "no model present on disk (need a ladder model)", 400)
			return
		}
	} else {
		var found bool
		for _, s := range specs {
			if s.Name == modelName {
				sp, found = s, true
			}
		}
		if !found || !sp.Present {
			http.Error(w, "unknown/absent model "+modelName, 400)
			return
		}
	}
	naiveMode := r.URL.Query().Get("naive") // "", "live", "project"

	runMu.Lock()
	defer runMu.Unlock()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	s := &sseWriter{w, flusher}
	runRace(scn, sp, naiveMode, s.emit)
}

// runRace is the shared race engine used by the HTTP handler and the headless -race.
func runRace(scn Scenario, sp spec, naiveMode string, emit emitter) {
	v := viewOf(scn)
	emit(event{"type": "plan", "view": v, "model": sp.Name})

	emit(event{"type": "info", "msg": "loading model " + sp.Name})
	l, err := reg.get(sp)
	if err != nil {
		emit(event{"type": "error", "msg": "load: " + err.Error()})
		return
	}
	runtime.GC()
	warm(l)

	w := v.Workload
	// fak arm (live)
	emit(event{"type": "start", "arm": "fak", "total_requests": scn.Agents * scn.Turns})
	fak := liveArmFak(l, w, emit)
	emit(event{"type": "done", "arm": "fak", "total_ms": fak.totalMS, "decode_ms": fak.decodeMS,
		"tokens_prefilled": fak.prefillTk})

	// tuned warm-cache arm (live) — the SOTA serving baseline (the HEADLINE comparison).
	// Tractable to run live always (no quadratic re-prefill).
	emit(event{"type": "start", "arm": "tuned", "total_requests": scn.Agents * scn.Turns})
	tuned := liveArmTuned(l, w, emit)
	emit(event{"type": "done", "arm": "tuned", "total_ms": tuned.totalMS, "decode_ms": tuned.decodeMS,
		"tokens_prefilled": tuned.prefillTk})

	// naive cold re-prefill arm — worst-case REFERENCE only (live for small scenarios,
	// projected for the long grind).
	project := naiveMode == "project"
	if naiveMode != "live" && v.Tokens.NaiveReprefill > projectThreshold {
		project = true
	}
	var naive armResult
	emit(event{"type": "start", "arm": "naive", "total_requests": scn.Agents * scn.Turns, "projected": project})
	if project {
		emit(event{"type": "info", "msg": fmt.Sprintf("cold no-cache reference arm projected from fak's measured throughput (%d re-prefill tokens would grind for minutes live)", v.Tokens.NaiveReprefill)})
		naive = projectNaive(w, fak, emit)
	} else {
		naive = liveArmNaive(l, w, emit)
	}
	emit(event{"type": "done", "arm": "naive", "total_ms": naive.totalMS, "decode_ms": naive.decodeMS,
		"tokens_prefilled": naive.prefillTk, "projected": project})

	// HEADLINE = tuned/fak: fak vs the tuned warm-cache SOTA baseline. naive_ratio (A/C) is
	// the worst-case reference, surfaced but never the headline.
	r := tuned.totalMS / fak.totalMS
	emit(event{"type": "result",
		"fak_ms": fak.totalMS, "tuned_ms": tuned.totalMS, "naive_ms": naive.totalMS,
		"ratio": r, "naive_ratio": naive.totalMS / fak.totalMS,
		"saved_ms": tuned.totalMS - fak.totalMS,
		"naive_projected": project, "scenario": scn.ID, "model": sp.Name,
		"tokens": v.Tokens})
}

// ---------------------------------------------------------------------------
// headless modes
// ---------------------------------------------------------------------------

// printCatalog dumps every scenario's exact, timing-free token accounting. No model,
// no browser, instant — the honest floor on its own, usable in CI.
func printCatalog(asJSON bool) {
	views := make([]scenarioView, 0)
	for _, s := range catalog() {
		views = append(views, viewOf(s))
	}
	if asJSON {
		json.NewEncoder(os.Stdout).Encode(views)
		return
	}
	fmt.Printf("fak · ctxdemo — exact, timing-free prefill-token work per scenario module\n")
	fmt.Printf("(baseline = a WARM per-agent KV cache; fak adds cross-agent prefix sharing on top.\n")
	fmt.Printf(" the cold no-cache column is a worst-case reference only. decode excluded — generated, not re-read)\n\n")
	fmt.Printf("%-15s %4s %4s %4s  %9s %9s %9s   %8s %7s  %8s\n",
		"scenario", "C", "T", "P", "no-cache", "warmKV", "fak", "fak-win", "(ref×)", "maxCtx")
	fmt.Printf("%s\n", strings.Repeat("-", 92))
	for _, v := range views {
		s, t := v.Scenario, v.Tokens
		fmt.Printf("%-15s %4d %4d %4d  %9d %9d %9d   %7.1f× %6.1f×  %8d\n",
			s.ID, s.Agents, s.Turns, s.Prefix,
			t.NaiveReprefill, t.PerAgentKV, t.FakFused, t.TunedOverFak, t.NaiveOverFak, t.MaxContext)
	}
	fmt.Printf("\nfak-win = WARM per-agent KV cache ÷ fak — fak's cross-agent prefix-sharing win (the honest baseline).\n")
	fmt.Printf("(ref×)  = cold no-cache re-prefill ÷ fak — a worst-case reference, NOT a serving baseline.\n")
}

// ---------------------------------------------------------------------------

func applyResourceCaps(jobs int, budget float64) {
	if jobs > 0 && budget > 0 {
		fmt.Fprintln(os.Stderr, "-jobs and -budget are mutually exclusive (one is absolute, the other a fraction)")
		os.Exit(2)
	}
	if jobs > 0 {
		runtime.GOMAXPROCS(jobs)
		if err := model.SetWorkers(jobs); err != nil {
			fmt.Fprintln(os.Stderr, "jobs:", err)
			os.Exit(2)
		}
		gomax = jobs
	} else if budget > 0 {
		if err := model.SetWorkerBudget(budget); err != nil {
			fmt.Fprintln(os.Stderr, "budget:", err)
			os.Exit(2)
		}
		runtime.GOMAXPROCS(model.NumWorkers())
		gomax = model.NumWorkers()
	}
}

func main() {
	addr := flag.String("addr", "127.0.0.1:8153", "listen address")
	jobs := flag.Int("jobs", 0, "cap parallelism to an ABSOLUTE core count (GOMAXPROCS + matmul workers). 0 = all cores. On a shared/active box pass e.g. 8 so the demo doesn't starve other work.")
	budget := flag.Float64("budget", 0, "cap parallelism to a FRACTION of the machine: 0.75 = 75% of the logical cores (75 or 0.75 accepted). Mutually exclusive with -jobs.")
	print := flag.Bool("print", false, "headless: print every scenario's exact timing-free token accounting and exit (no model, no server)")
	asJSON := flag.Bool("json", false, "with -print, emit JSON instead of a table")
	race := flag.String("race", "", "headless: run a live race for this scenario id and exit (needs a model on disk)")
	naive := flag.String("naive", "", "race naive arm mode: live | project (default: auto — live when small, projected when the grind would be minutes)")
	P := flag.Int("P", 0, "override prefix tokens (0 = scenario default)")
	C := flag.Int("C", 0, "override agent count")
	T := flag.Int("T", 0, "override turn count")
	D := flag.Int("D", 0, "override decode tokens/turn")
	flag.Parse()

	specs = ladderSpecs()

	if *print {
		printCatalog(*asJSON)
		return
	}

	applyResourceCaps(*jobs, *budget)
	reg = newRegistry()

	if *race != "" {
		scn, ok := findScenario(*race)
		if !ok {
			fmt.Fprintf(os.Stderr, "unknown scenario %q (try: support-bot, coding-agent, deep-research, mixed-fleet)\n", *race)
			os.Exit(2)
		}
		scn = scn.scale(*P, *C, *T, *D)
		sp, has := smallestPresent()
		if !has {
			fmt.Fprintln(os.Stderr, "no model present on disk — need a ladder model (see cmd/demorace ladder)")
			os.Exit(1)
		}
		emit := func(e event) {
			if t, _ := e["type"].(string); t == "result" {
				b, _ := json.MarshalIndent(e, "", "  ")
				fmt.Println(string(b))
			} else if t == "info" || t == "done" || t == "error" {
				b, _ := json.Marshal(e)
				fmt.Fprintln(os.Stderr, string(b))
			}
		}
		t0 := time.Now()
		runRace(scn, sp, *naive, emit)
		fmt.Fprintf(os.Stderr, "race wall-clock: %.1fs (GOMAXPROCS=%d)\n", time.Since(t0).Seconds(), gomax)
		return
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", handleIndex)
	mux.HandleFunc("/api/scenarios", handleScenarios)
	mux.HandleFunc("/api/race", handleRace)

	present := []string{}
	for _, s := range specs {
		if s.Present {
			present = append(present, s.Name+" ("+s.Params+")")
		}
	}
	fmt.Fprintf(os.Stderr, "ctxdemo %s on http://%s (GOMAXPROCS=%d)\n", version, listenAddr(*addr), gomax)
	if len(present) == 0 {
		fmt.Fprintf(os.Stderr, "ladder: NONE present — the live race needs a model on disk; -print still works\n")
	} else {
		fmt.Fprintf(os.Stderr, "ladder present: %s\n", strings.Join(present, ", "))
	}
	fmt.Fprintf(os.Stderr, "open the URL → pick a scenario → 'Run live race'. Headless: -print (instant token accounting).\n")
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
	if p := os.Getenv("PORT"); p != "" && addr == "127.0.0.1:8153" {
		return "0.0.0.0:" + p
	}
	return addr
}

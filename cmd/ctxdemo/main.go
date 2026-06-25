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
//     cross-agent prefix sharing ON TOP of that warm cache;
//   - a per-agent CONTEXT TIMELINE — every agent's turns, with each tool result drawn
//     to scale, so you can see the long context assemble unevenly; and
//   - a LIVE wall-clock race of the same session through a real in-kernel model: the
//     HEADLINE is fak vs the tuned warm-cache baseline (the SOTA serving baseline a
//     prefix-caching stack gives you).
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
//	go run ./cmd/ctxdemo -bars                      # the reuse axis as a side-by-side bar chart
//	go run ./cmd/ctxdemo -bars -scenario deep-research
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

	"github.com/anthony-chaudhary/fak/internal/demoui"
	"github.com/anthony-chaudhary/fak/internal/model"
)

//go:embed page.html
var pageFS embed.FS

const version = "fak-ctxdemo-v1"

// heartbeat is how often a blocking phase (model load + quantize) emits a keep-alive
// "tick" so the page updates ~1×/s instead of freezing on a long load (tens of seconds
// on the big rungs). Same cadence demorace uses.
const heartbeat = 700 * time.Millisecond

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
	PerAgentKV   int     `json:"per_agent_kv"`
	FakFused     int     `json:"fak_fused"`
	TunedOverFak float64 `json:"tuned_over_fak"`
	ResultTokens int     `json:"result_tokens"`
	MaxContext   int     `json:"max_context"`
}

func viewOf(s Scenario) scenarioView {
	w := s.Build()
	_, b, c := w.prefillTokens()
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
			PerAgentKV: b, FakFused: c,
			TunedOverFak: ratio(b, c),
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
		"hardware":   demoui.Probe(), // cores / workers / accelerator the demo actually runs on
	})
}

func intArg(r *http.Request, k string) int {
	v, _ := strconv.Atoi(r.URL.Query().Get(k))
	return v
}

// handleRace streams a live race of fak vs the tuned warm-cache baseline (SOTA) for
// one scenario. Both arms run fully live — the warm-cache arm has no quadratic
// re-prefill, so even deep-research stays tractable.
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
	runMu.Lock()
	defer runMu.Unlock()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	s := &sseWriter{w, flusher}
	runRace(scn, sp, s.emit)
}

// runRace is the shared race engine used by the HTTP handler and the headless -race.
func runRace(scn Scenario, sp spec, emit emitter) {
	v := viewOf(scn)
	emit(event{"type": "plan", "view": v, "model": sp.Name})

	hw := demoui.Probe()
	emit(event{"type": "hw", "hardware": hw})
	emit(event{"type": "info", "msg": "loading model " + sp.Name + " on " + hw.Summary})
	// Model load + quantize is the longest blocking phase (tens of seconds on the big
	// rungs); heartbeat it so the page shows a live elapsed counter instead of freezing.
	var l *loaded
	var err error
	demoui.Beat(heartbeat,
		func(el time.Duration) {
			emit(event{"type": "tick", "phase": "loading model " + sp.Name, "elapsed_ms": ms(el)})
		},
		func() { l, err = reg.get(sp) },
	)
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

	// HEADLINE = tuned/fak: fak vs the tuned warm-cache SOTA baseline.
	r := tuned.totalMS / fak.totalMS
	emit(event{"type": "result",
		"fak_ms": fak.totalMS, "tuned_ms": tuned.totalMS,
		"ratio":    r,
		"saved_ms": tuned.totalMS - fak.totalMS,
		"scenario": scn.ID, "model": sp.Name,
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
	fmt.Printf(" decode excluded — generated, not re-read)\n\n")
	fmt.Printf("%-15s %4s %4s %4s  %9s %9s   %8s  %8s\n",
		"scenario", "C", "T", "P", "warmKV", "fak", "fak-win", "maxCtx")
	fmt.Printf("%s\n", strings.Repeat("-", 80))
	for _, v := range views {
		s, t := v.Scenario, v.Tokens
		fmt.Printf("%-15s %4d %4d %4d  %9d %9d   %7.1f×  %8d\n",
			s.ID, s.Agents, s.Turns, s.Prefix,
			t.PerAgentKV, t.FakFused, t.TunedOverFak, t.MaxContext)
	}
	fmt.Printf("\nfak-win = WARM per-agent KV cache ÷ fak — fak's cross-agent prefix-sharing win (the honest baseline).\n")
}

// ---------------------------------------------------------------------------
// -bars: the reuse axis as a SIDE-BY-SIDE bar chart — the 30-second point with
// zero setup. The token counts are exact and timing-free (no model), so each
// scenario's two bars (tuned warm-cache / fak) are drawn to scale and the headline
// ratio is the honest fak-vs-tuned win. (The terminal twin of cmd/guarddemo -print
// and cmd/turntaxdemo -print, for the reuse axis.)
// ---------------------------------------------------------------------------

type barPalette struct{ red, amber, green, dim, bold, reset string }

func barColors() barPalette {
	tty := false
	if fi, err := os.Stdout.Stat(); err == nil {
		tty = fi.Mode()&os.ModeCharDevice != 0
	}
	if os.Getenv("NO_COLOR") != "" || !tty {
		return barPalette{}
	}
	return barPalette{red: "\033[31m", amber: "\033[33m", green: "\033[32m", dim: "\033[2m", bold: "\033[1m", reset: "\033[0m"}
}

func (p barPalette) paint(code, s string) string {
	if code == "" {
		return s
	}
	return code + s + p.reset
}

// commaInt formats an int with thousands separators (Go has no built-in for this).
func commaInt(n int) string {
	s := strconv.Itoa(n)
	neg := strings.HasPrefix(s, "-")
	if neg {
		s = s[1:]
	}
	var out []byte
	for i, c := range []byte(s) {
		if i > 0 && (len(s)-i)%3 == 0 {
			out = append(out, ',')
		}
		out = append(out, c)
	}
	if neg {
		return "-" + string(out)
	}
	return string(out)
}

// bar returns a block-character bar of length proportional to n/max, capped at width
// (a non-zero value always shows at least one block so it is never invisible).
func bar(n, max, width int) string {
	if max <= 0 || n <= 0 {
		return ""
	}
	l := int(float64(n)/float64(max)*float64(width) + 0.5)
	if l < 1 {
		l = 1
	}
	if l > width {
		l = width
	}
	return strings.Repeat("█", l)
}

// printBars renders every scenario's reuse comparison as a side-by-side bar chart.
// scenarioID == "" renders the whole catalog; a non-empty id renders just that one.
func printBars(scenarioID string) int {
	p := barColors()
	scens := catalog()
	if scenarioID != "" {
		s, ok := findScenario(scenarioID)
		if !ok {
			fmt.Fprintf(os.Stderr, "unknown scenario %q (try: support-bot, coding-agent, deep-research, mixed-fleet)\n", scenarioID)
			return 2
		}
		scens = []Scenario{s}
	}
	const width = 42
	fmt.Printf("\n  %s\n", p.paint(p.bold, "fak · context reuse, side by side"))
	fmt.Printf("  %s\n", p.paint(p.dim, "prefill tokens the model must RE-READ per session — lower is better (decode excluded: it's generated, not re-read)"))
	for _, s := range scens {
		v := viewOf(s)
		t := v.Tokens
		max := t.PerAgentKV
		if t.FakFused > max {
			max = t.FakFused
		}
		fmt.Printf("\n  %s  %s\n",
			p.paint(p.bold, s.ID),
			p.paint(p.dim, fmt.Sprintf("(C=%d agents · T=%d turns · P=%d prefix · maxCtx=%s)", s.Agents, s.Turns, s.Prefix, commaInt(t.MaxContext))))
		row := func(color, label string, n int) {
			fmt.Printf("    %s  %s  %s\n",
				p.paint(p.dim, padTo(label, 26)),
				p.paint(color, padTo(bar(n, max, width), width)),
				p.paint(color, commaInt(n)))
		}
		row(p.amber, "tuned warm-cache (SOTA)", t.PerAgentKV)
		row(p.green, "fak (cross-agent reuse)", t.FakFused)
		fmt.Printf("    %s\n", p.paint(p.dim, fmt.Sprintf(
			"→ fak makes the model re-read %.1f× fewer tokens than even a tuned warm-cache stack.",
			t.TunedOverFak)))
	}
	fmt.Println()
	return 0
}

// padTo right-pads a plain string to w runes (alignment-safe: color is applied after).
func padTo(s string, w int) string {
	r := []rune(s)
	if len(r) >= w {
		return s
	}
	return s + strings.Repeat(" ", w-len(r))
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
	const defaultAddr = "127.0.0.1:8153"
	addr := flag.String("addr", defaultAddr, "listen address")
	basePath := demoui.BasePathFlag(flag.CommandLine, "/ctxdemo")
	jobs := flag.Int("jobs", 0, "cap parallelism to an ABSOLUTE core count (GOMAXPROCS + matmul workers). 0 = all cores. On a shared/active box pass e.g. 8 so the demo doesn't starve other work.")
	budget := flag.Float64("budget", 0, "cap parallelism to a FRACTION of the machine: 0.75 = 75% of the logical cores (75 or 0.75 accepted). Mutually exclusive with -jobs.")
	print := flag.Bool("print", false, "headless: print every scenario's exact timing-free token accounting and exit (no model, no server)")
	asJSON := flag.Bool("json", false, "with -print, emit JSON instead of a table")
	bars := flag.Bool("bars", false, "headless: render the prefill-token reuse comparison as a SIDE-BY-SIDE bar chart (no model, no server) and exit. The 30-second point with zero setup; honors NO_COLOR. -scenario picks one (default: all).")
	scenario := flag.String("scenario", "", "with -bars, render just this scenario id (e.g. deep-research); empty = all")
	race := flag.String("race", "", "headless: run a live race for this scenario id and exit (needs a model on disk)")
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

	if *bars {
		os.Exit(printBars(*scenario))
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
		runRace(scn, sp, emit)
		fmt.Fprintf(os.Stderr, "race wall-clock: %.1fs (GOMAXPROCS=%d)\n", time.Since(t0).Seconds(), gomax)
		return
	}

	app := http.NewServeMux()
	app.HandleFunc("/", handleIndex)
	app.HandleFunc("/api/scenarios", handleScenarios)
	app.HandleFunc("/api/race", handleRace)
	mux := http.NewServeMux()
	base := demoui.MountWithBasePath(mux, *basePath, app)

	present := []string{}
	for _, s := range specs {
		if s.Present {
			present = append(present, s.Name+" ("+s.Params+")")
		}
	}
	bind := demoui.ListenAddr(*addr, defaultAddr)
	fmt.Fprintf(os.Stderr, "ctxdemo %s on %s (GOMAXPROCS=%d)\n", version, demoui.LocalURL(bind, base), gomax)
	if base != "" {
		fmt.Fprintf(os.Stderr, "base path: %s (set by -base-path or %s)\n", base, demoui.DemoBasePathEnv)
	}
	if len(present) == 0 {
		fmt.Fprintf(os.Stderr, "ladder: NONE present — the live race needs a model on disk; -print still works\n")
	} else {
		fmt.Fprintf(os.Stderr, "ladder present: %s\n", strings.Join(present, ", "))
	}
	fmt.Fprintf(os.Stderr, "open the URL → pick a scenario → 'Run live race'. Headless: -print (instant token accounting).\n")
	if err := http.ListenAndServe(bind, mux); err != nil {
		fmt.Fprintln(os.Stderr, "listen:", err)
		os.Exit(1)
	}
}

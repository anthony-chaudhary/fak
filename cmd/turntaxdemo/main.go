// Command turntaxdemo is the live, on-box demo of how fak SAVES MODEL TURNS.
//
// It replays a frozen, class-labeled tool-call trace through the REAL kernel
// (internal/turnbench → k.Syscall), call by call, and streams each call's live
// verdict to the browser so two lanes advance visibly:
//
//	lane "baseline (SOTA loop)" — the documented two-pass agent loop: a malformed
//	  (aliased) arg errors and the model is re-prompted (+1 turn); a duplicate read
//	  is re-issued (+1 turn); a pure/static tool call round-trips to the engine
//	  (+1 turn the model could have elided). The lane's turn counter TICKS UP.
//	lane "fak (1-shot)"        — the kernel resolves the very same condition INSIDE
//	  the syscall the call arrived on (grammar repair / vDSO local serve), so no
//	  second model round-trip fires. The lane's turn counter STAYS FLAT.
//
// Unlike cmd/demorace (which needs model weights on disk), this demo is FULLY
// SELF-CONTAINED: the trace is replayed through the kernel, not a model, so it
// reproduces identically on any box with no downloads.
//
// The safety floor (a poisoned result quarantined, a destructive op denied) is shown
// on a DELIBERATELY SEPARATE axis — never folded into the turn count — mirroring the
// two-axes discipline of TURN-TAX-RESULTS.md (§1 the moat vs §3 the efficiency upside).
//
// Serve it:
//
//	go run ./cmd/turntaxdemo -addr 127.0.0.1:8150 -jobs 8
//	# open http://127.0.0.1:8150  → pick a suite → "Replay through the kernel"
//	#   turntax-airline → baseline +9 turns, fak 0   (every lever fires)
//	#   turntax-happy   → both stay at 0             (the anti-inflation control, watchable)
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
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/turnbench"

	// Blank-import the built-in driver list so the full ABI (resolver, vDSO,
	// adjudicator, ctx-MMU, normgate, ifc, witness, engines) is wired before
	// kernel.New / agent.Configure run inside turnbench.RunWithCalls.
	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

//go:embed page.html
var pageFS embed.FS

const version = "fak-turntaxdemo-v1"

var gomax = runtime.GOMAXPROCS(0)

// knownSuites are the fixtures shipped under testdata/turntax. The airline slice
// fires every lever once (baseline +9); the happy slice is the clean control (0).
var knownSuites = []struct{ ID, Label string }{
	{"turntax-airline", "airline support (every lever fires — baseline +9)"},
	{"turntax-happy", "happy path (clean — the anti-inflation control, 0)"},
}

// turnTaxDir resolves the trace fixtures the same way cmd/fak does: prefer a
// testdata/turntax under the working dir (running from fak/), else next to the
// executable, else the source-tree path when invoked via `go run ./cmd/turntaxdemo`.
func turnTaxDir() string {
	cands := []string{
		filepath.Join("testdata", "turntax"),
		filepath.Join("..", "..", "testdata", "turntax"),
	}
	if exe, err := os.Executable(); err == nil {
		cands = append(cands, filepath.Join(filepath.Dir(exe), "testdata", "turntax"))
	}
	for _, d := range cands {
		if _, err := os.Stat(d); err == nil {
			return d
		}
	}
	return cands[0]
}

func suitePath(suite string) string { return filepath.Join(turnTaxDir(), suite+".json") }

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
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
	_, _ = w.Write(b)
}

// handleSuites lists the available trace fixtures and which are present on disk.
func handleSuites(w http.ResponseWriter, r *http.Request) {
	type s struct {
		ID      string `json:"id"`
		Label   string `json:"label"`
		Present bool   `json:"present"`
		Calls   int    `json:"calls"`
	}
	out := make([]s, 0, len(knownSuites))
	for _, ks := range knownSuites {
		row := s{ID: ks.ID, Label: ks.Label}
		if t, err := turnbench.LoadTrace(suitePath(ks.ID)); err == nil {
			row.Present, row.Calls = true, len(t.Calls)
		}
		out = append(out, row)
	}
	writeJSON(w, map[string]any{"suites": out, "gomaxprocs": gomax, "dir": turnTaxDir()})
}

// handleRun replays one suite through the real kernel and returns the ordered
// per-call dispositions (the lane stream) plus the rolled-up turn-tax report.
func handleRun(w http.ResponseWriter, r *http.Request) {
	suite := r.URL.Query().Get("suite")
	if suite == "" {
		suite = "turntax-airline"
	}
	t, err := turnbench.LoadTrace(suitePath(suite))
	if err != nil {
		http.Error(w, "load trace: "+err.Error(), 400)
		return
	}

	cm := turnbench.DefaultCostModel()
	// Optional cost knobs so a viewer can reprice the same fixed turns.
	if v := r.URL.Query().Get("latency_ms"); v != "" {
		if f, e := strconv.ParseFloat(v, 64); e == nil {
			cm.ModelTurnLatencyMs = f
		}
	}
	if v := r.URL.Query().Get("prompt_tokens"); v != "" {
		if n, e := strconv.Atoi(v); e == nil {
			cm.PromptTokensPerTurn = n
		}
	}

	rep, calls, err := turnbench.RunWithCalls(r.Context(), t, cm)
	if err != nil {
		http.Error(w, "replay: "+err.Error(), 500)
		return
	}
	writeJSON(w, map[string]any{
		"suite":  suite,
		"calls":  calls,
		"report": rep,
	})
}

// budgetCores maps a fraction-or-percent budget + machine width to a core count in
// [1,cores] with the same grammar and half-up rounding as the model package's
// budgetToWorkers (kept local so this weights-free demo needn't import the model engine).
func budgetCores(raw float64, cores int) (int, bool) {
	if cores < 1 {
		cores = 1
	}
	frac := raw
	if frac > 1 {
		frac = frac / 100.0
	}
	if frac <= 0 || frac > 1 {
		return 0, false
	}
	n := int(float64(cores)*frac + 0.5) // half-up
	if n < 1 {
		n = 1
	}
	if n > cores {
		n = cores
	}
	return n, true
}

func main() {
	addr := flag.String("addr", "127.0.0.1:8150", "listen address")
	jobs := flag.Int("jobs", 0, "cap GOMAXPROCS to an ABSOLUTE core count (0 = all cores). On a shared/active box pass e.g. 8 so the demo doesn't starve other work.")
	budget := flag.Float64("budget", 0, "cap GOMAXPROCS to a FRACTION of the machine: 0.75 = 75% of the logical cores (portable; 75 or 0.75 accepted). Mutually exclusive with -jobs. 0 = unset.")
	flag.Parse()
	if *jobs > 0 && *budget > 0 {
		fmt.Fprintln(os.Stderr, "-jobs and -budget are mutually exclusive (one is absolute, the other a fraction)")
		os.Exit(2)
	}
	if *jobs > 0 {
		runtime.GOMAXPROCS(*jobs)
		gomax = *jobs
	} else if *budget > 0 {
		// This demo replays through the turn-tax kernel (no model matmul), so a budget
		// only bounds the Go scheduler width. Resolve the same fraction/percent grammar
		// the bench budget uses: <=1 is a fraction, >1 is a percent.
		n, ok := budgetCores(*budget, runtime.NumCPU())
		if !ok {
			fmt.Fprintf(os.Stderr, "budget %g is not a fraction in (0,1] or a percent in (0,100]\n", *budget)
			os.Exit(2)
		}
		runtime.GOMAXPROCS(n)
		gomax = n
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", handleIndex)
	mux.HandleFunc("/api/suites", handleSuites)
	mux.HandleFunc("/api/run", handleRun)

	present := []string{}
	for _, ks := range knownSuites {
		if _, err := os.Stat(suitePath(ks.ID)); err == nil {
			present = append(present, ks.ID)
		}
	}
	fmt.Fprintf(os.Stderr, "turntaxdemo %s on http://%s (GOMAXPROCS=%d)\n", version, listenAddr(*addr), gomax)
	fmt.Fprintf(os.Stderr, "trace dir: %s\n", turnTaxDir())
	if len(present) == 0 {
		fmt.Fprintf(os.Stderr, "WARNING: no turntax fixtures found — run from the fak/ directory\n")
	} else {
		fmt.Fprintf(os.Stderr, "suites present: %v\n", present)
	}
	fmt.Fprintf(os.Stderr, "open the URL → pick a suite → 'Replay through the kernel'\n")
	if err := http.ListenAndServe(listenAddr(*addr), mux); err != nil {
		fmt.Fprintln(os.Stderr, "listen:", err)
		os.Exit(1)
	}
}

// listenAddr honors the $PORT contract used by container platforms (Cloud Run,
// Heroku, etc.): when PORT is set in the environment, bind 0.0.0.0:$PORT and ignore
// the -addr loopback default, so the same binary serves locally (-addr) and in a
// container (no flags, just $PORT) with no rebuild. An explicit -addr that is not the
// compiled-in loopback default still wins, so a local override is never silently lost.
func listenAddr(addr string) string {
	if p := os.Getenv("PORT"); p != "" && addr == "127.0.0.1:8150" {
		return "0.0.0.0:" + p
	}
	return addr
}

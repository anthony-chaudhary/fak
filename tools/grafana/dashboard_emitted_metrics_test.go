package grafana

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// dashboard_emitted_metrics_test.go — the "no dead sensor on the board" contract.
//
// The per-board allowlist in fak_engine_batching_test.go is hand-maintained, so a
// metric can sit on it (and on the board) long after the renderer stopped emitting
// it, or before any renderer ever did. This test closes that gap from the other
// side: every fak_* family a fak-engine-batching.json PromQL expr queries must be
// produced by the Go code that renders `fak serve` /metrics. It parses the
// renderer packages and collects every fak_* metric name that appears as a Go
// string constant — a literal, or a constant-folded concatenation such as
// `const p = "fak_sched_preempt_"; p + "swap_total"` — so a family that no
// production code can spell fails here instead of rendering "Unavailable" forever.

// emittedMetricSourceDirs are the packages whose non-test Go files render the
// families the batching board reads (repo-relative).
var emittedMetricSourceDirs = []string{
	"internal/gateway",     // metrics_render.go and its writers (fak_gateway_*, fak_sched_*, fak_serving_*, fak_otlp_*)
	"internal/enginestep",  // fak_engine_* continuous-batching cycle
	"internal/stepobs",     // fak_engine_kernel_*, fak_engine_planner_step_*
	"internal/engine",      // fak_engine_cache_* (DefaultCacheEvents)
	"internal/modelengine", // native scheduler fak_sched_preempt_*
	"internal/cachemeta",   // fak_cache_* stream fold
}

// emittedMetricAllowlist names families the board may query that are not spelled
// as Go constants (none today). Keep it empty unless a renderer builds a name at
// runtime from non-constant parts; each entry needs a comment naming its emitter.
var emittedMetricAllowlist = map[string]string{}

func repoRoot(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
}

// collectEmittedMetricNames returns every fak_* token found in a string constant
// of the non-test Go files under dirs.
func collectEmittedMetricNames(t *testing.T, root string, dirs []string) map[string]bool {
	t.Helper()
	names := map[string]bool{}
	fset := token.NewFileSet()
	for _, dir := range dirs {
		abs := filepath.Join(root, filepath.FromSlash(dir))
		entries, err := os.ReadDir(abs)
		if err != nil {
			t.Fatalf("read %s: %v", dir, err)
		}
		files := 0
		for _, e := range entries {
			n := e.Name()
			if e.IsDir() || !strings.HasSuffix(n, ".go") || strings.HasSuffix(n, "_test.go") {
				continue
			}
			f, err := parser.ParseFile(fset, filepath.Join(abs, n), nil, parser.SkipObjectResolution)
			if err != nil {
				t.Fatalf("parse %s/%s: %v", dir, n, err)
			}
			files++
			for _, s := range fileStringConstants(f) {
				for _, m := range fakMetricNameRE.FindAllString(s, -1) {
					names[m] = true
				}
			}
		}
		if files == 0 {
			t.Fatalf("%s has no non-test Go files; the source list is stale", dir)
		}
	}
	return names
}

// fileStringConstants returns every string literal in f plus every constant-folded
// `+` concatenation whose operands are literals or string constants declared in
// the same file (at any scope).
func fileStringConstants(f *ast.File) []string {
	consts := map[string]string{}
	var lits []string
	// Two passes so a const used before its declaration (package scope) still folds.
	for pass := 0; pass < 2; pass++ {
		ast.Inspect(f, func(n ast.Node) bool {
			gd, ok := n.(*ast.GenDecl)
			if !ok || gd.Tok != token.CONST {
				return true
			}
			for _, spec := range gd.Specs {
				vs, ok := spec.(*ast.ValueSpec)
				if !ok {
					continue
				}
				for i, name := range vs.Names {
					if i < len(vs.Values) {
						if v, ok := foldString(vs.Values[i], consts); ok {
							consts[name.Name] = v
						}
					}
				}
			}
			return true
		})
	}
	ast.Inspect(f, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.BasicLit:
			if v, ok := foldString(x, consts); ok {
				lits = append(lits, v)
			}
		case *ast.BinaryExpr:
			if v, ok := foldString(x, consts); ok {
				lits = append(lits, v)
			}
		}
		return true
	})
	return lits
}

func foldString(e ast.Expr, consts map[string]string) (string, bool) {
	switch x := e.(type) {
	case *ast.BasicLit:
		if x.Kind != token.STRING {
			return "", false
		}
		v, err := strconv.Unquote(x.Value)
		return v, err == nil
	case *ast.Ident:
		v, ok := consts[x.Name]
		return v, ok
	case *ast.ParenExpr:
		return foldString(x.X, consts)
	case *ast.BinaryExpr:
		if x.Op != token.ADD {
			return "", false
		}
		l, ok := foldString(x.X, consts)
		if !ok {
			return "", false
		}
		r, ok := foldString(x.Y, consts)
		if !ok {
			return "", false
		}
		return l + r, true
	}
	return "", false
}

func metricEmitted(emitted map[string]bool, name string) bool {
	if emitted[name] {
		return true
	}
	if _, ok := emittedMetricAllowlist[name]; ok {
		return true
	}
	for _, sfx := range histogramSuffixes {
		if base, ok := strings.CutSuffix(name, sfx); ok && (emitted[base] || emitted[base+"_bucket"]) {
			return true
		}
	}
	return false
}

func TestFakEngineBatchingQueriesOnlyEmittedMetrics(t *testing.T) {
	emitted := collectEmittedMetricNames(t, repoRoot(t), emittedMetricSourceDirs)
	d := loadFakEngineBatching(t)
	queried := map[string][]string{}
	for _, p := range flattenFakEngineBatchingPanels(d.Panels) {
		for _, tgt := range p.Targets {
			for _, name := range fakMetricNameRE.FindAllString(quotedRE.ReplaceAllString(tgt.Expr, `""`), -1) {
				queried[name] = append(queried[name], strconv.Itoa(p.ID)+" "+strconv.Quote(p.Title))
			}
		}
	}
	if len(queried) == 0 {
		t.Fatal("dashboard queries no fak_* metric; the expr scan is broken")
	}
	var dead []string
	for name, panels := range queried {
		if !metricEmitted(emitted, name) {
			dead = append(dead, name+" (panels: "+strings.Join(panels, ", ")+")")
		}
	}
	sort.Strings(dead)
	for _, d := range dead {
		t.Errorf("dashboard queries %s, which no renderer in %v emits; wire the producer or drop the panel", d, emittedMetricSourceDirs)
	}
}

// The scan must actually see real renderer output, including constant-folded
// prefix names; otherwise the contract above could pass vacuously.
func TestEmittedMetricScanSeesKnownFamilies(t *testing.T) {
	emitted := collectEmittedMetricNames(t, repoRoot(t), emittedMetricSourceDirs)
	for _, want := range []string{
		"fak_engine_cache_restore_miss_total", // literal (engine)
		"fak_engine_kernel_seconds",           // const (stepobs)
		"fak_gateway_inference_ttft_seconds",  // literal (gateway)
		"fak_sched_preempt_swap_total",        // const prefix + literal (modelengine)
		"fak_otlp_spans_total",                // inside a format string (gateway)
	} {
		if !emitted[want] {
			t.Errorf("emitted-metric scan missed %s", want)
		}
	}
	if emitted["fak_definitely_not_a_metric_total"] {
		t.Error("scan reports a name nothing emits")
	}
}

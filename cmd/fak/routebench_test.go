package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// runRB drives the testable runRoutebench core with captured streams.
func runRB(args ...string) (int, string, string) {
	var out, errb bytes.Buffer
	code := runRoutebench(&out, &errb, args)
	return code, out.String(), errb.String()
}

// The default run uses the built-in demo corpus + manifests: it prints the
// three-axis table and an honest verdict (cheaper/faster on compute, quality tied).
func TestRoutebenchDemoDefault(t *testing.T) {
	code, out, _ := runRB()
	if code != 0 {
		t.Fatalf("exit=%d", code)
	}
	for _, want := range []string{
		"offline routing benchmark",
		"routed (per-aspect + ensemble)",
		"single (one model)",
		"cost (rough $/Mtok-out, sum)",
		"latency (rough ms, sum)",
		"quality (fraction == expected)",
		"verdict:",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("output missing %q:\n%s", want, out)
		}
	}
	// The demo is cheaper + faster on compute; quality ties (rescue offsets downgrade).
	if !strings.Contains(out, "cheaper") || !strings.Contains(out, "quality tied") {
		t.Fatalf("demo verdict wrong (want cheaper + quality tied):\n%s", out)
	}
}

// JSON mode emits a stable comparison a consumer reads directly.
func TestRoutebenchJSON(t *testing.T) {
	code, out, _ := runRB("--json")
	if code != 0 {
		t.Fatalf("exit=%d", code)
	}
	var rep struct {
		Frontier string  `json:"frontier"`
		Cases    int     `json:"cases"`
		Cost     float64 `json:"cost_saving_frac"`
		Latency  float64 `json:"latency_saving_frac"`
		Quality  float64 `json:"quality_delta"`
		Verdict  string  `json:"verdict"`
		Routed   struct {
			Ensembles int     `json:"ensembles"`
			Quality   float64 `json:"quality"`
		} `json:"routed"`
		Single struct {
			Ensembles int `json:"ensembles"`
		} `json:"single"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, out)
	}
	if rep.Cases != 8 || rep.Routed.Ensembles != 2 || rep.Single.Ensembles != 0 {
		t.Fatalf("cases/ensembles wrong: %+v", rep)
	}
	if !(rep.Cost > 0.19 && rep.Cost < 0.20) {
		t.Fatalf("cost_saving_frac = %v, want ~0.198", rep.Cost)
	}
	if !(rep.Latency > 0.10 && rep.Latency < 0.11) {
		t.Fatalf("latency_saving_frac = %v, want ~0.104", rep.Latency)
	}
	if rep.Quality != 0 {
		t.Fatalf("quality_delta = %v, want 0", rep.Quality)
	}
}

// --dump-corpus emits a corpus that parses back (so an operator can edit it).
func TestRoutebenchDumpCorpus(t *testing.T) {
	code, out, _ := runRB("--dump-corpus")
	if code != 0 {
		t.Fatalf("exit=%d", code)
	}
	// The dumped demo corpus parses cleanly (it is the canonical form).
	if _, err := modelroute.ParseCorpus([]byte(out)); err != nil {
		t.Fatalf("dumped corpus does not parse: %v", err)
	}
}

// A custom corpus on disk drives the benchmark the same way as the demo.
func TestRoutebenchCustomCorpus(t *testing.T) {
	dir := t.TempDir()
	corpus := []byte(`[
		{"subject":{"aspect":"tool_call","tool":"write_file"},"outputs":{"frontier":"deny","guard-a":"approve","guard-b":"approve"},"expected":"approve"}
	]`)
	path := filepath.Join(dir, "corpus.json")
	if err := os.WriteFile(path, corpus, 0644); err != nil {
		t.Fatal(err)
	}
	code, out, _ := runRB("--corpus", path, "--json")
	if code != 0 {
		t.Fatalf("exit=%d", code)
	}
	var rep struct {
		Routed struct {
			Quality float64 `json:"quality"`
		} `json:"routed"`
		Single struct {
			Quality float64 `json:"quality"`
		} `json:"single"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, out)
	}
	// One-case ensemble rescue: routed correct, single wrong.
	if rep.Routed.Quality != 1.0 || rep.Single.Quality != 0.0 {
		t.Fatalf("custom corpus quality routed=%v single=%v, want 1.0 / 0.0", rep.Routed.Quality, rep.Single.Quality)
	}
}

// A missing corpus file is exit 1, not a silent fallback to the demo.
func TestRoutebenchMissingCorpus(t *testing.T) {
	code, _, errb := runRB("--corpus", "does-not-exist-xyz.json")
	if code != 1 || errb == "" {
		t.Fatalf("missing corpus: exit=%d err=%q", code, errb)
	}
}

// An unknown flag is a usage error (exit 2), matching `fak route`.
func TestRoutebenchUnknownFlag(t *testing.T) {
	code, _, _ := runRB("--no-such-flag")
	if code != 2 {
		t.Fatalf("unknown flag: exit=%d, want 2", code)
	}
}

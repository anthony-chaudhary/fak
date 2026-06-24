package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// runRT drives the testable runRoute core with captured streams.
func runRT(args ...string) (int, string, string) {
	var out, errb bytes.Buffer
	code := runRoute(&out, &errb, args)
	return code, out.String(), errb.String()
}

// The built-in manifest routes a write-shaped tool call to the guard ensemble.
func TestRouteBuiltinPerAspect(t *testing.T) {
	code, out, _ := runRT("--aspect", "tool_call", "--tool", "write_file")
	if code != 0 {
		t.Fatalf("exit=%d", code)
	}
	if !strings.Contains(out, "guard-writes") || !strings.Contains(out, "ENSEMBLE") {
		t.Fatalf("want guard ensemble, got:\n%s", out)
	}
}

// A high-complexity step routes to a single large-model PICK.
func TestRoutePickJSON(t *testing.T) {
	code, out, _ := runRT("--aspect", "step", "--complexity", "high", "--json")
	if code != 0 {
		t.Fatalf("exit=%d", code)
	}
	var rep struct {
		Matched  bool   `json:"matched"`
		Rule     string `json:"rule"`
		Ensemble bool   `json:"ensemble"`
		Primary  string `json:"primary"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, out)
	}
	if !rep.Matched || rep.Rule != "hard-reasoning" || rep.Ensemble || rep.Primary != "large" {
		t.Fatalf("unexpected decision: %+v", rep)
	}
}

// --simulate folds stand-in member outputs through the plan's reduction; a vote
// ensemble picks the majority output, deterministically.
func TestRouteSimulateVote(t *testing.T) {
	code, out, _ := runRT("--aspect", "tool_call", "--tool", "write_x", "--simulate", "approve,deny,approve", "--json")
	if code != 0 {
		t.Fatalf("exit=%d", code)
	}
	var rep struct {
		Ensemble  bool `json:"ensemble"`
		Reduction struct {
			Reduce string `json:"reduce"`
			Output string `json:"output"`
		} `json:"reduction"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, out)
	}
	if !rep.Ensemble || rep.Reduction.Reduce != "vote" || rep.Reduction.Output != "approve" {
		t.Fatalf("vote reduce wrong: %+v", rep)
	}
}

// --dump emits a manifest that --check (round-trip) accepts.
func TestRouteDumpIsValid(t *testing.T) {
	code, out, _ := runRT("--dump")
	if code != 0 || !strings.Contains(out, "fak-route/v1") {
		t.Fatalf("dump exit=%d out=%s", code, out)
	}
}

// An unparseable manifest path is an error (exit 1), not a silent fallback.
func TestRouteCheckMissingFile(t *testing.T) {
	code, _, errb := runRT("--check", "does-not-exist-xyz.json")
	if code != 1 || errb == "" {
		t.Fatalf("missing-file check: exit=%d err=%q", code, errb)
	}
}

// An unknown flag is a usage error (exit 2).
func TestRouteUnknownFlag(t *testing.T) {
	code, _, _ := runRT("--no-such-flag")
	if code != 2 {
		t.Fatalf("unknown flag exit=%d (want 2)", code)
	}
}

// A cheap interactive route prints a rough "usage saved vs the frontier" line.
func TestRouteUsageSavingHuman(t *testing.T) {
	code, out, _ := runRT("--latency", "interactive", "--prompt-tokens", "100")
	if code != 0 {
		t.Fatalf("exit=%d", code)
	}
	if !strings.Contains(out, "usage (rough") || !strings.Contains(out, "cheaper than always-frontier") {
		t.Fatalf("want a rough cheaper-than-frontier usage line, got:\n%s", out)
	}
}

// The JSON decision carries a machine-readable usage object; routing the hard step
// to the large (frontier) tier is the baseline (0% saving), not a fabricated win.
func TestRouteUsageJSONBaseline(t *testing.T) {
	code, out, _ := runRT("--complexity", "high", "--json")
	if code != 0 {
		t.Fatalf("exit=%d", code)
	}
	var rep struct {
		Usage struct {
			Frontier     string  `json:"frontier"`
			SavedOutFrac float64 `json:"saved_out_frac"`
			Estimable    bool    `json:"estimable"`
		} `json:"usage"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, out)
	}
	if !rep.Usage.Estimable || rep.Usage.SavedOutFrac != 0 || rep.Usage.Frontier != "frontier" {
		t.Fatalf("hard-reasoning -> large should be the frontier baseline (0), got %+v", rep.Usage)
	}
}

// A write-shaped tool call routes to a two-model guard ensemble — reported as a
// PREMIUM (it runs more compute than one frontier call), never as a saving.
func TestRouteUsageEnsembleIsPremium(t *testing.T) {
	code, out, _ := runRT("--aspect", "tool_call", "--tool", "write_file")
	if code != 0 {
		t.Fatalf("exit=%d", code)
	}
	if !strings.Contains(out, "vs one frontier call") || !strings.Contains(out, "deliberate") || strings.Contains(out, "cheaper") {
		t.Fatalf("guard ensemble must read as a premium, not cheaper:\n%s", out)
	}
}

// --prices overrides the rough book: pricing "small" at the frontier rate collapses
// the saving to the baseline — proving the number is a function of stated inputs.
func TestRouteUsagePricesOverride(t *testing.T) {
	code, out, _ := runRT("--latency", "interactive", "--prompt-tokens", "100", "--prices", "small=3/15", "--json")
	if code != 0 {
		t.Fatalf("exit=%d", code)
	}
	var rep struct {
		Usage struct {
			SavedOutFrac float64 `json:"saved_out_frac"`
		} `json:"usage"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, out)
	}
	if rep.Usage.SavedOutFrac != 0 {
		t.Fatalf("small priced at frontier should yield 0 saving, got %v", rep.Usage.SavedOutFrac)
	}
}

// --check prints the per-rule cost lens (cheaper / premium / baseline) and a footer.
func TestRouteCheckShowsCostLens(t *testing.T) {
	_, dump, _ := runRT("--dump")
	path := filepath.Join(t.TempDir(), "m.json")
	if err := os.WriteFile(path, []byte(dump), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	code, out, _ := runRT("--check", path)
	if code != 0 {
		t.Fatalf("check exit=%d\n%s", code, out)
	}
	if !strings.Contains(out, "cost lens") || !strings.Contains(out, "save ~") || !strings.Contains(out, "premium +") {
		t.Fatalf("want a per-rule cost lens with save/premium tags, got:\n%s", out)
	}
}

// An unmatched aspect falls through to the fail-closed default.
func TestRouteDefault(t *testing.T) {
	code, out, _ := runRT("--aspect", "nonsense_aspect")
	if code != 0 {
		t.Fatalf("exit=%d", code)
	}
	if !strings.Contains(out, "fail-closed default") || !strings.Contains(out, "PICK -> default") {
		t.Fatalf("want default, got:\n%s", out)
	}
}

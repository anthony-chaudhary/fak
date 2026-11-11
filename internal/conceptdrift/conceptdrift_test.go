package conceptdrift

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// NOTE: the comments must not spell the family roots (cache / gate / guard) as
// standalone words - the identifier sweep would discover them as extra tokens.
const goSrc = `package demo

// DemoCachePolicy selects the demo policy set.
func DemoCachePolicy() string { return "lru" }

// DemoCachePolicyHint refines the demo policy choice.
func DemoCachePolicyHint() string { return "warm" }

// DemoCacheGate secures the demo entry point.
func DemoCacheGate() bool { return true }
`

// buildWorkspace builds the TINY synthetic workspace the Python test used: three
// .go files in internal/demo mentioning DemoCachePolicy / DemoCachePolicyHint /
// DemoCacheGate across all three files, plus a minimal data directory (two
// families, two rows files). The three identifiers each span all three files
// (presence 3 >= min_files 2), so the scorecard discovers exactly three
// confusable tokens: democachepolicy, democachepolicyhint (family cache) and
// democachegate (families cache AND guard-gate; deduped once in the headline
// universe). Two rows files position all three, so the clean state is
// coverage-debt 0.
func buildWorkspace(t *testing.T, root string) (ws, data string) {
	t.Helper()
	ws = filepath.Join(root, "ws")
	demo := filepath.Join(ws, "internal", "demo")
	if err := os.MkdirAll(demo, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"a.go", "b.go", "c.go"} {
		if err := os.WriteFile(filepath.Join(demo, name), []byte(goSrc), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	data = filepath.Join(root, "data")
	if err := os.MkdirAll(data, 0o755); err != nil {
		t.Fatal(err)
	}
	meta := `{"meta": {"as_of": "2026-11-08", "fak_version": "t"}, "families": [{"id": "cache", "roots": ["cache"], "min_files": 2}, {"id": "guard-gate", "roots": ["gate", "guard"], "min_files": 2}]}`
	writeJSONFileT(t, filepath.Join(data, "_meta.json"), meta)
	rowsCache := `{
  "rows": [
    {"id": "dc1", "canonical": "Demo Cache Policy", "family": "cache", "kind": "concept",
     "definition": "the demo cache policy selector.", "distinction": "not the hint variant",
     "distinct_from": ["dc2"], "aliases": [], "grounding": "demoCachePolicy",
     "grounding_kind": "symbol", "glossary_anchor": "", "verdict": "defined", "gaps": []},
    {"id": "dc2", "canonical": "Demo Cache Policy Hint", "family": "cache", "kind": "concept",
     "definition": "the demo cache policy hint.", "distinction": "not the base policy",
     "distinct_from": ["dc1"], "aliases": [], "grounding": "demoCachePolicyHint",
     "grounding_kind": "symbol", "glossary_anchor": "", "verdict": "defined", "gaps": []}
  ]
}`
	rowsGate := `{
  "rows": [
    {"id": "dg1", "canonical": "Demo Cache Gate", "family": "guard-gate", "kind": "concept",
     "definition": "the demo cache gate guard.", "distinction": "guards the demo cache entry point",
     "distinct_from": [], "aliases": [], "grounding": "demoCacheGate",
     "grounding_kind": "symbol", "glossary_anchor": "", "verdict": "defined", "gaps": []}
  ]
}`
	writeJSONFileT(t, filepath.Join(data, "rows-cache.json"), rowsCache)
	writeJSONFileT(t, filepath.Join(data, "rows-gate.json"), rowsGate)
	return ws, data
}

func writeJSONFileT(t *testing.T, path, body string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// repoRootOfTest walks up from the package dir to the fak repo root (the tree
// carrying tools/concept_disambiguation_scorecard.py).
func repoRootOfTest(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if fi, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil && fi.Mode().IsRegular() {
			if _, err := os.Stat(filepath.Join(dir, "tools", ScorecardName)); err == nil {
				return dir
			}
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("fak repo root not found above " + dir)
		}
		dir = parent
	}
}

// runCLI drives Run the way the cmd/fak shell does for a fully-flagged
// invocation against the synthetic workspace.
func runCLI(t *testing.T, ws, data, baseline string, recordBaseline, simulate, autoTicket bool) (int, string, string) {
	t.Helper()
	var out, errw bytes.Buffer
	rc := Run(&out, &errw, Options{
		Workspace:      ws,
		DataDir:        data,
		Baseline:       baseline,
		Scorecard:      filepath.Join(repoRootOfTest(t), "tools", ScorecardName),
		RecordBaseline: recordBaseline,
		Simulate:       simulate,
		AutoTicket:     autoTicket,
	})
	return rc, out.String(), errw.String()
}

// --- pure helpers ----------------------------------------------------------

func TestDecodeJSONBytesHandlesUTF8BOMAndUTF16(t *testing.T) {
	obj, _ := json.Marshal(map[string]any{"a": 1})
	cases := map[string][]byte{
		"plain utf-8": obj,
		"utf-8 BOM":   append([]byte{0xef, 0xbb, 0xbf}, obj...),
	}
	// Python's utf-16 codec emits a BOM-prefixed payload, like PowerShell 5 '>' does.
	u16, err := json.Marshal(map[string]any{"a": 1})
	if err != nil {
		t.Fatal(err)
	}
	var le []byte
	le = append(le, 0xff, 0xfe)
	for _, c := range u16 { // ASCII -> UTF-16LE: low byte first, zero high byte
		le = append(le, c, 0)
	}
	cases["utf-16 LE with BOM"] = le
	for name, raw := range cases {
		got, err := DecodeJSONBytes(raw)
		if err != nil {
			t.Errorf("%s: DecodeJSONBytes: %v", name, err)
			continue
		}
		m, ok := got.(map[string]any)
		if !ok || m["a"] != float64(1) {
			t.Errorf("%s: got %#v, want {a:1}", name, got)
		}
	}
}

func TestRowTokensCountsCanonicalGroundingAliases(t *testing.T) {
	r := map[string]any{"canonical": "KV cache", "grounding": "kv_cache", "aliases": []any{"vCache", "kvCache"}}
	toks := RowTokens(r)
	if len(toks) != 2 || !toks["kvcache"] || !toks["vcache"] {
		t.Errorf("RowTokens = %v, want {kvcache vcache}", toks)
	}
	empty := RowTokens(map[string]any{"canonical": "", "grounding": "", "aliases": []any{nil}})
	if len(empty) != 0 {
		t.Errorf("RowTokens(empty/nil aliases) = %v, want {}", empty)
	}
}

func TestPickVictimPrefersMostTokensThenName(t *testing.T) {
	td := t.TempDir()
	d := filepath.Join(td, "data")
	if err := os.MkdirAll(d, 0o755); err != nil {
		t.Fatal(err)
	}
	writeJSONFileT(t, filepath.Join(d, "_meta.json"), `{"families": []}`)
	writeJSONFileT(t, filepath.Join(d, "rows-a.json"), `{"rows": [{"canonical": "Alpha", "grounding": "alpha", "aliases": []}, {"canonical": "Beta", "grounding": "beta", "aliases": []}]}`)
	writeJSONFileT(t, filepath.Join(d, "rows-b.json"), `{"rows": [{"canonical": "Gamma", "grounding": "gamma", "aliases": []}]}`)

	victim, toks, err := PickVictim(d)
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(victim) != "rows-a.json" || len(toks) != 2 || !toks["alpha"] || !toks["beta"] {
		t.Fatalf("victim = %s %v, want rows-a.json {alpha beta}", victim, toks)
	}
	// Tie on token count breaks deterministically by file name.
	if err := os.Remove(filepath.Join(d, "rows-a.json")); err != nil {
		t.Fatal(err)
	}
	writeJSONFileT(t, filepath.Join(d, "rows-c.json"), `{"rows": [{"canonical": "Delta", "grounding": "delta", "aliases": []}]}`)
	victim, toks, err = PickVictim(d)
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(victim) != "rows-b.json" || len(toks) != 1 || !toks["gamma"] {
		t.Fatalf("victim = %s %v, want rows-b.json {gamma}", victim, toks)
	}
}

func TestUncoveredByFamilyGroupsAndSorts(t *testing.T) {
	ubf := UncoveredByFamily([]UncoveredToken{
		{Family: "cache", Token: "zeta"},
		{Family: "", Token: "orphan"},
		{Family: "cache", Token: "alpha"},
		{Family: "cache", Token: ""},
	})
	if len(ubf) != 2 {
		t.Fatalf("families = %v, want cache + ?", ubf)
	}
	if strings.Join(ubf["cache"], ",") != "alpha,zeta" {
		t.Errorf("cache tokens = %v, want [alpha zeta]", ubf["cache"])
	}
	if strings.Join(ubf["?"], ",") != "orphan" {
		t.Errorf("? tokens = %v, want [orphan]", ubf["?"])
	}
}

// --- end to end through the CLI --------------------------------------------

func TestRecordBaselineWritesTheFile(t *testing.T) {
	root := t.TempDir()
	ws, data := buildWorkspace(t, root)
	baseline := filepath.Join(root, "baseline.json")
	rc, out, errw := runCLI(t, ws, data, baseline, true, false, false)
	if rc != 0 {
		t.Fatalf("record-baseline rc = %d, stderr: %s stdout: %s", rc, errw, out)
	}
	raw, err := os.ReadFile(baseline)
	if err != nil {
		t.Fatalf("baseline not written: %v", err)
	}
	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("baseline is not JSON: %v\n%s", err, raw)
	}
	if doc["schema"] != BaselineSchema {
		t.Errorf("schema = %v, want %s", doc["schema"], BaselineSchema)
	}
	if toInt(doc["discovered"]) != 3 || toInt(doc["covered"]) != 3 || toInt(doc["coverage_debt"]) != 0 {
		t.Errorf("counters = disc %v cov %v debt %v, want 3/3/0", doc["discovered"], doc["covered"], doc["coverage_debt"])
	}
	if doc["recorded_at"] == nil || doc["recorded_at"].(string) == "" {
		t.Error("recorded_at missing or empty")
	}
	standing, _ := doc["standing"].(map[string]any)
	if toInt(standing["defined"]) != 3 {
		t.Errorf("standing.defined = %v, want 3 (standing=%v)", standing["defined"], standing)
	}
}

func TestCheckModeCleanExit0WithReceipt(t *testing.T) {
	root := t.TempDir()
	ws, data := buildWorkspace(t, root)
	baseline := filepath.Join(root, "baseline.json")
	if rc, out, errw := runCLI(t, ws, data, baseline, true, false, false); rc != 0 {
		t.Fatalf("record-baseline rc = %d: %s %s", rc, errw, out)
	}
	rc, out, errw := runCLI(t, ws, data, baseline, false, false, false)
	if rc != 0 {
		t.Fatalf("check rc = %d, stderr: %s stdout: %s", rc, errw, out)
	}
	receipts, err := filepath.Glob(filepath.Join(root, ReceiptPrefix+"*.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(receipts) != 1 {
		t.Fatalf("receipts = %d, want 1 (%v)", len(receipts), receipts)
	}
	raw, err := os.ReadFile(receipts[0])
	if err != nil {
		t.Fatal(err)
	}
	var rec map[string]any
	if err := json.Unmarshal(raw, &rec); err != nil {
		t.Fatalf("receipt is not JSON: %v", err)
	}
	if rec["schema"] != ReceiptSchema || rec["kind"] != "check" {
		t.Errorf("schema/kind = %v/%v, want %s/check", rec["schema"], rec["kind"], ReceiptSchema)
	}
	if rec["verdict"] != "OK" || rec["drift"] != false {
		t.Errorf("verdict/drift = %v/%v, want OK/false", rec["verdict"], rec["drift"])
	}
	delta, _ := rec["delta"].(map[string]any)
	if toInt(delta["coverage_debt"]) != 0 || toInt(delta["discovered"]) != 0 {
		t.Errorf("delta = %v, want zeros", delta)
	}
	ubf, _ := rec["uncovered_by_family"].(map[string]any)
	if len(ubf) != 0 {
		t.Errorf("uncovered_by_family = %v, want {}", ubf)
	}
}

func TestCheckModeRemovedRowsFileExit3(t *testing.T) {
	root := t.TempDir()
	ws, data := buildWorkspace(t, root)
	baseline := filepath.Join(root, "baseline.json")
	if rc, out, errw := runCLI(t, ws, data, baseline, true, false, false); rc != 0 {
		t.Fatalf("record-baseline rc = %d: %s %s", rc, errw, out)
	}
	// Simulate the merge losing the biggest rows file: copy the data dir, remove it.
	shrunk := filepath.Join(root, "shrunk-data")
	if err := copyTree(data, shrunk); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(shrunk, "rows-cache.json")); err != nil { // covers 2 tokens - the most
		t.Fatal(err)
	}
	rc, out, errw := runCLI(t, ws, shrunk, baseline, false, false, false)
	if rc != ExitDrift {
		t.Fatalf("check rc = %d, want %d (drift); stderr: %s stdout: %s", rc, ExitDrift, errw, out)
	}
	receipts, _ := filepath.Glob(filepath.Join(root, ReceiptPrefix+"*.json"))
	if len(receipts) != 1 {
		t.Fatalf("receipts = %d, want 1", len(receipts))
	}
	raw, err := os.ReadFile(receipts[0])
	if err != nil {
		t.Fatal(err)
	}
	var rec map[string]any
	if err := json.Unmarshal(raw, &rec); err != nil {
		t.Fatal(err)
	}
	if rec["verdict"] != "DRIFT" || rec["drift"] != true {
		t.Errorf("verdict/drift = %v/%v, want DRIFT/true", rec["verdict"], rec["drift"])
	}
	delta, _ := rec["delta"].(map[string]any)
	if toInt(delta["coverage_debt"]) != 2 || toInt(delta["discovered"]) != 0 {
		t.Errorf("delta = %v, want debt +2 discovered 0", delta)
	}
	ubf, _ := rec["uncovered_by_family"].(map[string]any)
	cache, _ := ubf["cache"].([]any)
	if len(cache) != 2 {
		t.Fatalf("uncovered cache = %v, want 2 tokens", cache)
	}
	set := map[string]bool{}
	for _, c := range cache {
		set[c.(string)] = true
	}
	if !set["democachepolicy"] || !set["democachepolicyhint"] {
		t.Errorf("uncovered cache tokens = %v, want {democachepolicy democachepolicyhint}", cache)
	}
	// The operator-facing line names the family and its tokens.
	if !strings.Contains(out, "cache: democachepolicy, democachepolicyhint") {
		t.Errorf("stdout lacks the per-family drift line:\n%s", out)
	}
}

func TestSimulateDetectsItsOwnDrift(t *testing.T) {
	root := t.TempDir()
	ws, data := buildWorkspace(t, root)
	rc, out, errw := runCLI(t, ws, data, "", false, true, false)
	if rc != 0 {
		t.Fatalf("simulate rc = %d, want 0; stderr: %s stdout: %s", rc, errw, out)
	}
	idx := strings.Index(out, "{")
	if idx < 0 {
		t.Fatalf("simulate printed no JSON receipt:\n%s", out)
	}
	var rec map[string]any
	if err := json.Unmarshal([]byte(out[idx:]), &rec); err != nil {
		t.Fatalf("simulate receipt is not JSON: %v\n%s", err, out)
	}
	if rec["kind"] != "simulate" || rec["drift_detected"] != true {
		t.Errorf("kind/drift_detected = %v/%v, want simulate/true", rec["kind"], rec["drift_detected"])
	}
	if rec["verdict"] != "DRIFT-DETECTED" {
		t.Errorf("verdict = %v, want DRIFT-DETECTED", rec["verdict"])
	}
	if rec["removed_file"] != "rows-cache.json" {
		t.Errorf("removed_file = %v, want rows-cache.json", rec["removed_file"])
	}
	removed, _ := rec["removed_tokens"].([]any)
	set := map[string]bool{}
	for _, r := range removed {
		set[r.(string)] = true
	}
	if len(set) != 2 || !set["democachepolicy"] || !set["democachepolicyhint"] {
		t.Errorf("removed_tokens = %v, want {democachepolicy democachepolicyhint}", removed)
	}
	// The self-witness is a DIFF: clean pre-state, debt re-opened post-removal,
	// and every newly-uncovered token attributable to the removed file.
	pre, _ := rec["pre"].(map[string]any)
	post, _ := rec["post"].(map[string]any)
	delta, _ := rec["delta"].(map[string]any)
	if toInt(pre["coverage_debt"]) != 0 || toInt(post["coverage_debt"]) != 2 || toInt(delta["coverage_debt"]) != 2 {
		t.Errorf("pre/post/delta = %v/%v/%v, want debt 0 -> 2 (+2)", pre, post, delta)
	}
	newly, _ := rec["newly_uncovered"].([]any)
	nset := map[string]bool{}
	for _, n := range newly {
		nset[n.(string)] = true
	}
	if len(nset) != 2 || !nset["democachepolicy"] || !nset["democachepolicyhint"] {
		t.Errorf("newly_uncovered = %v, want {democachepolicy democachepolicyhint}", newly)
	}
}

func TestCheckModeWithoutBaselineIsOperationalError(t *testing.T) {
	root := t.TempDir()
	ws, data := buildWorkspace(t, root)
	rc, out, errw := runCLI(t, ws, data, filepath.Join(root, "nope.json"), false, false, false)
	if rc != ExitOperational {
		t.Fatalf("check rc = %d, want %d; stdout: %s", rc, ExitOperational, out)
	}
	if !strings.Contains(errw, "no baseline at") {
		t.Errorf("stderr lacks the no-baseline reason: %s", errw)
	}
}

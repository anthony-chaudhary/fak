package toon

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// realFamilies builds the three REAL fak payload families the scorecard measures:
//
//  1. index_leaves    — a uniform array of flat objects modeling `fak index leaves` rows
//     (name/tree/dir/exists/desc); TOON's happy path, expect a WIN.
//  2. nightrun_jsonl  — rows read from an ACTUAL docs/nightrun/*.jsonl file in the tree;
//     real telemetry, the semi-uniform interesting middle.
//  3. nested_config   — a deliberately nested/mixed config object (a serve/guard tool
//     result); expect TOON to LOSE, and the scorecard must show the loss honestly.
//
// Nothing here is synthetic-only: families 1 and 3 use real fak field names and shapes;
// family 2 is parsed straight off a committed telemetry file.
func realFamilies(t *testing.T) []Family {
	t.Helper()
	return []Family{
		{Name: "index_leaves (uniform flat)", Payload: indexLeavesPayload()},
		{Name: "nightrun_jsonl (real telemetry)", Payload: nightrunPayload(t)},
		{Name: "nested_config (mixed object)", Payload: nestedConfigPayload()},
	}
}

// indexLeavesPayload is a uniform array of flat objects modeling `fak index leaves` output —
// the flat projection an index tool renders for tabular display (real fak leaf names, the
// devindex.Leaf scalar fields; the nested Status rollup is omitted, as a tabular row view is).
func indexLeavesPayload() any {
	rows := []struct {
		name, tree, dir, desc string
		exists                bool
	}{
		{"gateway", "internal/gateway/**", "internal/gateway", "the model wire and MCP tool-result wrap point", true},
		{"guard", "cmd/fak/guard.go", "cmd/fak", "the pre-flight guard gate and exit summary", true},
		{"toon", "internal/toon/**", "internal/toon", "the JSON to TOON codec, gate, and scorecard", true},
		{"memview", "internal/memview/**", "internal/memview", "memory recall surfaces and format sweep", true},
		{"cachewitness", "internal/cachewitness/**", "internal/cachewitness", "the observed prompt-cache savings ledger", true},
		{"dispatch", "internal/dispatch/**", "internal/dispatch", "the multi-agent work dispatch tick", true},
		{"devindex", "internal/devindex/**", "internal/devindex", "the self-index over lanes, claims, and docs", true},
		{"cachebp", "internal/agent/anthropic_cachebp.go", "internal/agent", "cache-breakpoint splicing and volatility refusal", true},
	}
	arr := make([]any, len(rows))
	for i, r := range rows {
		arr[i] = map[string]any{
			"name":   r.name,
			"tree":   r.tree,
			"dir":    r.dir,
			"exists": r.exists,
			"desc":   r.desc,
		}
	}
	return arr
}

// nightrunPayload reads real rows off a committed docs/nightrun/*.jsonl telemetry file and
// returns them as a []any of map[string]any — exactly the encoding/json-native shape a tool
// result carries on the wire. It caps the row count so the payload is a representative slab,
// not the whole file, and skips lines that fail to parse (a shared multi-session tree may be
// mid-write). If the file cannot be read the test skips this family rather than inventing data.
func nightrunPayload(t *testing.T) any {
	t.Helper()
	const maxRows = 12
	path := repoPath(t, "docs", "nightrun", "harness-resources.jsonl")
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("nightrun family: cannot open %s: %v", path, err)
	}
	defer f.Close()
	var rows []any
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<20)
	for sc.Scan() && len(rows) < maxRows {
		text := strings.TrimSpace(sc.Text())
		if text == "" {
			continue
		}
		var row any
		if err := json.Unmarshal([]byte(text), &row); err != nil {
			continue // tolerate a partially-written line in a shared tree
		}
		rows = append(rows, row)
	}
	if len(rows) == 0 {
		t.Skipf("nightrun family: no parseable rows in %s", path)
	}
	return rows
}

// nestedConfigPayload is a deliberately nested/mixed configuration object — the shape TOON is
// known to LOSE on (fixed structural overhead, no repeated field-name anchors to amortize).
// It models a real fak serve/guard tool-result config: nested policy sub-objects, a mixed
// array, and scalars at several depths. Expect a low TabularEligibility and a SKIP.
func nestedConfigPayload() any {
	return map[string]any{
		"session_type": "serve",
		"provider":     "anthropic",
		"cache": map[string]any{
			"mechanism":             "provider_prompt_cache",
			"breakpoints":           4.0,
			"volatile_head_refused": true,
			"ttl": map[string]any{
				"seconds": 300.0,
				"policy":  "sliding",
			},
		},
		"guard": map[string]any{
			"off_trunk_refuses": true,
			"reasons":           []any{"OFF_TRUNK", "HARDWARE_TELL", "NOTHING_STAGED"},
			"headroom": map[string]any{
				"soft_pct": 80.0,
				"hard_pct": 95.0,
			},
		},
		"routing": map[string]any{
			"anchor_account": "july6-netra",
			"tiers":          map[string]any{"T0": "frontier", "T2": "docs"},
		},
	}
}

// repoPath resolves a path relative to the repository root by walking up from the test's
// working directory (the package dir under `go test`) until it finds go.mod. This lets the
// test read a real file in the tree without hard-coding an absolute path.
func repoPath(t *testing.T, parts ...string) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return filepath.Join(append([]string{dir}, parts...)...)
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("repoPath: go.mod not found walking up from working dir")
		}
		dir = parent
	}
}

// TestScorecard runs the scorecard over the three real families and asserts the
// both-directions honesty property: at least one family where TOON WINS (strictly fewer
// tokens) AND at least one where it does NOT (tie/loss). A report that only ever shows wins
// is the silent-truncation anti-pattern issue #3068 rejects, and this assertion FAILS it.
// It also checks the two anchor cases: the uniform index array FIRES and the nested config
// SKIPS — the concrete "where it helps / where it hurts" the scorecard exists to witness.
func TestScorecard(t *testing.T) {
	fams := realFamilies(t)
	if len(fams) < 3 {
		t.Fatalf("want >= 3 real families, got %d", len(fams))
	}
	rep := Scorecard(fams)

	// The -v-friendly dump: the exact table pasted into docs/toon-scorecard/README.md.
	t.Logf("\n%s", rep.String())

	// Honesty gate — both directions must be present, each with real numbers.
	if rep.Wins < 1 {
		t.Errorf("both-directions: no family where TOON wins (strictly fewer tokens); the happy path is not being measured")
	}
	if rep.Losses < 1 {
		t.Errorf("both-directions: no family where TOON loses/ties; a scorecard that only shows wins is the silent-truncation anti-pattern #3068 rejects")
	}

	byName := map[string]FamilyResult{}
	for _, f := range rep.Families {
		byName[f.Name] = f
	}

	// Anchor 1 (where it helps): the uniform flat index array must WIN and FIRE.
	leaf := byName["index_leaves (uniform flat)"]
	if !leaf.Won() {
		t.Errorf("index_leaves: expected a TOON win (json=%d toon=%d delta=%.1f%%)", leaf.JSONTokens, leaf.TOONTokens, leaf.DeltaPct)
	}
	if !leaf.Fire {
		t.Errorf("index_leaves: expected the gate to FIRE, got %s", leaf.Verdict)
	}

	// Anchor 2 (where it hurts): the nested config must NOT win, and the gate must SKIP it.
	cfg := byName["nested_config (mixed object)"]
	if cfg.Won() {
		t.Errorf("nested_config: expected TOON to lose/tie, but it won (json=%d toon=%d delta=%.1f%%)", cfg.JSONTokens, cfg.TOONTokens, cfg.DeltaPct)
	}
	if cfg.Fire {
		t.Errorf("nested_config: expected the gate to SKIP, but it fired")
	}

	// The accuracy half is honestly stubbed on every row — never silently a number.
	for _, f := range rep.Families {
		if f.AccuracyNote != AccuracyNotYet {
			t.Errorf("%s: accuracy note must be the honest %q placeholder, got %q", f.Name, AccuracyNotYet, f.AccuracyNote)
		}
	}
}
